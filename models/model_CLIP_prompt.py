from collections import OrderedDict 
from typing import Tuple, Union 
import numpy as np
import torch 
import torch.nn.functional as F 
from torch import nn 
from .transformer import VisionTransformer, Transformer, LayerNorm, MultiheadAttention 

class PromptTokenEmbedding(nn.Module):
    def __init__(self, embed_dim, num_prompt_tokens):
        super().__init__()
        self.prompt_tokens = nn.Parameter(torch.randn(1, num_prompt_tokens, embed_dim)) 
        self.projector = nn.Linear(num_prompt_tokens, 1)  
        self.norm = nn.LayerNorm(embed_dim)
        self.fusion_proj = nn.Linear(2 * embed_dim, embed_dim)  

    def forward(self, image_features):
        B, C = image_features.shape  # [B, C]

        # 1. Prompt 복제
        prompt = self.prompt_tokens.expand(B, -1, -1)  # [B, P, C]

        # 2. prompt를 FC로 압축
        prompt = prompt.permute(0, 2, 1)  # [B, C, P]
        prompt_summary = self.projector(prompt).squeeze(-1)  # [B, C]

        # 3. 정규화
        prompt_summary = self.norm(prompt_summary)  # [B, C]

        # 4. Concat 후 projection으로 다시 C로 압축
        fused_features = torch.cat([image_features, prompt_summary], dim=1)  # [B, 2C]
        fused_features = self.fusion_proj(fused_features)  # [B, C]

        return fused_features
    
class CLIP(nn.Module): 
    def __init__(self,
                 embed_dim: int, 
                 # vision
                 image_resolution: int, 
                 vision_layers: Union[Tuple[int, int, int, int], int], 
                 vision_width: int, 
                 vision_patch_size: int, 
                 # text
                 context_length: int, 
                 vocab_size: int, 
                 transformer_width: int, 
                 transformer_heads: int, 
                 transformer_layers: int, 
                 deep_prompt_len: int, 
                 total_d_layer_len: int, 
                 num_prompt_tokens = 3 
                 ): 
        super().__init__() 

        self.context_length = context_length 


        vision_heads = vision_width // 64 
        self.visual = VisionTransformer( 
            input_resolution=image_resolution, 
            patch_size=vision_patch_size, 
            width=vision_width, 
            layers=vision_layers, 
            heads=vision_heads, 
            output_dim=embed_dim 
        )

        self.transformer = Transformer(
            width=transformer_width, 
            layers=transformer_layers, 
            heads=transformer_heads, 
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size 
        self.token_embedding = nn.Embedding(vocab_size, transformer_width) 
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width)) 
        self.ln_final = LayerNorm(transformer_width) 
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim)) 
        self.prompt_text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim)) 
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) 

        self.initialize_parameters() 

        self.num_layers = transformer_layers    
        self.total_d_layer = total_d_layer_len
        if self.total_d_layer != 0:
            assert self.total_d_layer == transformer_layers - 1 
        self.num_tokens = deep_prompt_len 
        self.prompt_dim = transformer_width 

        self._init_prompt(self.num_tokens, self.prompt_dim, self.total_d_layer) 

       ####################################################################################################################
        self.prompt_token_embedding = PromptTokenEmbedding(embed_dim, num_prompt_tokens) 
       ####################################################################################################################
 
    def _init_prompt(self, num_tokens, prompt_dim, total_d_layer):
        val = 1 
        if total_d_layer >= 0: 
            self.prompt_embeddings = nn.Parameter(torch.zeros(1, num_tokens, prompt_dim)) 
            
            nn.init.uniform_(self.prompt_embeddings.data, -val, val) 

            if total_d_layer > 0:  
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(total_d_layer, num_tokens, prompt_dim)) 
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

            self.prompt_proj = nn.Linear(prompt_dim, prompt_dim) 
            nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode='fan_out') 
            self.prompt_norm = LayerNorm(prompt_dim, eps=1e-6)
            self.prompt_dropout = nn.Dropout(0.1) 

        else: 
            self.deep_prompt_embeddings = nn.Parameter(torch.zeros(abs(total_d_layer), num_tokens, prompt_dim)) 
            nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val) 
            self.prompt_proj = nn.Linear(prompt_dim, prompt_dim) 
            nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode='fan_out') 
            self.prompt_norm = LayerNorm(prompt_dim, eps=1e-6) 
            self.prompt_dropout = nn.Dropout(0.1) 
          
    def initialize_parameters(self): 
        nn.init.normal_(self.token_embedding.weight, std=0.02) 
        nn.init.normal_(self.positional_embedding, std=0.01) 

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5) 
        attn_std = self.transformer.width ** -0.5 
        fc_std = (2 * self.transformer.width) ** -0.5 
        for block in self.transformer.resblocks: 
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std) 
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std) 
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std) 
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std) 

        if self.text_projection is not None: 
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5) 
            nn.init.normal_(self.prompt_text_projection, std=self.transformer.width ** -0.5) 
       
    def build_attention_mask(self): 
        mask = torch.empty(self.context_length, self.context_length) 
        mask.fill_(float("-inf")) 
        mask.triu_(1) 
        return mask 

    @property 
    def dtype(self): 
        return self.visual.conv1.weight.dtype 

    def encode_image(self, image, out_layers): 
        return self.visual(image, out_layers)

    def encode_text(self, text, visual_feature, out_layers = [2,5,8,12]): 
    
        pos_x, pos_y = torch.where(text == 49408)
       
        x = self.token_embedding(text).type(self.dtype)  
        
        N, L, D = x.shape

        text_feature_list = [] 
        for i in range(visual_feature.shape[0]): 
            x_new = torch.zeros_like(x).to(x.device) 
            for j in range(x.shape[0]): 
                x_new[j, :, :] = torch.cat([
                    x[j, 0:pos_y[j], :], 
                    visual_feature[i,:,:], 
                    x[j, (pos_y[j]+1):(self.context_length - visual_feature.shape[1] + 1)]
                ], dim = 0).unsqueeze(0)
              
            x_new = x_new + self.positional_embedding.type(self.dtype) 

            if self.total_d_layer > 0: # self.total_d_layer > 0: Deep Prompting 사용 여부
                # concat prompt
                x_new = torch.cat(( 
                    x_new[:, :1, :],
                        self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(N, -1, -1)), 
                        x_new[:, 1:-self.num_tokens, :]
                    ), dim=1)

            x_new = x_new.permute(1,0,2) 
           
            features = []
            attns = []
            if self.total_d_layer == 0: #shallow
                x_new, attns, tokens = self.transformer(x_new) 
            elif self.total_d_layer > 0: # deep
               x_new, features, attns = self.forward_deep_prompt(x_new, features,attns,out_layers) 
            else:
                AttributeError('Input correct total_d_layer')
           

            x_new = x_new.permute(1, 0, 2)  
            x_new = self.ln_final(x_new).type(self.dtype) 

            x_new = x_new[torch.arange(x_new.shape[0]), torch.where(text == 49407)[1] + visual_feature.shape[1] - 1] @ self.text_projection  
            
            x_new = x_new / x_new.norm(dim=-1, keepdim=True)
            x_new = x_new.mean(dim = 0, keepdim = True)
            x_new = x_new / x_new.norm(dim=-1, keepdim=True)
            
            text_feature_list.append(x_new) 
            
        result = torch.stack(text_feature_list, dim = 0) 
        return result 
    
    def forward_deep_prompt(self, embedding_output, features,attns, out_layers,out_last=False): 
        N,B = embedding_output.shape[0], embedding_output.shape[1] 
        
        for i in range(self.num_layers): 
            if i == 0: 
                hidden_states, attn = self.transformer.resblocks[i](embedding_output) 
            elif i <= self.deep_prompt_embeddings.shape[0]: 
                deep_prompt_emb = self.prompt_dropout(self.prompt_proj(self.deep_prompt_embeddings[i-1]).expand(B, -1, -1)).permute(1, 0, 2)
                hidden_states = torch.cat((
                    hidden_states[:1, :, :],
                    deep_prompt_emb,
                    hidden_states[(1+self.num_tokens):, :, :]
                ), dim=0) 
                
                hidden_states, attn = self.transformer.resblocks[i](hidden_states) 
            else: 
                hidden_states = torch.cat((
                    hidden_states[:1, :, :],
                    hidden_states[(1+self.num_tokens):, :, :]
                ), dim=0) 
                hidden_states, attn= self.transformer.resblocks[i](hidden_states) 
            if len(out_layers) > 1: 
                if (i + 1) in out_layers: 
                    attns.append(attn)
                if (i+1) in out_layers:
                    xp = hidden_states.permute(1, 0, 2) 
                    xp = torch.cat([xp[:,:1,:], xp[:, (1+self.num_tokens):, :]], dim = 1) 
                    features.append(xp.contiguous()) 
            
            if i == (self.num_layers-2): 
                before_last_feats = self.prompt_norm(hidden_states) 

        hidden_states_new = hidden_states.permute(1, 0, 2) 
        hidden_states_new = torch.cat([hidden_states_new[:,:1,:], hidden_states_new[:, (1+self.num_tokens):, :]], dim = 1) 
        hidden_states = hidden_states_new.permute(1, 0, 2) 
        encoded = self.prompt_norm(hidden_states)
        if out_last:
            return before_last_feats 
        else:
            return encoded, features, attns 
       
    
    def forward(self, image, text): 
        image_features = self.encode_image(image) 
        ##########################################################################
        image_features_with_prompt = self.prompt_token_embedding(image_features) # image feature 앞에 learnable한 프롬프트 토큰을 추가
        ##########################################################################

        text_features = self.encode_text(text) 
        
        image_features_with_prompt = image_features_with_prompt / image_features_with_prompt.norm(dim=1, keepdim=True)
       
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
       
        logit_scale = self.logit_scale.exp() 
        logits_per_image = logit_scale * image_features_with_prompt @ text_features.t() 
        
        logits_per_text = logits_per_image.t()
       
        return logits_per_image, logits_per_text
         
def convert_weights(model: nn.Module): 
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l): 
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):  
            l.weight.data = l.weight.data.half() 
            if l.bias is not None: 
                l.bias.data = l.bias.data.half() 

        if isinstance(l, MultiheadAttention): 
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]: 
                tensor = getattr(l, attr) 
                if tensor is not None: 
                    tensor.data = tensor.data.half() 

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if isinstance(attr, nn.Parameter): 
                    attr.data = attr.data.half()
                elif isinstance(attr, nn.Linear):  
                    attr.weight.data = attr.weight.data.half()
                    if attr.bias is not None:
                        attr.bias.data = attr.bias.data.half()


    model.apply(_convert_weights_to_fp16) 


def build_model(state_dict: dict, deep_prompt_len  = 1, total_d_layer_len = 0): 
    
    vit = "visual.proj" in state_dict 
    
    if vit: 
        vision_width = state_dict["visual.conv1.weight"].shape[0] 
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
       
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
       
        image_resolution = vision_patch_size * grid_size 
       
    else: 
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        
        vision_layers = tuple(counts)
       
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
      
        vision_patch_size = None 
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32 

    embed_dim = state_dict["text_projection"].shape[1] 
    context_length = state_dict["positional_embedding"].shape[0] 
    vocab_size = state_dict["token_embedding.weight"].shape[0] 
    transformer_width = state_dict["ln_final.weight"].shape[0] 
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))
    

    canshu = [embed_dim,image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, deep_prompt_len, total_d_layer_len] # 모든 필요한 하이퍼파라미터를 리스트로 정리
    model = CLIP(*canshu) 
    

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model) 
    model.load_state_dict(state_dict, strict=False) 
    return model.eval(), canshu 



from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop 
from PIL import Image 
def _convert_image_to_rgb(image): 
    return image.convert("RGB") 
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC 
except ImportError:
    BICUBIC = Image.BICUBIC 
def _transform_test(n_px): 
    return Compose([
        Resize((n_px,n_px), interpolation=BICUBIC), 
        CenterCrop((n_px,n_px)), 
        _convert_image_to_rgb, 
        ToTensor(), 
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), 
    ])

def _transform_train(n_px): # 훈련 데이터 변환 함수
    return Compose([
                RandomResizedCrop(n_px, interpolation=BICUBIC), 
               
                _convert_image_to_rgb,
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])


def Load_CLIP(image_size:int , name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = False, download_root: str = None, deep_prompt_len = 1, total_d_layer_len = 0):
    model_path = name 
   
    with open(model_path, 'rb') as opened_file:         
        model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
        state_dict = None

   
    model,canshu = build_model(model.state_dict(), deep_prompt_len  = deep_prompt_len, total_d_layer_len = total_d_layer_len)
    model = model.to("cpu") 
    model.float() 
    
    state_dict = model.state_dict()
    canshu[1] = image_size 
    canshu[6] = canshu[6] + 1 
    model_new = CLIP(*canshu) 
    if image_size != model.visual.input_resolution: 
        resize_pos_embed(state_dict, model_new)


    add_embedding = True 
    if add_embedding: 
        add_word_embedding(state_dict, model_new)
    
    incompatible_keys = model_new.load_state_dict(state_dict, strict=True)
    model_new.to(device=device) 
    
    del model
    torch.cuda.empty_cache()
    
    return model_new,  _transform_train(image_size),  _transform_test(image_size)


from .utils import to_2tuple
import math
import logging

def add_word_embedding(state_dict, model, len_anotoken = 1): 
    old_word_embedding = state_dict.get('token_embedding.weight', None) 
    assert old_word_embedding is not None, f"No token embedding"

    token_size = model.vocab_size 
    assert (old_word_embedding.shape[0]+1)  == token_size 
   
    new_word_embedding = torch.cat([old_word_embedding, torch.zeros((len_anotoken, old_word_embedding.shape[1])).type(old_word_embedding.dtype)], dim = 0)

    state_dict['token_embedding.weight'] =  new_word_embedding



def resize_pos_embed(state_dict, model, interpolation: str = 'bicubic', antialias: bool = True):
    # Rescale the grid of position embeddings when loading from state_dict
    flag = 1
    old_pos_embed = state_dict.get('visual.positional_embedding', None)
    if old_pos_embed is None:
        flag = 0
        old_pos_embed = state_dict.get('visual.attnpool.positional_embedding', None)
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    logging.info('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    if flag:
        state_dict['visual.positional_embedding'] = new_pos_embed
    else:
        state_dict['visual.attnpool.positional_embedding'] = new_pos_embed

from typing import Any, Union, List
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from pkg_resources import packaging
_tokenizer = _Tokenizer()


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = Load_CLIP("./weight/ViT-L-14-336px.pt", device=device)

    image1 = torch.rand((1,3,140,140),dtype = torch.float32).to(device)   # torch.Size([1, 3, 336, 336])
    image2 = preprocess(Image.open("Dog.jpeg")).unsqueeze(0).to(device)
    print(image2.size())
    image3 = preprocess(Image.open("Cat.jpg")).unsqueeze(0).to(device)
    text = tokenize(["a object", "a cat", "a pig", "a defect", "a dog"]).to(device)
    with torch.no_grad():
        logits_per_image, logits_per_text = model(image3, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        print(probs)
        
    
