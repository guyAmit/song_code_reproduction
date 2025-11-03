import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

MIN_NUM_PATCHES = 31

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None, return_map = False):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max
        #embed()
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        if return_map:
            return out, attn
        else:
            return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
            
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x
    
    def forward_attention(self, x, mask=None):
        attn_list = [] 
        for attn, ff, in self.layers:
            pre = attn.fn.norm(x)
            pre, attn_map = attn.fn.fn(pre, return_map=True)
            x = x + pre
            x = ff(x)
            attn_list.append(attn_map.detach())
        return x, attn_list

    
class ViT_Encoder_Decoder(nn.Module):
    def __init__(self, image_size=128,
                 patch_size=16, dim=384, depth=12, heads=6, mlp_dim=384*4,
                 in_channels = 3, out_channels=1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        in_patch_dim = in_channels * patch_size ** 2 # 768
        out_patch_dim = out_channels*patch_size ** 2

        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'

        self.patch_size = patch_size
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.patch_to_embedding = nn.Linear(in_patch_dim, dim) # 768 to 384
        # print(f'patch_to_embedding: {self.patch_to_embedding}')
        self.embedding_to_patch = nn.Linear(dim, out_patch_dim) 
        # print(f'embedding_to_patch: {self.embedding_to_patch}')
        self.dropout = nn.Dropout(emb_dropout)
        self.to_latent = nn.Identity()
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)


    def forward(self, img):
        p = self.patch_size
        f = self.image_size // self.patch_size
        # print(f'Input shape: {img.shape}')
        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        # print(f'Shape after rearrange: {x.shape}')
        x = self.patch_to_embedding(x)
        # print(f'Shape after patch_to_embedding: {x.shape}')
        b, n, _ = x.shape
        
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)
        x = self.transformer(x, None)
        
        recon_out = self.embedding_to_patch(x)
        # print(f'Shape after embedding_to_patch: {recon_out.shape}') 
        recon_out = rearrange(recon_out,
                              'b (f1 f2) (c p1 p2) -> b c (f1 p1) (f2 p2)',
                              p1=p, p2 =p , c=self.out_channels, f1 = f, f2 = f)
        # print(f'Shape after final rearrange: {recon_out.shape}')  
        




        return recon_out
    
    def attention_map(self, img):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x, attn_map = self.transformer.forward_attention(x, None)

        x = x.mean(dim = 1)

        x = self.to_latent(x)
        emb = self.mlp_head(x)
        return emb, attn_map
    
    