import torch
import torch.nn as nn
import torch.nn.functional as F


# Loss for ViT on CIFAR-100 with label smoothing
class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim)) 
    
class PatchEmbed(nn.Module):
    """Split image into patches and then embed them."""
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=384):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return self.norm(x)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.norm = nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return self.norm(x)

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)  # Set eps to 1e-6
        self.attn = Attention(dim, num_heads=num_heads, dropout=dropout)
        self.drop_path1 = nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)  # Set eps to 1e-6
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=dropout)
        self.drop_path2 = nn.Identity()
        self.ls1 = nn.Identity()
        self.ls2 = nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

class ViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=100, embed_dim=384, depth=12, num_heads=8, mlp_ratio=4., dropout=0.):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + (img_size // patch_size) ** 2, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        self.patch_drop = nn.Identity()
        self.norm_pre = nn.Identity()
        self.blocks = nn.Sequential(*[Block(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)  # Set eps to 1e-6
        self.fc_norm = nn.Identity()
        self.head_drop = nn.Dropout(p=0.0)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.norm_pre(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x[:, 0])
        x = self.fc_norm(x)
        x = self.head_drop(x)
        x = self.head(x)
        return x

LEARNING_RATE = 0.0001
hidden = 384
mlp_hidden = 1536  
num_layers = 12
head = 8
num_classes = 100
img_size = 32
patch_size = 4
dropout = 0.0
in_chans = 3  

ViT_parameters = dict(img_size=img_size, patch_size=patch_size, 
                      in_chans=in_chans, num_classes=num_classes, 
                      embed_dim=hidden,  depth=num_layers,
                      num_heads=head, mlp_ratio=(mlp_hidden/hidden),
                      dropout=dropout)



# model = ViT(**ViT_parameters)

# optimizer = optim.AdamW(model.parameters(), lr= LEARNING_RATE)