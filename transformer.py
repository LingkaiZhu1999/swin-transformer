import torch
from einops import rearrange, repeat, einsum
import torch.nn as nn
import torch.nn.functional as F
import math

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, dff, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.dff = dff
        self.linear1 = nn.Linear(d_model, dff)
        self.linear2 = nn.Linear(dff, d_model)
        self.linear3 = nn.Linear(d_model, dff)

    def forward(self, x):
        x1 = self.linear1(x)
        # x1 = x1 * torch.sigmoid(x1)
        x1 = F.silu(x1)
        x2 = self.linear3(x)
        x3 = x1 * x2
        x3 = self.linear2(x3)
        return x3

class PatchMerging(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(4 * d_model, 2 * d_model)
        self.rmsnorm = nn.RMSNorm(4 * d_model)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = rearrange(x, "b (h h2 w w2) c -> b (h w) (w2 h2 c)", h2=2, w2=2, h=H//2, w=W//2)
        x = self.rmsnorm(x)
        return self.linear(x)


class WindowMultiheadSelfAttention(torch.nn.Module):
    """
    Compute multi-head self-attention within non-overlapping windows of the input feature map.
    """
    def __init__(self, d_model: int, num_heads: int, window_size: int, patch_size: int, dropout: float = 0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.window_size = window_size
        self.patch_size = patch_size
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.q_proj = nn.Linear(d_model, num_heads * self.d_k)
        self.k_proj = nn.Linear(d_model, num_heads * self.d_k)
        self.v_proj = nn.Linear(d_model, num_heads * self.d_v)
        # Use einops for relative position bias table and index
        self.relative_position_bias_table = nn.Parameter(torch.nn.init.trunc_normal_(
            torch.empty((2 * window_size - 1) * (2 * window_size - 1), num_heads),
            std=0.02
        ))
        self.register_buffer("relative_position_index", self.compute_relative_position_index(window_size))
        self.o_proj = nn.Linear(d_model, num_heads * self.d_v)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, shift: bool = False) -> torch.Tensor:
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        shift_required = shift and (H > self.window_size)
        relative_position_bias = rearrange(self.relative_position_bias_table[rearrange(self.relative_position_index, "w h -> (w h)")], 
                                "(w h) c -> 1 c w h", 
                                c=self.num_heads, w=self.window_size ** 2, h=self.window_size ** 2)
        if shift_required:
            attn_mask = self.create_attention_mask(H, W, window_size=self.window_size, shift_size=self.window_size//2).to(x.device)
            # (num_windows, ws*ws, ws*ws) -> (B*num_windows, 1, ws*ws, ws*ws)
            attn_mask = repeat(attn_mask, "nw s1 s2 -> (b nw) 1 s1 s2", b=B)
            x = rearrange(x, "b (h w) c -> b h w c", h=H, w=W)
            x = torch.roll(x, shifts=(-self.window_size // 2, -self.window_size // 2), dims=(1, 2))
            x = rearrange(x, "b (h wsh) (w wsw) c -> (b h w) (wsh wsw) c", wsh=self.window_size, wsw=self.window_size)
            attn_mask_with_bias = attn_mask + relative_position_bias
        else:
            x = rearrange(x, "b (h wsh w wsw) c -> (b h w) (wsh wsw) c", 
                      wsh=self.window_size, wsw=self.window_size, h=H//self.window_size, w=W//self.window_size)
            attn_mask_with_bias = relative_position_bias
        Q = rearrange(self.q_proj(x), "... seq (num_heads d_q) -> ... num_heads seq d_q", num_heads=self.num_heads)
        K = rearrange(self.k_proj(x), "... seq (num_heads d_k) -> ... num_heads seq d_k", num_heads=self.num_heads)
        V = rearrange(self.v_proj(x), "... seq (num_heads d_v) -> ... num_heads seq d_v", num_heads=self.num_heads)
        attention = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask_with_bias)
        attention = self.attn_dropout(attention)
        attention = rearrange(attention, "... num_heads seq d_v -> ... seq (num_heads d_v)")
        output = self.o_proj(attention)
        output = self.proj_dropout(output)
        output = rearrange(output, "(b h w) (wsh wsw) c -> b (h wsh) (w wsw) c", 
                           wsh=self.window_size, wsw=self.window_size, 
                           h=H//self.window_size, w=W//self.window_size)
        if shift_required:
            output = torch.roll(output, shifts=(self.window_size // 2, self.window_size // 2), dims=(1, 2))
        return rearrange(output, "b h w c -> b (h w) c")
    
    def create_attention_mask(self, feat_height, feat_width, window_size, shift_size):
        img_mask = torch.zeros((feat_height, feat_width))
        
        # Create regions that correspond to boundaries AFTER shifting
        h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
        w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
        
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[h, w] = cnt
                cnt += 1
                
        # Group into windows and subtract to isolate disconnected sub-regions
        mask_windows = rearrange(img_mask, "(h wsh) (w wsw) -> (h w) (wsh wsw)", wsh=window_size, wsw=window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask
    
    def compute_relative_position_index(self, window_size: int):
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing="ij"
        ))  # (2, window_size, window_size)
        coords_flat = rearrange(coords, 'c h w -> c (h w)')  # (2, window_size*window_size)
        rel_coords = rearrange(coords_flat, "c ws -> c ws 1") - rearrange(coords_flat, "c ws -> c 1 ws")  # (2, window_size*window_size, window_size*window_size)
        rel_coords = rearrange(rel_coords, 'c i j -> i j c')  # (window_size*window_size, window_size*window_size, 2)
        rel_coords[..., 0] += window_size - 1
        rel_coords[..., 1] += window_size - 1
        rel_coords[..., 0] *= 2 * window_size - 1
        rel_pos_index = rel_coords.sum(-1)  # (window_size*window_size, window_size*window_size)
        return rel_pos_index

class TransformerBlock(torch.nn.Module):
    "Two successive Swin Transformer blocks"
    def __init__(self, d_model, num_heads, d_ff, window_size=4, patch_size=2, shift=False, dropout=0.0):
        super().__init__()
        self.shift = shift
        self.rmsnorm1 = nn.RMSNorm(d_model)
        self.rmsnorm2 = nn.RMSNorm(d_model)
        self.multihead_self_att = WindowMultiheadSelfAttention(d_model, num_heads, window_size, patch_size, dropout=dropout)
        self.swiglu = SwiGLU(d_model, d_ff)

    def forward(self, x):
        x = x + self.multihead_self_att(self.rmsnorm1(x), shift=self.shift)
        x = x + self.swiglu(self.rmsnorm2(x))
        return x

class SwinTransformer(torch.nn.Module):
    def __init__(
        self,
       image_size,
        patch_size,
        window_size,
        in_channels,
        embed_dim,        # Base d_model for Stage 1
        depths,           # List of ints: number of blocks per stage (e.g., [2, 2, 6, 2])
        num_heads,        # List of ints: number of heads per stage (e.g., [4, 8, 16, 32])
        d_ff_ratio=4,     # Multiplier to calculate d_ff (SwiGLU hidden dim)
        num_classes=100,
        dropout=0.0,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if len(depths) != len(num_heads):
            raise ValueError("depths and num_heads must have the same length (number of stages)")
        self.num_stages = len(depths)
        self.window_size = window_size
        self.patch_size = patch_size
        self.patch_dim = in_channels * patch_size * patch_size

        self.proj = nn.Linear(self.patch_dim, embed_dim)
        self.initial_rmsnorm = nn.RMSNorm(embed_dim)
        # self.positional_encoding = nn.Parameter(
        #     torch.nn.init.trunc_normal_(torch.empty(1, self.num_patches_per_window + 1, d_model))
        # )
        # self.shifted_transformer_blocks = torch.nn.ModuleList(
        #     [TransformerBlock(d_model, num_heads, d_ff, window_size, patch_size, shift=(i % 2 != 0)) for i in range(num_layers)]
        # )

        self.stages = nn.ModuleList()
        current_dim = embed_dim
        
        for i_stage in range(self.num_stages):
            # Create blocks for the current stage
            blocks = nn.ModuleList([
                TransformerBlock(
                    d_model=current_dim,
                    num_heads=num_heads[i_stage],
                    d_ff=current_dim * d_ff_ratio,
                    window_size=window_size,
                    patch_size=patch_size,
                    shift=(i % 2 != 0), # Alternate shift: False, True, False, True...
                    dropout=dropout
                ) for i in range(depths[i_stage])
            ])
            
            # Create PatchMerging at the end of the stage (except for the last stage)
            if i_stage < self.num_stages - 1:
                downsample = PatchMerging(d_model=current_dim)
                current_dim = current_dim * 2 # Keep track of the doubled dimension
            else:
                downsample = nn.Identity()
                
            self.stages.append(nn.ModuleDict({
                'blocks': blocks,
                'downsample': downsample
            }))

        self.rmsnorm = nn.RMSNorm(current_dim)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(current_dim, num_classes)


    def forward(self, x):
        if x.dim() == 4:
            x = rearrange(
                x,
                "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                ph=self.patch_size,
                pw=self.patch_size,
            )
        elif x.dim() != 3:
            raise ValueError("Input x must be [B, C, H, W] or [B, N, patch_dim]")

        x = self.proj(x)
        x = self.initial_rmsnorm(x)

        for stage in self.stages:
            for block in stage['blocks']:
                x = block(x)
            x = stage['downsample'](x)

        x = self.rmsnorm(x)
        x = x.mean(dim=1)  # Global average pooling
        logits = self.head(x)
        return logits

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(8, 3, 32, 32, device=device)
    transformer = SwinTransformer(
        image_size=32,
        patch_size=2,
        window_size=2,
        in_channels=3,
        embed_dim=128,
        num_heads=[4, 8],
        d_ff_ratio=4,
        num_classes=100,
        depths=[2, 2],
    ).to(device)
    y = transformer(x)
    print(y.shape)
    