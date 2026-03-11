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
        x1 = x1 * torch.sigmoid(x1)
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
    
def scaled_dot_product_attention(
    keys: torch.Tensor, 
    queries: torch.Tensor, 
    values: torch.Tensor, 
    relative_position_bias: torch.Tensor=None,
    mask: torch.Tensor=None) -> torch.Tensor:

    scores = einsum(queries, keys, "... n d_k, ... m d_k -> ... n m") / math.sqrt(keys.shape[-1])
    if mask is not None:
        # scores = torch.where(mask, scores, torch.tensor(float('-inf')))
        scores = mask + scores
    if relative_position_bias is not None:
        print(scores.shape, relative_position_bias.shape)
        scores = scores + relative_position_bias
    attention_weights = F.softmax(scores, dim=-1)
    attention = einsum(attention_weights, values, "... n m, ... m d_v -> ... n d_v")
    # mask = torch.where(mask, 0, torch.tensor(float('-inf')))
    return attention


class WindowMultiheadSelfAttention(torch.nn.Module):
    """
    Compute multi-head self-attention within non-overlapping windows of the input feature map.
    """
    def __init__(self, d_model: int, num_heads: int, window_size: int, patch_size: int):
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
        self.relative_position_index = self.compute_relative_position_index(window_size)
        self.o_proj = nn.Linear(d_model, num_heads * self.d_v)

    def forward(self, x: torch.Tensor, shift: bool = False) -> torch.Tensor:
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        relative_position_bias = rearrange(self.relative_position_bias_table[rearrange(self.relative_position_index, "w h -> (w h)")], 
                                "(w h) c -> 1 c w h", 
                                c=self.num_heads, w=self.window_size ** 2, h=self.window_size ** 2)
        if shift:
            attn_mask = self.create_attention_mask(height=H, width=W, window_size=self.window_size, shift_size=self.window_size//2).to(x.device)
            # (num_windows, ws*ws, ws*ws) -> (B*num_windows, 1, ws*ws, ws*ws)
            attn_mask = repeat(attn_mask, "nw s1 s2 -> (b nw) 1 s1 s2", b=B)
            x = rearrange(x, "b (h w) c -> b h w c", h=H, w=W)
            x = torch.roll(x, shifts=(-self.window_size // 2, -self.window_size // 2), dims=(1, 2))
            x = rearrange(x, "b (h wsh) (w wsw) c -> (b h w) (wsh wsw) c", wsh=self.window_size, wsw=self.window_size)
        else:
            x = rearrange(x, "b (h wsh w wsw) c -> (b h w) (wsh wsw) c", 
                      wsh=self.window_size, wsw=self.window_size, h=H//self.window_size, w=W//self.window_size)
        Q = rearrange(self.q_proj(x), "... seq (num_heads d_q) -> ... num_heads seq d_q", num_heads=self.num_heads)
        K = rearrange(self.k_proj(x), "... seq (num_heads d_k) -> ... num_heads seq d_k", num_heads=self.num_heads)
        V = rearrange(self.v_proj(x), "... seq (num_heads d_v) -> ... num_heads seq d_v", num_heads=self.num_heads)
        attention = scaled_dot_product_attention(Q, K, V, relative_position_bias=relative_position_bias, mask=attn_mask if shift else None)
        attention = rearrange(attention, "... num_heads seq d_v -> ... seq (num_heads d_v)")
        output = self.o_proj(attention)
        return output
    
    def create_attention_mask(self, height, width, window_size, shift_size):
        # Number each window, roll, then compare within new windows
        row = repeat(torch.arange(height), "h -> h w", w=width)
        col = repeat(torch.arange(width), "w -> h w", h=height)
        window_ids = (row // window_size) * (width // window_size) + col // window_size
        window_ids = torch.roll(window_ids, shifts=(-shift_size, -shift_size), dims=(0, 1))
        window_ids = rearrange(window_ids, "(h wsh) (w wsw) -> (h w) (wsh wsw)", wsh=window_size, wsw=window_size)
        mask = rearrange(window_ids, "num area -> num area 1") != rearrange(window_ids, "num area -> num 1 area")
        attn_mask = mask.float() * (-100.0)
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
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.rmsnorm1 = nn.RMSNorm(d_model)
        self.rmsnorm2 = nn.RMSNorm(d_model)
        self.multihead_self_att = MultiheadSelfAttention(d_model, num_heads)
        self.swiglu = SwiGLU(d_model, d_ff)

    def forward(self, x):
        x = x + self.multihead_self_att(self.rmsnorm1(x))
        x = x + self.swiglu(self.rmsnorm2(x))
        return x

class SwinTransformer(torch.nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        window_size,
        in_channels,
        d_model,
        num_heads,
        d_ff,
        num_classes,
        num_layers,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.window_size = window_size
        self.patch_size = patch_size
        self.patch_dim = in_channels * patch_size * patch_size
        self.num_patches_per_window = (window_size // patch_size) ** 2
        self.num_windows = (image_size // window_size) ** 2

        self.proj = nn.Linear(self.patch_dim, d_model)
        self.positional_encoding = nn.Parameter(
            torch.nn.init.trunc_normal_(torch.empty(1, self.num_patches_per_window + 1, d_model))
        )
        self.transformer_blocks = torch.nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )
        self.rmsnorm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)


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
        batch_size = x.shape[0]
        cls_tokens = repeat(self.class_token, "1 1 d -> b 1 d", b=batch_size)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.positional_encoding[:, :x.shape[1]]

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        x = self.rmsnorm(x)
        cls_rep = x[:, 0]
        logits = self.head(cls_rep)
        return logits

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # x = torch.randn(8, 3, 32, 32, device=device)
    # transformer = SwinTransformer(
    #     image_size=32,
    #     patch_size=4,
    #     in_channels=3,
    #     d_model=128,
    #     num_heads=8,
    #     d_ff=256,
    #     num_classes=100,
    #     num_layers=4,
    # ).to(device)
    # y = transformer(x)
    # print(y.shape)
    # test window attention with shift window
    x = torch.randn(8, 64, 128, device=device)
    window_attention = WindowMultiheadSelfAttention(d_model=128, num_heads=8, window_size=4, patch_size=4).to(device)
    y = window_attention(x, shift=True)
    print(y.shape)
    