import torch
import math
from einops import rearrange

def image_to_patches(image: torch.Tensor, patch_size: int) -> torch.Tensor:
	channels, height, width = image.shape
	if height % patch_size != 0 or width % patch_size != 0:
		raise ValueError("Image height and width must be divisible by patch_size.")

	patches = rearrange(
		image,
		"c (h ph) (w pw) -> (h w) (c ph pw)",
		ph=patch_size,
		pw=patch_size,
	)
	return patches

def learning_rate_schedule(t, lr_max, lr_min, t_warm_up, t_cos_anneal):
    if t < t_warm_up:
        return t / t_warm_up * lr_max
    elif t >= t_warm_up and t <= t_cos_anneal:
        return lr_min + 0.5 * (1 + math.cos((t - t_warm_up) / (t_cos_anneal - t_warm_up) * math.pi)) * (lr_max - lr_min)
    else:
        return lr_min