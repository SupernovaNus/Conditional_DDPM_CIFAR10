import torch
from torch.utils.data import Dataset
from torch.utils.checkpoint import checkpoint
import torchvision
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F

import os, math, random, argparse
from glob import glob
from typing import Optional, Tuple


class CIFAR10DenoiseDataset(Dataset):
    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        download: bool = True,
        sigma_range=(0.0, 50.0),
        fixed_sigma: Optional[float] = None,  # instead of float | None
        to_grayscale: bool = False,
        extra_transform: Optional[T.Compose] = None,
        seed: Optional[int] = None,
    ):
        self.train = train
        self.sigma_range = sigma_range
        self.fixed_sigma = fixed_sigma
        self.to_grayscale = to_grayscale
        self.extra_transform = extra_transform
        self.seed = seed

        base_tfms = [T.ToTensor()]
        if self.to_grayscale:
            base_tfms.insert(0, T.Grayscale(num_output_channels=1))
        self.base_transform = T.Compose(base_tfms)

        if self.seed is not None:
            random.seed(self.seed)

        self.dataset = torchvision.datasets.CIFAR10(
            root=root,
            train=self.train,
            download=download,
            transform=None  # we'll apply our own transforms to keep control of order
        )

    def __len__(self):
        return len(self.dataset)

    def _sample_sigma01(self) -> float:
        # sample sigma on [0,1] scale; input sigma_range is on [0..255]
        if self.fixed_sigma is not None:
            return float(self.fixed_sigma) / 255.0
        sigma = random.uniform(*self.sigma_range)
        return float(sigma) / 255.0

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]  # PIL image, label unused for denoising
        # (optional) data aug BEFORE noise
        if self.extra_transform is not None:
            img = self.extra_transform(img)
        # to tensor in [0,1]; (C,H,W) with C=3 or 1
        x0 = self.base_transform(img)
        sigma01 = self._sample_sigma01()
        y = (x0 + sigma01 * torch.randn_like(x0)).clamp(0.0, 1.0)
        return x0, y  # x0: clean image, y: noisy image  (C,H,W); for CIFAR10: (3, 32, 32)

# ==============================
# Diffusion helpers (cosine schedule, timesteps, etc.)
# ==============================
def cosine_beta_schedule(T, s=0.008):
    steps = T
    t = torch.linspace(0, T, steps + 1, dtype=torch.float64)
    alphas_cumprod = torch.cos(((t / T) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-5, 0.999).float()

class DiffusionSchedule:
    def __init__(self, T=1000, device='cpu'):
        self.T = T
        self.device = device
        betas = cosine_beta_schedule(T).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.], device=device), alphas_cumprod[:-1]], dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

# ==============================
# UNet building blocks with time embeddings
# ==============================
def sinusoidal_time_embedding(timesteps, dim):
    """
    timesteps: (B,) int or float in [0, T-1]
    returns: (B, dim)
    """
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(
            math.log(1.0), math.log(10000.0), half, device=device
        ) * -1.0
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb

class TimeMLP(nn.Module):
    def __init__(self, time_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(time_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )
    def forward(self, t_emb):
        return self.net(t_emb)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(self.act(self.norm1(x)))
        # FiLM-like add time conditioning
        h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)

class Down(nn.Module):
    def __init__(self, ch, time_dim):
        super().__init__()
        self.pool = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
        self.res = ResBlock(ch, ch, time_dim)
    def forward(self, x, t):
        x = self.pool(x)
        x = self.res(x, t)
        return x

class Up(nn.Module):
    def __init__(self, ch, time_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(ch, ch, 2, stride=2)
        self.res = ResBlock(ch, ch, time_dim)
    def forward(self, x, t):
        x = self.up(x)
        x = self.res(x, t)
        return x

class UNetCond(nn.Module):
    """
    Conditional UNet that takes [x_t, y] concatenated as input.
    - in_ch = C_x + C_y  (C_x==C_y==1 for grayscale, 3 for RGB)
    - predicts epsilon (same channels as clean image C_x)
    """
    def __init__(self, in_ch, base=64, time_dim=256, out_ch=1, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.time_dim = time_dim
        self.time_embed = TimeMLP(time_dim, time_dim)

        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)

        self.rb1 = ResBlock(base, base, time_dim)
        self.down1 = Down(base, time_dim)      # base -> base (H/2)
        self.rb2 = ResBlock(base, base*2, time_dim)
        self.down2 = Down(base*2, time_dim)    # base*2 -> base*2 (H/4)
        self.rb3 = ResBlock(base*2, base*4, time_dim)
        self.down3 = Down(base*4, time_dim)    # base*4 -> base*4 (H/8)

        self.mid1 = ResBlock(base*4, base*4, time_dim)
        self.mid2 = ResBlock(base*4, base*4, time_dim)

        self.up3 = Up(base*4, time_dim)
        self.rb_up3 = ResBlock(base*4, base*2, time_dim)
        self.up2 = Up(base*2, time_dim)
        self.rb_up2 = ResBlock(base*2, base, time_dim)
        self.up1 = Up(base, time_dim)
        self.rb_up1 = ResBlock(base, base, time_dim)

        self.out_conv = nn.Conv2d(base, out_ch, 3, padding=1)


    def maybe_ckpt(self, fn, *args):
        if self.use_checkpoint and self.training:
            return checkpoint(fn, *args, use_reentrant=False)
        else:
            return fn(*args)

    def forward(self, x_in, t):
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        t_emb = self.time_embed(t_emb)

        x = self.in_conv(x_in)

        # Down path
        x1 = self.maybe_ckpt(self.rb1, x, t_emb)
        x2 = self.maybe_ckpt(self.down1, x1, t_emb)
        x2 = self.maybe_ckpt(self.rb2, x2, t_emb)
        x3 = self.maybe_ckpt(self.down2, x2, t_emb)
        x3 = self.maybe_ckpt(self.rb3, x3, t_emb)
        x4 = self.maybe_ckpt(self.down3, x3, t_emb)

        # Mid
        x_mid = self.maybe_ckpt(self.mid1, x4, t_emb)
        x_mid = self.maybe_ckpt(self.mid2, x_mid, t_emb)

        # Up path
        x = self.maybe_ckpt(self.up3, x_mid, t_emb)
        x = x + F.interpolate(x3, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = self.maybe_ckpt(self.rb_up3, x, t_emb)

        x = self.maybe_ckpt(self.up2, x, t_emb)
        x = x + F.interpolate(x2, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = self.maybe_ckpt(self.rb_up2, x, t_emb)

        x = self.maybe_ckpt(self.up1, x, t_emb)
        x = x + F.interpolate(x1, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = self.maybe_ckpt(self.rb_up1, x, t_emb)

        return self.out_conv(x)


    # def forward(self, x_in, t):
    #     # t: (B,) integer or float in [0,T)
    #     t_emb = sinusoidal_time_embedding(t, self.time_dim)
    #     t_emb = self.time_embed(t_emb)

    #     x = self.in_conv(x_in)
    #     x1 = self.rb1(x, t_emb)
    #     x2 = self.down1(x1, t_emb)
    #     x2 = self.rb2(x2, t_emb)
    #     x3 = self.down2(x2, t_emb)
    #     x3 = self.rb3(x3, t_emb)
    #     x4 = self.down3(x3, t_emb)

    #     x_mid = self.mid1(x4, t_emb)
    #     x_mid = self.mid2(x_mid, t_emb)

    #     x = self.up3(x_mid, t_emb)
    #     # simple skip via resize+add (keeps code compact)
    #     x = x + F.interpolate(x3, size=x.shape[-2:], mode='bilinear', align_corners=False)
    #     x = self.rb_up3(x, t_emb)

    #     x = self.up2(x, t_emb)
    #     x = x + F.interpolate(x2, size=x.shape[-2:], mode='bilinear', align_corners=False)
    #     x = self.rb_up2(x, t_emb)

    #     x = self.up1(x, t_emb)
    #     x = x + F.interpolate(x1, size=x.shape[-2:], mode='bilinear', align_corners=False)
    #     x = self.rb_up1(x, t_emb)

    #     return self.out_conv(x)

# ==============================
# Training (ε-prediction loss)
# ==============================
class CondDDPM:
    def __init__(self, T=1000, device='cuda'):
        self.sched = DiffusionSchedule(T=T, device=device)
        self.T = T
        self.device = device

    def q_sample(self, x0, t, noise=None):
        """
        x_t = sqrt(alpha_bar_t) x0 + sqrt(1 - alpha_bar_t) eps
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sched.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_om = self.sched.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_ab * x0 + sqrt_om * noise, noise

    def training_loss(self, model, x0, y):
        """
        a) Sample t ~ Uniform({0,...,T-1})
        b) Form x_t from x0
        c) Predict eps with model([x_t, y], t)
        d) MSE loss between predicted eps and true eps

        Input to model: [x_t, y], t
        - x_t: partially noised clean image at time t
        - y: condition (externally noisy image)
        - t: diffusion timestep (scalar per sample)
        Output: predicted noise ε̂
        Loss: MSE(ε̂, ε)
        """
        B = x0.size(0)
        t = torch.randint(0, self.T, (B,), device=x0.device)
        x_t, eps = self.q_sample(x0, t)    # The clean image x0 is diffused into x_t = √ᾱₜ x₀ + √(1−ᾱₜ) ε.
        x_in = torch.cat([x_t, y], dim=1)  # Condition on measured noisy image y. x_in.shape = (B,2C,H,W)
        eps_pred = model(x_in, t)          # The model predicts noise ε (same shape as x0)
        return F.mse_loss(eps_pred, eps)

    @torch.no_grad()
    def ddim_sample(self, model, y, shape, eta=0.0, steps=50):
        """
        Deterministic DDIM sampler (eta=0) or stochastic if eta>0.
        Always conditions on y by concatenation.

        Input to model: [x_t, y], t
        - x_t: current denoising state (initially random noise x_T)
        - y: observed noisy image (fixed condition)
        - t: diffusion timestep
        Output: predicted noise ε̂
        Used iteratively to recover clean x₀ estimate (x_hat)
        """
        b = y.size(0)
        device = y.device
        C, H, W = shape
        # choose a subset of timesteps
        interval = self.T // steps
        tau = list(range(self.T-1, -1, -interval))
        x = torch.randn((b, C, H, W), device=device)   # Start from pure noise (initially: x_T)

        for i, t in enumerate(tau):
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            alpha_t = self.sched.alphas_cumprod[t]
            alpha_prev = self.sched.alphas_cumprod_prev[t]
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

            x_in = torch.cat([x, y], dim=1)
            eps_pred = model(x_in, t_batch)

            # x0_pred from eps prediction
            x0_pred = (x - sqrt_one_minus_alpha_t * eps_pred) / (sqrt_alpha_t + 1e-8)
            x0_pred = x0_pred.clamp(0.0, 1.0)

            if i == len(tau) - 1:
                x = x0_pred
                break

            # DDIM update
            alpha_prev_sqrt = torch.sqrt(alpha_prev)
            sigma_t = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_prev)
            dir_xt = torch.sqrt(alpha_prev) * (x0_pred)  # deterministic direction
            noise = torch.randn_like(x) if eta > 0 else 0.0
            x = dir_xt + sigma_t * noise

        return x.clamp(0.0, 1.0)