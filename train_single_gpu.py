import os, sys, time
import argparse
import yaml

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.utils as vutils

from cond_diffusion_denoiser import CIFAR10DenoiseDataset, UNetCond, CondDDPM

class Trainer():
    def __init__(self, params):
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            torch.backends.cudnn.benchmark = True
            self.device = torch.cuda.current_device()
        else:
            self.device = torch.device('cpu')
        print("Device:", self.device)

        self.params = params

        self._setup_data()
        self._setup_model()
        self._setup_optimizer()

    def train(self):
        @torch.no_grad()
        def psnr_batch(x, y, eps=1e-8):
            mse = torch.mean((x - y) ** 2, dim=(1,2,3))
            ps = 20 * torch.log10(1.0 / torch.sqrt(mse + eps))
            return ps.mean().item()

        # --- training loop ---
        print(f"Starting training for {self.params['epochs']} epochs...")
        best_psnr = -1e9

        for ep in range(1, self.params['epochs']+1):
            self.model.train()
            running = 0.0
            for x0, y in self.train_dl:
                x0, y = x0.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                self.opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type = 'cuda', enabled=torch.cuda.is_available()):
                    loss = self.ddpm.training_loss(self.model, x0, y)  # ε-prediction MSE
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()
                running += loss.item() * x0.size(0)
            self.sched.step()
            tr_loss = running / len(self.train_dl.dataset)

            # --- validation: DDIM sample (deterministic) and PSNR ---
            self.model.eval()
            v_psnrs = []
            with torch.no_grad():
                for x0, y in self.val_dl:
                    x0, y = x0.to(self.device), y.to(self.device)
                    # 100 steps is a good starting point; raise to 200–250 for better quality
                    x_hat = self.ddpm.ddim_sample(self.model, y, shape=x0.shape[1:], eta=0.0, steps=self.params['sample_steps'])
                    v_psnrs.append(psnr_batch(x_hat, x0))
            mpsnr = float(np.mean(v_psnrs))
            if mpsnr > best_psnr:
                best_psnr = mpsnr
                torch.save({
                    'model': self.model.state_dict(), 
                    'cfg': {
                        'T': self.ddpm.T, 
                        'Cx': self.Cx
                    }
                }, self.params['save_path'])
            print(f"[{ep:03d}] train_loss={tr_loss:.4f}  val_PSNR={mpsnr:.2f}  best={best_psnr:.2f}")


    def _setup_data(self):
        augs = T.RandomHorizontalFlip(p=0.5)
        self.train_ds = CIFAR10DenoiseDataset(
            train=True, 
            fixed_sigma=None, 
            sigma_range=(self.params['train_sigma_min'], self.params['train_sigma_max']), 
            extra_transform=augs
        )
        self.val_ds = CIFAR10DenoiseDataset(
            train=False, 
            fixed_sigma=self.params['val_sigma'],
            sigma_range=(self.params['val_sigma'], self.params['val_sigma']), 
            extra_transform=None
        )

        self.train_dl = DataLoader(
            self.train_ds, 
            batch_size=self.params['batch_size'], 
            shuffle=True, 
            num_workers=self.params['num_workers'], 
            pin_memory=True
        )
        self.val_dl = DataLoader(
            self.val_ds, 
            batch_size=self.params['val_batch_size'], 
            shuffle=False, 
            num_workers=self.params['num_workers']//2, 
            pin_memory=True
        )

    def _setup_model(self):
        self.Cx = self.params['in_channels']  # CIFAR-10 is RGB so Cx=3
        self.model = UNetCond(
            in_ch=self.Cx+self.Cx, 
            base=self.params['base_channels'], 
            time_dim=self.params['time_dim'], 
            out_ch=self.Cx
        ).to(self.device)
        self.ddpm = CondDDPM(T=self.params['timesteps'], device=self.device)

    def _setup_optimizer(self):
        self.opt = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.params['lr'], 
            weight_decay=self.params['weight_decay']
        )
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.params['epochs'])
        self.scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())


def load_config(yaml_path):
    """Load parameters from YAML config file."""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    # parsers for any cmd line args
    parser = argparse.ArgumentParser()

    # YAML config file
    parser.add_argument("--yaml_config", default='./configs/default.yaml', type=str)

    # Training parameters
    parser.add_argument('--epochs', type=int, help='number of epochs')
    parser.add_argument('--batch_size', type=int, help='training batch size')
    parser.add_argument('--val_batch_size', type=int, help='validation batch size')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--weight_decay', type=float, help='weight decay')
    
    # Model parameters
    parser.add_argument('--in_channels', type=int, help='number of channels of the input images')
    parser.add_argument('--base_channels', type=int, help='base number of channels in UNet')
    parser.add_argument('--time_dim', type=int, help='time embedding dimension')
    
    # Diffusion parameters
    parser.add_argument('--timesteps', type=int, help='number of diffusion timesteps')
    parser.add_argument('--sample_steps', type=int, help='number of sampling steps for DDIM')
    
    # Data parameters
    parser.add_argument('--train_sigma_min', type=float, help='min sigma for training')
    parser.add_argument('--train_sigma_max', type=float, help='max sigma for training')
    parser.add_argument('--val_sigma', type=float, help='fixed sigma for validation')
    
    # Other parameters
    parser.add_argument('--num_workers', type=int, help='number of data loading workers')
    parser.add_argument('--save_path', type=str, help='path to save model')

    # The default values of parameters are defined by the YAML config file
    args = parser.parse_args()
    params = load_config(args.yaml_config)
    parser.set_defaults(**params)  
    # If there are command line arguments, they will overwrite the parameters defined by the YAML config
    args = parser.parse_args()

    params = vars(args)  # Overwrite params with args for simplicity
    print(f'Config parameters: {params}')
    

    print('Starting trainer')
    trainer = Trainer(params)
    trainer.train()

    print('Training complete')