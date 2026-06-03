import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional


class BaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def print_architecture(self, verbose=False):
        name = type(self).__name__
        result = '-------------------%s---------------------\n' % name
        total_num_params = 0
        for i, (name, child) in enumerate(self.named_children()):
            if 'loss' in name:
                continue
            num_params = sum([p.numel() for p in child.parameters()])
            total_num_params += num_params
            if verbose:
                result += "%s: %3.3fM\n" % (name, (num_params / 1e6))
            for i, (name, grandchild) in enumerate(child.named_children()):
                num_params = sum([p.numel() for p in grandchild.parameters()])
                if verbose:
                    result += "\t%s: %3.3fM\n" % (name, (num_params / 1e6))
        result += '[Network %s] Total number of parameters : %.3f M\n' % (name, total_num_params / 1e6)
        result += '-----------------------------------------------\n'
        print(result)

    def set_requires_grad(self, requires_grad):
        for param in self.parameters():
            param.requires_grad = requires_grad

    def get_parameters_for_train(self):
        return self.parameters()

    def forward(self):
        raise NotImplementedError()


class BaseDiffusionModel(BaseModel):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.scheduler = FlowMatchScheduler(num_steps=1000, shift=1, sigma_min=0.0, extra_one_step=False)
        self.scheduler.set_timesteps(opt.num_train_timestep, training=True)
        self.dtype = torch.bfloat16 if opt.mixed_precision else torch.float32

    def _get_timestep(
            self,
            min_timestep: int,
            max_timestep: int,
            batch_size: int,
            num_frame: int,
            num_frame_per_block: int,
            uniform_timestep: bool = False
    ) -> torch.Tensor:
    
        """
        Randomly generate a timestep tensor based on the generator's task type. It uniformly samples a timestep
        from the range [min_timestep, max_timestep], and returns a tensor of shape [batch_size, num_frame].
        - If uniform_timestep, it will use the same timestep for all frames.
        - If not uniform_timestep, it will use a different timestep for each block.
        """
        if uniform_timestep:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, 1],
                device=self.rank,
                dtype=torch.long
            ).repeat(1, num_frame)
            return timestep

        else:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, num_frame],
                device=self.rank,
                dtype=torch.long
            )
            if self.opt.block_causal:
                timestep = timestep.reshape(
                    timestep.shape[0], -1, num_frame_per_block)
                timestep[:, :, 1:] = timestep[:, :, 0:1]
                timestep = timestep.reshape(timestep.shape[0], -1)

            return timestep

    def forward(self):
        raise NotImplementedError()


class FlowMatchScheduler:
    def __init__(self, num_steps=1000, shift=3.0, sigma_max=1.0, sigma_min=0, inverse_timesteps=False, extra_one_step=False, reverse_sigmas=False):
        self.num_steps = num_steps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.set_timesteps(num_steps)

    def set_timesteps(self, num_steps=1000, denoising_strength=1.0, training=False):
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        
        if self.extra_one_step: # 1 ~ 0
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_steps)
        
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])

        # squash function
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
            
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas

        self.timesteps = self.sigmas * self.num_steps # [1, epsilon] >> [1000, 1]

        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_steps / 2) / num_steps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (num_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing


    def step(self, model_output, timestep, sample, to_final=False):
        self.sigmas = self.sigmas.to(model_output.device)
        self.timesteps = self.timesteps.to(model_output.device)

        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma = self.sigmas[timestep_id].reshape(-1, 1)

        if to_final or (timestep_id + 1 >= len(self.timesteps)).any():
            sigma_ = 1 if (
                self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1].reshape(-1, 1)
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample

    def add_noise(self, original_samples, noise, timestep):
        """
        Diffusion forward corruption process.
        Input:
            - clean_latent: the clean latent with shape [B, d]
            - noise: the noise with shape [B, d]
            - timestep: the timestep with shape [B]
        Output: the corrupted latent with shape [B, d]
        """
        
        self.sigmas = self.sigmas.to(noise.device)
        self.timesteps = self.timesteps.to(noise.device)

        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma = self.sigmas[timestep_id].reshape(-1, 1)
        # breakpoint()
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample.type_as(noise)

    def training_target(self, sample, noise, timestep):
        target = noise - sample
        return target

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        self.linear_timesteps_weights = self.linear_timesteps_weights.to(timestep.device)
        t = timestep.reshape(-1)                     # [B*T]
        grid = self.timesteps.view(1, -1)            # [1, N]
        idx = (grid - t.view(-1, 1)).abs().argmin(dim=1)  # [M]
        w = self.linear_timesteps_weights[idx]       # [M]
        return w.view_as(timestep)                   # [B, T]

