import torch
import torch.nn as nn

class NoisyA(nn.Module):
    def __init__(self,
                ori_norm,
                noise_config=None
                ):
        super().__init__()
        self.register_buffer('weight',ori_norm.weight)
        self.bias = None
        self.variance_epsilon = ori_norm.variance_epsilon
        self.use_noise = False
        self.noise_amplitude = noise_config.noise_amplitude

    def forward(self, x):
        weight = self.weight

        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x =  x.to(input_dtype) * weight

        if self.use_noise and self.noise_config:
            noise_amplitude = self.noise_config.noise_amplitude
            noise = torch.randn_like(x) * noise_amplitude
            x = x + noise

        return x

    def set_noise_state(self, use_noise: bool = False):
        self.use_noise = use_noise
