import torch
from torch import nn
from utils import Cosine

class ResBlock(nn.Module):
    def __init__(self, in_channels = 64, out_channels = 64):
        super().__init__()

        self.fw = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features = out_channels),
            nn.PReLU(),
            nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features = out_channels),
            nn.PReLU(),
        )

        if in_channels == out_channels:
            self.skip = lambda x : x
        else:
            self.skip = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, stride = 1, padding = 0, bias = False)

    def forward(self, x):
        return self.fw(x) + self.skip(x)

class SSRGenerator(nn.Module):
    def __init__(self, model_size = 64):
        super().__init__()

        self.model_size = model_size

        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = model_size, kernel_size = 9, stride = 1, padding = 4),
            nn.PReLU(),
        )

        self.res_blocks = nn.Sequential(
            ResBlock(in_channels = model_size, out_channels = model_size),
            ResBlock(in_channels = model_size, out_channels = model_size),
            ResBlock(in_channels = model_size, out_channels = model_size),
            ResBlock(in_channels = model_size, out_channels = model_size),
            ResBlock(in_channels = model_size, out_channels = model_size),
            ResBlock(in_channels = model_size, out_channels = model_size),
        )

        self.post_rb_conv = nn.Sequential(
            nn.Conv2d(in_channels = model_size, out_channels = model_size, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features = model_size),
        )

        self.upsampling = nn.Sequential(
            nn.Conv2d(in_channels = model_size, out_channels = model_size, kernel_size = 3, stride = 1, padding = 1),
            nn.PixelShuffle(upscale_factor = 2),
            nn.PReLU(),
            nn.Conv2d(in_channels = model_size // 4, out_channels = model_size // 4, kernel_size = 3, stride = 1, padding = 1),
            nn.PixelShuffle(upscale_factor = 2),
            nn.PReLU(),
        )

        self.final_conv = nn.Conv2d(in_channels = model_size // 16, out_channels = 3, kernel_size = 9, stride = 4, padding = 4)

    def forward(self, x):
        x = self.initial_conv(x)
        r = x
        x = self.res_blocks(x)
        x = self.post_rb_conv(x) + r
        x = self.upsampling(x)
        x = self.final_conv(x)
        return x

class ModifiedSSRGenerator(nn.Module):
    def __init__(self, model_size = 64, cutout = False):
        super().__init__()

        self.model_size = model_size
        self.cutout = False

        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels = 3 + cutout, out_channels = model_size, kernel_size = 9, stride = 1, padding = 4),
            nn.PReLU(),
        )

        self.p_encodings = nn.ModuleList([self._make_p_enc(dim_out = model_size) for i in range(8)])
        self.p_encodings.append(self._make_p_enc(dim_out = model_size // 4))
        self.p_encodings.append(self._make_p_enc(dim_out = model_size // 16))

        self.res_blocks = nn.ModuleList([ResBlock(in_channels = model_size, out_channels = model_size) for i in range(6)])

        self.post_rb_conv = nn.Sequential(
            nn.Conv2d(in_channels = model_size, out_channels = model_size, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features = model_size),
        )

        self.upsampling = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels = model_size, out_channels = model_size, kernel_size = 3, stride = 1, padding = 1),
                nn.PixelShuffle(upscale_factor = 2),
                nn.PReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(in_channels = model_size // 4, out_channels = model_size // 4, kernel_size = 3, stride = 1, padding = 1),
                nn.PixelShuffle(upscale_factor = 2),
                nn.PReLU(),
            )
        ])

        self.final_conv = nn.Conv2d(in_channels = model_size // 16, out_channels = 3, kernel_size = 9, stride = 4, padding = 4)


    def _make_p_enc(self, dim_out):
        return nn.Sequential(
            nn.Linear(in_features = 1, out_features = self.model_size),
            Cosine(),
            nn.Linear(in_features = self.model_size, out_features = self.model_size),
            nn.SiLU(),
            nn.Linear(in_features = self.model_size, out_features = dim_out),
        )
    
    def forward(self, x):
        x_pre_res = x
        if self.cutout:
            x[torch.isclose(x, torch.zeros_like(x))] = (torch.rand_like(x).to(x.device) * 2 - 1) * 0.2
            x = torch.cat([x, torch.isclose(x, torch.zeros_like(x)) * (torch.rand_like(x) * 2 - 1).to(x.device)], axis = 1)
        p_approx = torch.isclose(x, torch.zeros_like(x)).float().view(x.shape[0], -1).mean(axis = 1).view(-1, 1)
        x = self.initial_conv(x)
        r = x
        for i in range(6):
            x = x + self.p_encodings[i](p_approx).view(x.shape[0], -1, 1, 1)
            x = self.res_blocks[i](x)
        x = x + self.p_encodings[6](p_approx).view(x.shape[0], -1, 1, 1)
        x = self.post_rb_conv(x) + r + self.p_encodings[7](p_approx).view(x.shape[0], -1, 1, 1)
        x = self.upsampling[0](x) + self.p_encodings[8](p_approx).view(x.shape[0], -1, 1, 1)
        x = self.upsampling[1](x) + self.p_encodings[9](p_approx).view(x.shape[0], -1, 1, 1)
        x = self.final_conv(x)
        
        x[~torch.isclose(x_pre_res, torch.zeros_like(x_pre_res))] = x_pre_res[~torch.isclose(x_pre_res, torch.zeros_like(x_pre_res))]
        return x

class SSRDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(0.2),
        )

        self.blocks = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features = 64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features = 128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features = 256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features = 512),
            nn.LeakyReLU(0.2),
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels = 512, out_channels = 1, kernel_size = 4, stride = 4, padding = 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.blocks(x)
        x = self.final_conv(x)
        return x

