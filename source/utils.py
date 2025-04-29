import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.utils import make_grid
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

class SquarePad(nn.Module):
    def __init__(self, value = 0.0):
        super().__init__()
        
        self.value = 0.0
    def forward(self, img):
        if img.shape[1] == img.shape[2]:
            return img
        elif img.shape[1] < img.shape[2]:
            diff = img.shape[2] - img.shape[1]
            return F.pad(img, (0, 0, diff // 2, diff - (diff // 2), 0, 0), value = self.value)
        elif img.shape[1] > img.shape[2]:
            diff = img.shape[1] - img.shape[2]
            return F.pad(img, (diff // 2, diff - (diff // 2), 0, 0, 0, 0), value = self.value)

class Cosine(nn.Module):
    def forward(self, x):
        return torch.cos(x)

def random_mask(batch, p = 0.5):
    mask = (torch.rand(batch.shape[0], 1, batch.shape[2], batch.shape[3]) >= p).bool().to(batch.device)
    return batch * mask, mask

def patch_mask(shape, device):
    row, column = torch.arange(shape[3]).to(device), torch.arange(shape[2]).to(device)
    lower_bound, upper_bound = torch.randint(0, shape[3], (shape[0],)).view(-1, 1, 1, 1).to(device), torch.randint(0, shape[3], (shape[0],)).view(-1, 1, 1, 1).to(device)
    lower_bound, upper_bound = torch.min(lower_bound, upper_bound), torch.max(lower_bound, upper_bound) + 1
    mask_row = ((lower_bound <= row.view(1, 1, 1, -1)) * (row.view(1, 1, 1, -1) < upper_bound))
    lower_bound, upper_bound = torch.randint(0, shape[2], (shape[0],)).view(-1, 1, 1, 1).to(device), torch.randint(0, shape[2], (shape[0],)).view(-1, 1, 1, 1).to(device)
    lower_bound, upper_bound = torch.min(lower_bound, upper_bound), torch.max(lower_bound, upper_bound) + 1
    mask_column = ((lower_bound <= column.view(1, 1, -1, 1)) * (column.view(1, 1, -1, 1) < upper_bound))
    return (mask_row * mask_column)

def cutout_mask(batch, n_patches = 3):
    mask = torch.zeros(batch.shape[0], 1, batch.shape[2], batch.shape[3]).bool().to(batch.device)
    for i in range(n_patches):
        mask = mask + patch_mask(batch.shape, batch.device)
    mask = ~mask
    return batch * mask, mask

def save_ssrgan(ssrgan, directory, model_name):
    torch.save(ssrgan.models.generator.state_dict(), f"{directory}/SSRGAN_generator_{model_name}.pt")
    torch.save(ssrgan.models.discriminator.state_dict(), f"{directory}/SSRGAN_discriminator_{model_name}.pt")

def show_restored(generator, batch, param_grid = torch.linspace(0, 9, 10), how = "random"):
    if how == "random":
        mask_function = random_mask
    elif how == "cutout":
        mask_function = cutout_mask
    device = generator.parameters().__next__().device
    with torch.no_grad():
        generator.eval()
        fig, axs = plt.subplots(batch.shape[0] * 2, len(param_grid) + 1, figsize = (5 * (len(param_grid) + 2), 5 * batch.shape[0]))
        for i in range(batch.shape[0]):
            image = batch[i].unsqueeze(0).to(device)
            axs[i * 2][0].imshow(t_invnormalize(image[0].permute(1, 2, 0).cpu().detach()))
            axs[i * 2][0].axis("off")
            axs[i * 2 + 1][0].axis("off")
            for j, param in enumerate(param_grid):
                image_masked, _ = mask_function(image, param)
                image_rec = generator(image_masked)
                axs[i * 2][j + 1].imshow(t_invnormalize(image_masked[0].permute(1, 2, 0).cpu().detach()))
                axs[i * 2 + 1][j + 1].imshow(t_invnormalize(image_rec[0].permute(1, 2, 0).cpu().detach()))
                axs[i * 2][j + 1].axis("off")
                axs[i * 2 + 1][j + 1].axis("off")
    plt.tight_layout()
    plt.savefig("test.png")

t_normalize = lambda x: x * 2 - 1
t_invnormalize = lambda x: (x + 1) / 2

def show_dataset(dataset):
    images = torch.stack([dataset[(i * 1000003) % len(dataset)] for i in range(16)])
    grid = make_grid(t_invnormalize(images), nrow = 4)
    plt.figure(figsize = (20, 20))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0))
