import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from munch import Munch
import os

from source.dataset import FolderDataset
from source.utils import *
from source.train_functions import *
from source.models import *

def ssrgan_pipeline(dataset_name,
                    dataset_path,
                    save_path,
                    test_size = 0.5,
                    ssrgan_type = "vanilla",
                    mask = "random",
                    p = 0.5,
                    n_patches = 3,
                    model_size = 64,
                    loss_function = "mse",
                    n_epochs = 100,
                    optimizer = "adam",
                    initial_lr = 2e-4,
                    scheduler_step_size = 25,
                    scheduler_gamma = 0.5,
                    image_size = 128,
                    batch_size = 8):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        SquarePad(0.0),
        transforms.Resize(image_size),
        t_normalize,
    ])

    train_set = FolderDataset(image_folder = dataset_path, test = False, test_size = test_size, transforms = transform)
    val_set = FolderDataset(image_folder = dataset_path, test = True, test_size = test_size, transforms = transform)

    loaders = Munch()
    loaders.train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
    loaders.val_loader = DataLoader(val_set, batch_size = batch_size, shuffle = False)

    logs = Munch()
    logs.generator = []
    logs.discriminator = []
    logs.nmse = []

    ssrgan = Munch()

    ssrgan.models = Munch()
    ssrgan.optims = Munch()
    ssrgan.schedulers = Munch()

    if ssrgan_type == "vanilla":
        ssrgan.models.generator = SSRGenerator(model_size = model_size).to(device)
    elif ssrgan_type == "modified":
        p = None
        ssrgan.models.generator = ModifiedSSRGenerator(model_size = model_size, cutout = (mask == "cutout")).to(device)
    ssrgan.models.discriminator = SSRDiscriminator().to(device)

    if optimizer == "adam":
        ssrgan.optims.generator = optim.Adam(ssrgan.models.generator.parameters(), lr = initial_lr)
        ssrgan.optims.discriminator = optim.Adam(ssrgan.models.discriminator.parameters(), lr = initial_lr)
    elif optimizer == "sgd":
        ssrgan.optims.generator = optim.SGD(ssrgan.models.generator.parameters(), lr = initial_lr, momentum = 0.9)
        ssrgan.optims.discriminator = optim.SGD(ssrgan.models.discriminator.parameters(), lr = initial_lr, momentum = 0.9)

    ssrgan.schedulers.generator = optim.lr_scheduler.StepLR(ssrgan.optims.generator, step_size = scheduler_step_size, gamma = scheduler_gamma)
    ssrgan.schedulers.discriminator = optim.lr_scheduler.StepLR(ssrgan.optims.discriminator, step_size = scheduler_step_size, gamma = scheduler_gamma)

    train(ssrgan.models, ssrgan.optims, loaders, logs, mask, device, loss_function = loss_function, p = p, n_patches = n_patches, n_epochs = n_epochs, schedulers = ssrgan.schedulers)

    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    
    model_name = dataset_name + "_" + ssrgan_type + "_" + str(model_size) + "_" + mask
    save_ssrgan(ssrgan, save_path, model_name)

    plt.figure(figsize = (20, 10))
    plt.grid()
    plt.plot(torch.arange(len(logs.nmse)) + 1, logs.nmse, label = "validation NMSE")
    plt.xticks(torch.arange(len(logs.nmse)) + 1, size = 6)
    plt.title("validation NMSE per epoch")
    plt.xlabel("epoch")
    plt.ylabel("NMSE")
    plt.legend()
    plt.savefig(f"{save_path}/{model_name}_NMSE_log.png")

def loading_pipeline(dataset_name,
                     device,
                     ssrgan_type = "vanilla",
                     mask = "random",
                     model_size = 64):
    model_name = dataset_name + "_" + ssrgan_type + "_" + str(model_size) + "_" + mask
    ssrgan = Munch()
    if ssrgan_type == "vanilla":
        ssrgan.generator = SSRGenerator(model_size = model_size).to(device)
    elif ssrgan_type == "modified":
        p = None
        ssrgan.generator = ModifiedSSRGenerator(model_size = model_size).to(device)
    ssrgan.discriminator = SSRDiscriminator().to(device)

    ssrgan.generator.load_state_dict(torch.load(f"./trained_models/SSRGAN_generator_{model_name}.pt"))
    ssrgan.discriminator.load_state_dict(torch.load(f"./trained_models/SSRGAN_discriminator_{model_name}.pt"))
    
