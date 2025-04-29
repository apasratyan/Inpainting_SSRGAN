import torch
from torch import nn
from tqdm import tqdm
from utils import cutout_mask, random_mask

def discriminator_loss(D_r, D_f, gen_mask = None):
    bce = nn.BCELoss()
    loss = bce(D_r, torch.ones_like(D_r)) + bce(D_f, torch.rand_like(D_f) * 0.1 * (~gen_mask) + (torch.ones_like(D_f) - 0.1 * torch.rand_like(D_f)) * gen_mask)
    return loss

def generator_loss(H_sr_true, H_sr_rec, D_f):
    bce = nn.BCELoss()
    mse = nn.MSELoss()
    loss = mse(H_sr_true, H_sr_rec) + 1e-3 * bce(D_f, torch.ones_like(D_f) - 0.1 * torch.rand_like(D_f))
    return loss

def nmse(x, y):
    return (x - y).pow(2).sum() / x.pow(2).sum()

def train_epoch(models, optims, train_loader, logs, mask, device, p = None, n_patches = 3):
    models.generator.train()
    models.discriminator.train()
    for batch in tqdm(train_loader):
        batch = batch.to(device)
        if mask == "random":
            if p != None:
                batch_masked, gen_mask = random_mask(batch, p = p)
                gen_mask = torch.zeros(batch.shape[0], 1, batch.shape[2], batch.shape[3]).bool().to(batch.device)
            else:
                batch_masked, gen_mask = random_mask(batch, p = torch.rand((batch.shape[0], 1, 1, 1)))
        elif mask == "cutout":
            batch_masked, gen_mask = cutout_mask(batch, n_patches = n_patches)

        optims.generator.zero_grad()
        optims.discriminator.zero_grad()
        batch_rec = models.generator(batch_masked)
        D_r = models.discriminator(batch)
        D_f = models.discriminator(batch_rec)
        d_loss = discriminator_loss(D_r, D_f, gen_mask)
        d_loss.backward()
        optims.discriminator.step()

        optims.generator.zero_grad()
        optims.discriminator.zero_grad()
        batch_rec = models.generator(batch_masked)
        D_f = models.discriminator(batch_rec)
        g_loss = generator_loss(batch, batch_rec, D_f)
        g_loss.backward()
        optims.generator.step()

        logs.discriminator.append(d_loss.item())
        logs.generator.append(g_loss.item())

def val_epoch(generator, val_loader, logs_nmse, mask, device, p = None, n_patches = 3):
    with torch.no_grad():
        generator.eval()
        val_nmse = 0
        for batch in tqdm(val_loader):
            batch = batch.to(device)
            if mask == "random":
                if p != None:
                    batch_masked, _ = random_mask(batch, p = p)
                else:
                    p = torch.rand((batch.shape[0],))
                    batch_masked, _ = random_mask(batch, p = p.view(-1, 1, 1, 1))
                    p = None
            elif mask == "cutout":
                batch_masked, _ = cutout_mask(batch, n_patches = n_patches)
            batch_rec = generator(batch_masked)
            val_nmse += nmse(batch, batch_rec).item() * batch.shape[0]
    val_nmse /= len(val_loader.dataset)
    logs_nmse.append(val_nmse)
    return val_nmse

def train(models, optims, loaders, logs, mask, device, p = None, n_patches = 3, n_epochs = 100, schedulers = None):
    for epoch in range(n_epochs):
        print(f"epoch {epoch + 1}/{n_epochs}")
        train_epoch(models, optims, loaders.train_loader, logs, mask, device, p = p, n_patches = n_patches)
        val_nmse = val_epoch(models.generator, loaders.val_loader, logs.nmse, mask, device, p = p, n_patches = n_patches)
        print(f"validation NMSE: {val_nmse}")
        print()

        if schedulers != None:
            schedulers.generator.step()
            schedulers.discriminator.step()

def check_model_stability(generator, loader, device, p_grid):
    log = []
    for p in p_grid:
        log.append(val_epoch(generator, loader, logs, "random", device, p = p))
        print(f"p = {p}")
        print(f"nmse = {log[-1]}")
    return log
