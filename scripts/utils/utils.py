import torch
import numpy as np


def latent_to_sdfimage(latent, decoder, size):
    x_coord = np.linspace(0, 1, size).astype(np.float32)
    y_coord = np.linspace(0, 1, size).astype(np.float32)
    xy = np.stack(np.meshgrid(x_coord, y_coord), -1).reshape(-1, 2)
    xy = torch.tensor(xy, dtype=torch.float32).to(latent.device)
    xy = xy.unsqueeze(0).expand(latent.shape[0], -1, -1)
    with torch.no_grad():
        glob_cond = torch.cat([latent.unsqueeze(1).expand(-1, xy.shape[1], -1), xy], dim=2)
        sdf = decoder(glob_cond)
        sdf_imgs = sdf.view(latent.shape[0], size, size)
    return sdf_imgs
        