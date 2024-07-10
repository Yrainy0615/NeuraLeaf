import torch
import numpy as np
from matplotlib import pyplot as plt
import trimesh
import cv2
import meshlib.mrmeshpy as mrmeshpy
import meshlib.mrmeshnumpy as mrmeshnumpy
import numpy as np
import math
from torchvision.utils import make_grid, save_image
 


def latent_to_mask(latent, decoder, size,k=None):
    x_coord = np.linspace(0, 1, size).astype(np.float32)
    y_coord = np.linspace(0, 1, size).astype(np.float32)
    xy = np.stack(np.meshgrid(x_coord, y_coord), -1).reshape(-1, 2)
    xy = torch.tensor(xy, dtype=torch.float32).to(latent.device)
    xy = xy.unsqueeze(0).expand(latent.shape[0], -1, -1)

    glob_cond = torch.cat([latent.unsqueeze(1).expand(-1, xy.shape[1], -1), xy], dim=2)
    sdf = decoder(glob_cond)
    sdf_2d = sdf.view(latent.shape[0], size, size)
    mask = sdf_to_mask(sdf_2d, k)
    # mask = torch.nn.Hardsigmoid()(sdf_2d*k)
    return 1-mask


def sdf_to_mask(sdf:torch.Tensor,k:float=1):
    mask = 1/ (1+torch.exp(-k*sdf))
    return mask


def mask_to_mesh(mask_file:str):
    dm = mrmeshpy.loadDistanceMapFromImage(mrmeshpy.Path(mask_file), 0)
    # find the boundary contour between black and white:
    polyline2 = mrmeshpy.distanceMapTo2DIsoPolyline(dm, isoValue=127)
    holes_vert_ids = mrmeshpy.HolesVertIds()
    contours = polyline2.contours2(holes_vert_ids)

    # compute the triangulation inside the contour
    mesh = mrmeshpy.triangulateContours(contours)
    mrmeshpy.subdivideMesh(mesh)
    return mesh

def save_tensor_image(tensor,path):
    grid  = make_grid(tensor, n_row=int(math.sqrt(tensor.shape[0])))
    save_image(grid, path)


if __name__ == "__main__":
    mask_path = "dataset/LeafData/Chinar/healthy/Chinar_healthy_0001_mask_aligned.JPG"
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mesh = mask_to_mesh(mask_path) 
    mrmeshpy.saveMesh(mesh, mrmeshpy.Path("leaf_2d.ply"))
    
    
    