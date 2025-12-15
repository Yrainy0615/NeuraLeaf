import torch
import torch.nn as nn
import numpy as np
from pytorch3d.loss import chamfer_distance
from torch.autograd import grad


def deformation_mapping_loss(d_cham, latent_code, phi=1.0):
    """
    Deformation mapping loss: encourages consistent mapping between 
    deformation magnitude (chamfer distance) and latent code norm.

    """
    # Compute L2 norm for each sample in the batch
    if latent_code.dim() > 1:
        # [batch_size, latent_dim] -> [batch_size]
        latent_norm = torch.norm(latent_code, p=2, dim=1)
    else:
        # [latent_dim] -> scalar
        latent_norm = torch.norm(latent_code, p=2)
    
    # If batch, use mean of latent norms; otherwise use scalar
    if latent_norm.dim() > 0:
        latent_norm = latent_norm.mean()
    
    # Compute mapping loss: (d_cham / latent_norm - phi)^2
    # This encourages: d_cham / latent_norm ≈ phi
    loss_map = ((d_cham / (latent_norm + 1e-8)) - phi) ** 2
    
    return loss_map

def gradient_penalty(critic, real, fake, alpha, train_step, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, alpha, train_step)
 
    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

class ARAPLoss(nn.Module):
    def __init__(self, vertex, faces, average=False):
        super(ARAPLoss, self).__init__()
        self.nv = vertex.size(1)
        self.nf = faces.size(1)
        self.average = average
        laplacian = np.zeros([self.nv, self.nv]).astype(np.float32)
        faces = faces.squeeze().cpu().numpy()
        laplacian[faces[:, 0], faces[:, 1]] = 1
        laplacian[faces[:, 1], faces[:, 0]] = 1
        laplacian[faces[:, 1], faces[:, 2]] = 1
        laplacian[faces[:, 2], faces[:, 1]] = 1
        laplacian[faces[:, 2], faces[:, 0]] = 1
        laplacian[faces[:, 0], faces[:, 2]] = 1

        self.register_buffer('laplacian', torch.from_numpy(laplacian))

    def forward(self, dx, x):
        # lap: Nv Nv
        # dx: N, Nv, 3
        diffx = torch.zeros(x.shape[0], x.shape[1], x.shape[1]).to(dx.device)
        diffdx = torch.zeros(x.shape[0], x.shape[1], x.shape[1]).to(dx.device)
        for i in range(3):
            self.laplacian = self.laplacian.to(dx.device)
            dx_sub = self.laplacian.matmul(torch.diag_embed(dx[:,:,i])) # N, Nv, Nv)
            dx_diff = (dx_sub - dx[:,:,i:i+1])
            
            x_sub = self.laplacian.matmul(torch.diag_embed(x[:,:,i])) # N, Nv, Nv)
            x_diff = (x_sub - x[:,:,i:i+1])
            
            diffdx += (dx_diff).pow(2)
            diffx +=   (x_diff).pow(2)

        diff = (diffx-diffdx).abs()
        diff = torch.stack([diff[i][self.laplacian.bool()].mean() for i in range(x.shape[0])])
        #diff = diff[self.laplacian[None].repeat(x.shape[0],1,1).bool()]
        return diff


def compute_loss_corresp_forward(batch, decoder, latent_deform, device,cfg):
    batch_cuda = {k: v.to(device).float() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    deform_code = latent_deform(batch['deform_idx'].to(device))
    points_neutral = batch_cuda['points_neutral'].clone().detach().requires_grad_()
    deform_code = deform_code.unsqueeze(1)
    cond = deform_code.repeat(1, points_neutral.shape[1], 1)
    delta= decoder(points_neutral, cond)
    pred_posed = points_neutral + delta.squeeze()
    # mse loss
    points_posed = batch_cuda['points_posed']
    loss_corresp = (pred_posed - points_posed[:, :, :3])**2#.abs()
    
    # loss_distance = torch.tensor(0)
    # distance regularizer
    # if cfg['use_distance']:
    #     distance = torch.norm(deform_code,p=2,dim=-1)
    #     delta_gt = points_posed - points_neutral
    #     delta_norm = torch.norm(delta_gt,p=2,dim=(1,2)) 
    #     loss_distance = ((distance.squeeze()/delta_norm) - phi)**2
    

    
    # latent code regularization
    lat_mag = torch.norm(deform_code, dim=-1)**2
    # samps = (torch.rand(cond.shape[0], 100, 3, device=cond.device, dtype=cond.dtype) -0.5)*2.5
    # delta = decoder(samps, cond[:, :100, :])
    # loss_reg_zero = (delta**2).mean()
    return {'corresp': loss_corresp.mean(),
            'lat_reg': lat_mag.mean(),
    }

def surface_area_loss(verts, faces, verts_deformed):
    v0 = verts[:, faces[:, 0], :]  # (b, F, 3)
    v1 = verts[:, faces[:, 1], :]  # (b, F, 3)
    v2 = verts[:, faces[:, 2], :]  # (b, F, 3)

    v0_def = verts_deformed[:, faces[:, 0], :]  # (b, F, 3)
    v1_def = verts_deformed[:, faces[:, 1], :]  # (b, F, 3)
    v2_def = verts_deformed[:, faces[:, 2], :]  # (b, F, 3)

    # 计算原始面的面积
    e1 = v1 - v0  # (b, F, 3)
    e2 = v2 - v0  # (b, F, 3)
    cross_product = torch.cross(e1, e2, dim=2)  # (b, F, 3)
    area_original = 0.5 * torch.norm(cross_product, dim=2)  # (b, F)

    # 计算变形后面的面积
    e1_def = v1_def - v0_def  # (b, F, 3)
    e2_def = v2_def - v0_def  # (b, F, 3)
    cross_product_def = torch.cross(e1_def, e2_def, dim=2)  # (b, F, 3)
    area_deformed = 0.5 * torch.norm(cross_product_def, dim=2)  # (b, F)

    # 计算面积差异的平方
    loss = torch.mean((area_deformed - area_original) ** 2)

    return loss

def gradient(outputs, inputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True)[0][:, :, -3:]
    return points_grad