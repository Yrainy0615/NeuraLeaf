import torch
import argparse
import yaml
import os
import sys
import torchvision.transforms as transforms
# sys.path.append('scripts')
from scripts.data.dataset import BaseShapeDataset
from scripts.models.decoder import SDFDecoder, UDFNetwork
from scripts.utils.utils import latent_to_mask, save_tensor_image, deform_mesh, mask_to_mesh
from matplotlib import pyplot as plt
import numpy as np
from scripts.data.mesh_process import MeshProcessor
from submodule.pixel2pixel.models import create_model
from submodule.pixel2pixel.options.test_options import TestOptions
from torchvision.transforms import Resize
from pytorch3d.io import save_obj

def generate_baseshape(decoder, latent, dataset,save_folder,idx,mask_transform,k,save_image=False): 
    """
    shape latent -> 2D SDF -> binary mask 
    """
    idx1, idx2 = idx
    masks1, mask2 = dataset[idx1]['hint'], dataset[idx2]['hint']
    masks_gt = torch.stack([torch.tensor(masks1), torch.tensor(mask2)], dim=0)
    masks_gt = masks_gt.permute(0, 3, 1, 2)
    # masks_gt = masks_gt[:,0,:,:].unsqueeze(1)
    masks_gt = mask_transform(masks_gt)
    latent_x = latent[idx1]
    latent_y = latent[idx2]
    save_name = os.path.join(save_folder, f'mask_inter_{idx1}_{idx2}.png')
    save_name_gt = os.path.join(save_folder, f'mask_gt_{idx1}_{idx2}.png')
    # linear interpolation
    weights = torch.linspace(0, 1, 10).to(latent_x.device)
   
    latent = latent_x * (1-weights[:, None]) + latent_y * weights[:, None]
    with torch.no_grad():
        masks = latent_to_mask(latent.to('cuda'), decoder=decoder, k=k,size=256)
        masks = masks.unsqueeze(1)
        # binary masks to 0 & 1
        masks = torch.round(masks)
        masks = masks.repeat(1, 3, 1, 1)
        masks = mask_transform(masks)

        if save_image:
            save_tensor_image(masks, save_name)
            save_tensor_image(masks_gt,save_name_gt)

    return masks, masks_gt

def generate_texture(model_texture,masks:torch.tensor, masks_gt,save_folder:str,idx,save_image=False):
    """
    generate texture from binary mask
    """
    idx1, idx2 = idx
    texture_pred  = model_texture.netG(masks)
    texture_pred = (texture_pred+1)/2
    texture_gt = model_texture.netG(masks_gt)
    texture_gt = (texture_gt+1)/2
    save_name_pred  = os.path.join(save_folder, f'texture_pred_{idx1}_{idx2}.png')
    save_name_gt = os.path.join(save_folder, f'texture_gt_{idx1}_{idx2}.png') 
    if save_image:
        save_tensor_image(texture_pred, save_name_pred)
        save_tensor_image(texture_gt, save_name_gt)
    return texture_pred, texture_gt



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RUN Leaf NPM')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    parser.add_argument('--save_folder', type=str, default='results', help='output directory')
    parser.add_argument('--config', type=str, default='scripts/configs/bashshape.yaml', help='config file')
    
    # setting
    args = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CFG = yaml.safe_load(open(args.config, 'r')) 
    mask_transform =  transforms.Compose([transforms.Normalize((0.5,), (0.5,),(0.5,)),
                                        transforms.Resize((256, 256)),])
    meshprocessor = MeshProcessor('dataset')
    
    # load dataset
    dataset = BaseShapeDataset(CFG['Training']['data_dir'], CFG['Training']['n_sample'])
    
    # load mask decoder
    decoder_base = UDFNetwork(d_in=CFG['Base']['Z_DIM'],
                         d_hidden=CFG['Base']['decoder_hidden_dim'],
                         d_out=CFG['Base']['decoder_out_dim'],
                         n_layers=CFG['Base']['decoder_nlayers'],
                         udf_type='sdf',
                         geometric_init=False) 
    decoder_base.eval()
    decoder_base.to(device)
    checkpoint_base = torch.load('checkpoints/baseshape/latest_sigmoid.pth',map_location='cpu')
    decoder_base.load_state_dict(checkpoint_base['decoder'])
    if 'k' in checkpoint_base:
        k = checkpoint_base['k']
    else:
        k=1
    latent_shape =checkpoint_base['latent_shape']['weight']
    
    # baseshape generation 
    idx = [200, 600]

    masks, masks_gt = generate_baseshape(decoder_base, latent_shape, dataset,CFG['Training']['test_result'], idx,mask_transform, k, save_image=True)
    # texture generation  
    opt_texture = TestOptions().parse()
    opt_texture.gpu_ids = [args.gpu]
    model_texture = create_model(opt_texture)
    model_texture.setup(opt_texture)
    model_texture.eval()
    texture, texture_gt = generate_texture(model_texture, masks, masks_gt,CFG['Training']['test_result'], idx, save_image=True)


    
    # base mesh with texture
    base_mesh = mask_to_mesh(masks) # return pytorch3d mesh
    
    # uv mapping 
    textured_base_mesh = meshprocessor.uv_mapping(base_mesh, texture)

    # deformation generation
    checkpoint_deform = torch.load('checkpoints/deformation/eccv/deform.tar')
    decoder_def = UDFNetwork(d_in=CFG['Deform']['decoder_lat_dim'],
                         d_hidden=CFG['Deform']['decoder_hidden_dim'],
                         d_out=CFG['Deform']['decoder_out_dim'],
                         n_layers=CFG['Deform']['decoder_nlayers'],
                         udf_type='sdf',
                         geometric_init=False)
    decoder_def.load_state_dict(checkpoint_deform['decoder_state_dict'])
    lat_def_all = checkpoint_deform['latent_deform_state_dict']['weight']
    
    
    
    
    
    
    
    