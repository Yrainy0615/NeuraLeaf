import torch
import argparse
import yaml
import os
import sys
import torchvision.transforms as transforms
from pytorch3d import transforms as trans

# sys.path.append('scripts')
from scripts.data.dataset import BaseShapeDataset, DeformLeafDataset
from scripts.models.neuralbs import NBS
from scripts.models.decoder import  UDFNetwork, SWPredictor, TransPredictor
from scripts.utils.utils import latent_to_mask, save_tensor_image, deform_mesh, mask_to_mesh
from matplotlib import pyplot as plt
import numpy as np
import meshlib.mrmeshpy as mm
import meshlib.mrmeshnumpy as mn
from pytorch3d.loss import chamfer_distance, mesh_edge_loss,mesh_laplacian_smoothing
from scripts.data.mesh_process import MeshProcessor
from submodule.pixel2pixel.models import create_model
from submodule.pixel2pixel.options.test_options import TestOptions
from torchvision.transforms import Resize
from pytorch3d.io import save_obj, IO, load_obj, load_ply, load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures
import trimesh
import time
from scripts.utils.vis_utils import visualize_deform_part


class Generator:
    def __init__(self, cfg, device, shape_decoder,SWpredictor, Transpredictor, bone, shape_codes, deform_codes,shape_code_extra, model_texture,dataset,k):
        self.nbs_model = NBS(cfg['NBS'])
        self.device = device
        self.cfg = cfg['Trainer']
        self.k = k
        self.SWpredictor = SWpredictor
        self.Transpredictor = Transpredictor
        self.bone = bone
        self.shape_codes = shape_codes.to(self.device)
        self.shape_extra = shape_code_extra
        self.deform_codes = deform_codes.to(self.device)
        self.shape_decoder = shape_decoder
        self.dataset = dataset
        self.dataloader = dataset.get_loader(batch_size=1, shuffle=False)
        self.baseshape_dir = 'dataset/deformation_cvpr_new/base_space'
        self.baseshapes = [os.path.join(self.baseshape_dir, f) for f in os.listdir(self.baseshape_dir) if f.endswith('.ply')]
        self.model_texture = model_texture    
        self.mask_transform =  transforms.Compose([transforms.Normalize((0.5,), (0.5,),(0.5,)),
                                        transforms.Resize((256, 256)),])
        use_extra_shape = False
        if use_extra_shape:
            self.shape_codes = self.shape_extra


    @staticmethod
    def mask2mesh(masks):
        meshes = []
        for mask in masks:
            y_indices, x_indices = np.where(mask == 1)
            z_coords = np.zeros_like(x_indices)
            u_coords = x_indices / mask.shape[1]
            v_coords = (y_indices / mask.shape[0])
            verts = np.stack([u_coords, v_coords, z_coords], axis=-1)
            # random sample 15000 points
            idx = np.random.choice(verts.shape[0], 15000, replace=False)
            verts = verts[idx]
            pc =mn.pointCloudFromPoints(verts)
            
            pc.validPoints = mm.pointUniformSampling(pc, 1e-3)
            pc.invalidateCaches()
            mesh = mm.triangulatePointCloud(pc)
            verts = mn.getNumpyVerts(mesh)
            faces = mn.getNumpyFaces(mesh.topology)
            mesh_torch3d = Meshes(verts=[torch.tensor(verts).float()], faces=[torch.tensor(faces).long()])
            meshes.append(mesh_torch3d)
            
        return meshes

    def deform_interpolation(self, shape_idx,idx1, idx2,save_folder='results/deform/interpolation'):
        os.makedirs(save_folder, exist_ok=True)
        cond1 = self.deform_codes[idx1]
        cond2 = self.deform_codes[idx2]
        shape_cond = self.shape_codes[shape_idx]
        base_mesh = load_ply(self.baseshapes[shape_idx])
        base_points = base_mesh[0].to(self.device)
        base_faces = base_mesh[1].to(self.device)
        interp = torch.linspace(0, 1, 5).to(self.device)
        deform_cond = cond1 * (1-interp[:, None]) + cond2 * interp[:, None]
        base_points = base_points.unsqueeze(0).repeat(1, 1, 1)
        shape_cond = shape_cond.unsqueeze(0).repeat(1, 1)
        sw = self.SWpredictor(latent_code=shape_cond, centers=base_points)
        for i in range(5):
            rts_fw = self.Transpredictor(latent_code=deform_cond[i].unsqueeze(0), centers=self.bone)
            v0 = self.deform_lbs(base_points=base_points, sw=sw, rts_fw=rts_fw)
            deformed_mesh = Meshes(v0, base_faces.unsqueeze(0))
            save_path = os.path.join(save_folder, f'base_{shape_idx}_deform_{idx1}_{idx2}_{i}.obj')
            IO().save_mesh(deformed_mesh, save_path)
            
    def generate_deformation(self, shape_idx, base_shapes, deform_idx, save_folder='results/cvpr/generation',save_mesh=False):
        deform_codes = self.deform_codes[deform_idx] 
        shape_codes = self.shape_codes[shape_idx]
        meshes = []
        for i,base in enumerate(base_shapes):
            base_points = base.verts_packed().unsqueeze(0).to(self.device)
            base_face = base.faces_packed().unsqueeze(0).to(self.device)
            latent_deform = deform_codes[i]
            latent_shape = shape_codes[i]
            rts_fw = self.Transpredictor(latent_code=latent_deform.unsqueeze(0), centers=self.bone)
            sw = self.SWpredictor(latent_code=latent_shape.unsqueeze(0), centers=base_points)
            deform_points = self.deform_lbs(base_points,sw,rts_fw)
            current_mesh = Meshes(deform_points, base_face)
            # visual skinning weights
            visualize_deform_part(deform_points, base_face, self.bone, sw, save_path=os.path.join(save_folder, f'part_{shape_idx}_{deform_idx[i]}.obj'))
            if save_mesh:
                IO().save_mesh(current_mesh, os.path.join(save_folder,f'base_{shape_idx[i]}_deform_{deform_idx[i]}.obj'))
            meshes.append(current_mesh) 
        return meshes
    
    def shape_interpolation(self):
        base_list = ['leaf_11', 'leaf_5', 'leaf_6','leaf_19','leaf_23','leaf_42','maple4_d8','1-3','1-6','3-3',
                     '5-2','20','70', '6-3', '0006_0040_rgb', '0011_0019_rgb']
        deform_list = [139,135,134,132,131,127,118,116,113,2,3,10,20,21,25,32,33,39,51,56,61,64,200,212,207,198,179,10,35,23,45,68,89,67,56,32,12]
        # random two shape codes
        idx1, idx2 = np.random.randint(0,self.shape_codes.shape[0],2)
        interp = torch.linspace(0, 1, 5).to(self.device)
        deform_idx = 139
        deform_code = self.deform_codes[deform_idx]
        for i in range(5):
            shape_code = self.shape_codes[idx1] * (1-interp[i]) + self.shape_codes[idx2] * interp[i]
            base_mask = latent_to_mask(shape_code.unsqueeze(0), self.shape_decoder, k=self.k, size=256)
            base_mesh = mask_to_mesh(base_mask,shape_idx=[idx1,idx2],save_mesh=False)
            base_mesh = base_mesh.to(self.device)
            sw = self.SWpredictor(latent_code=shape_code.unsqueeze(0), centers=base_mesh.verts_packed().unsqueeze(0))
            rts_fw = self.Transpredictor(latent_code=deform_code.unsqueeze(0), centers=self.bone)
            base_points = base_mesh.verts_packed().unsqueeze(0) - base_mesh.verts_packed().unsqueeze(0).mean(1).unsqueeze(1)
            deformed_points = self.deform_lbs(base_points,sw,rts_fw)
            deformed_mesh = Meshes(deformed_points, base_mesh.faces_packed().unsqueeze(0))
            save_path = os.path.join('results/cvpr/rebuttal/', f'base_{idx1}_{idx2}_{deform_idx}_{i}.obj')
            IO().save_mesh(deformed_mesh, save_path)
            IO().save_mesh(base_mesh, os.path.join('results/cvpr/rebuttal/', f'base_{idx1}_{idx2}_{deform_idx}_{i}_base.obj'))
     
    @staticmethod        
    def uv_mapping(meshes:Meshes, texture_image:torch.tensor, save_mesh=True, return_mesh=False):
        texture_lsit = []
        for i in range(len(meshes)):
            mesh = meshes[i]
            texture_img = texture_image[i]
            verts= mesh.verts_packed()
            faces = mesh.faces_packed().unsqueeze(0)
            faces = faces.to(texture_image.device)
            uvs = torch.zeros_like(verts,device=texture_image.device)
            uvs[:,0] =  verts[:,0]
            uvs[:,1] = 1- verts[:,1]
            uvs = uvs.unsqueeze(0)
            texture = Textures(verts_uvs=uvs[:,:,:2], faces_uvs=faces, maps=texture_img.permute(1,2,0).unsqueeze(0))
            mesh.textures = texture
            mesh.verts_uvs = uvs[:,:,:2].squeeze()
            mesh.faces_uvs = faces.squeeze()
            mesh.textures_map = texture_img.permute(1,2,0)
            if save_mesh:
                save_obj(f'mesh_{i}.obj', mesh.verts_packed(), mesh.faces_packed(),verts_uvs=uvs[:,:,:2].squeeze(),faces_uvs=faces.squeeze(),texture_map = texture_img.permute(1,2,0))
        
    
    def deform_lbs(self, base_points, sw, rts_fw):
        """
        rts_fw:(b,K,7)
        sw: (b,N,K)
        bone: (B,K,3)
        
        """
        bone = self.bone.unsqueeze(0).repeat(base_points.shape[0],1,1).to(self.device )
        B, N, K = sw.shape
        B, K, _ = bone.shape
        v0 = base_points.view(-1,3)
        disp = rts_fw[:,:,:3]
        rot = rts_fw[:,:,3:]
        rot = trans.quaternion_to_matrix(rot.view(B*K, 4).contiguous()).view(B, K, 3, 3).contiguous()
        hd_disp = torch.repeat_interleave(disp, N, dim=0)
        hd_rot = torch.repeat_interleave(rot, N, dim=0)
        hd_pos = torch.repeat_interleave(bone, N, dim=0)
        per_hd_v = torch.einsum("abcd,abd->abc", hd_rot, (v0[:, None] - hd_pos)) + hd_pos + hd_disp  # (B*V, 40, 3)
        # per_hd_v = torch.einsum("abcd,abd->abc", hd_rot, (v0[:, None]))  + hd_disp  # (B*V, 40, 3)
        region_score = sw.view(-1, K)
        v = torch.sum(region_score[:, :, None] * per_hd_v, 1)  # (B*V, 3)
        return v.view(B, N, 3)

    def generate_baseshape(self, shape_idx, dataset,save_folder,k,mode="generation",save_image=False): 
        """
        shape latent -> 2D SDF -> binary mask 
        """
        if mode == "interpolation":
            idx1, idx2 = np.random.randint(0,dataset.__len__(),2)
            masks1, mask2 = dataset[idx1]['hint'], dataset[idx2]['hint']
            masks_gt = torch.stack([torch.tensor(masks1), torch.tensor(mask2)], dim=0)
            # masks_gt = masks_gt.permute(0, 3, 1, 2)
            # masks_gt = masks_gt[:,0,:,:].unsqueeze(1)
            # masks_gt = mask_transform(masks_gt)
            latent_x = latent[idx1]
            latent_y = latent[idx2]
            save_name = os.path.join(save_folder, f'mask_inter_{idx1}_{idx2}.png')
            save_name_gt = os.path.join(save_folder, f'mask_gt_{idx1}_{idx2}.png')
            # linear interpolation
            weights = torch.linspace(0, 1, 10).to(latent_x.device)
        
            latent = latent_x * (1-weights[:, None]) + latent_y * weights[:, None]
        else:
            latent = self.shape_codes[shape_idx]
        # batch split 
        max_batch = 16
        n = latent.shape[0]
        masks = []
        for i in range(0, n, max_batch):
            with torch.no_grad():
                masks_batch = latent_to_mask(latent[i:i+max_batch], decoder=self.shape_decoder, k=k,size=256)
                masks_batch = masks_batch.unsqueeze(1)
                # binary masks to 0 & 1
                masks_batch = torch.round(masks_batch)
                masks_batch = masks_batch.repeat(1, 3, 1, 1)
                masks_batch = self.mask_transform(masks_batch)
                masks.append(masks_batch)


        if save_image:
            masks = torch.vstack(masks)
            save_name = os.path.join(save_folder, f'masks.png')
            save_name_gt = os.path.join(save_folder, f'masks_gt.png')
            save_tensor_image(masks, save_name,normalize=True)
            # save_tensor_image(masks_gt,save_name_gt, normalize=True)
        masks = torch.vstack(masks)
        return masks
    
    def generate_texture(self, latent_tex:torch.tensor, masks:torch.tensor,save_folder:str,save_image=False):
        """
        generate texture from binary mask
        """
        # idx1, idx2 = idx
        # random generate 50 idx 
        tex_idx = np.random.randint(0,latent_tex.shape[0],1)
        # batch
        
        # idx= tex_idx[i]
        cond = latent_tex[tex_idx]
        
        # texture_pred  = self.model_texture.netG(masks,cond)
        texture_pred = self.model_texture.netG(masks,cond)
        texture_pred = (texture_pred+1)
        # texture_gt = model_texture.netG(masks_gt,cond)
        # texture_gt = (texture_gt+1)/2
        save_name_pred  = os.path.join(save_folder, f'texture_pred_tex.png')
        # save_name_gt = os.path.join(save_folder, f'texture_gt_{idx1}_{idx2}.png') 
        if save_image:
            save_tensor_image(texture_pred, save_name_pred)
            # save_tensor_image(texture_gt, save_name_gt)
        return texture_pred

    def fitting(self, target, save_folder='results/deform/fitting',epoch=2001):
        os.makedirs(save_folder, exist_ok=True)
        deform_target = torch.mean(self.deform_codes, dim=0)
        deform_target.requires_grad = True
        target_points = load_ply(target)[0].to(self.device)
        optimizer = torch.optim.Adam([deform_target], lr=1e-3)
        shape_cond = self.shape_codes[150]
        base_mesh = load_ply(self.baseshapes[150])
        base_points = base_mesh[0].to(self.device)
        base_faces = base_mesh[1].to(self.device)
        base_points = base_points.unsqueeze(0).repeat(1, 1, 1)
        shape_cond = shape_cond.unsqueeze(0).repeat(1, 1)
        sw = self.SWpredictor(latent_code=shape_cond, centers=base_points)
        for i in range(epoch):
            optimizer.zero_grad()
            rts_fw = self.Transpredictor(latent_code=deform_target.unsqueeze(0), centers=self.bone)
            v0 = self.deform_lbs(base_points=base_points, sw=sw, rts_fw=rts_fw)
            deform_mesh = Meshes(v0, base_faces.unsqueeze(0))   
            loss_chamfer = chamfer_distance(v0, target_points.unsqueeze(0))[0]
            loss_edge = mesh_edge_loss(deform_mesh)
            loss_laplacian = mesh_laplacian_smoothing(deform_mesh)
            loss_dict = {'shape': loss_chamfer, 'smooth': loss_laplacian, 'edge': loss_edge} # 'pme':loss_pme
            loss_total = 0
            for key in loss_dict.keys():
                loss_total += loss_dict[key] * self.cfg['lambdas'][key]
            loss_total.backward(retain_graph=True)
            optimizer.step()
            printstr = "Epoch: {} ".format(i)
            if i %500==0:
                optimizer.param_groups[0]['lr'] *= 0.5
            for k in loss_dict:
                printstr += "{}: {:.4f} ".format(k, loss_dict[k])
            print(printstr)
        IO().save_mesh(deform_mesh, os.path.join(save_folder, f'deform_optimized_42.obj'))
    
    def generate_deformation_old(self, rts_fw, skin_aux, bone, base_points, base_faces,mode='linear'):
        if mode =='linear':
            random_idx = np.random.randint(0,rts_fw.shape[0],2)
            rts_fw_1 = rts_fw[random_idx[0]]
            rts_fw_2 = rts_fw[random_idx[1]]
            interp = torch.linspace(0, 1, 10).to(rts_fw.device)
            for i in range(10):
                rts_fw = rts_fw_1 * (1-interp[i]) + rts_fw_2 * interp[i]
                skin_fw = self.nbs_model.skinning(bone,base_points.unsqueeze(0),None,skin_aux[0])
                deformed_pred, bones_dfm = self.nbs_model.lbs(bone, base_points.unsqueeze(0),rts_fw.to(self.device), skin_fw, backward=False)
                deformed_mesh = Meshes(deformed_pred, base_faces.unsqueeze(0))
                save_path = 'results/deform/generation'
                IO().save_mesh(deformed_mesh,os.path.join(save_path, f'deformed_mesh_{i}.obj'))
                # IO().save_mesh(base_mesh, os.path.join(save_path, f'base_mesh_{i}.obj') )
         
    def deform_mesh_displacement(self,mesh,deformer,lat_rep):
        
        points_neutral = torch.from_numpy(np.array(mesh.vertices)).float().unsqueeze(0).to(lat_rep.device)

        with torch.no_grad():
            grid_points_split = torch.split(points_neutral, 2000, dim=1)
            delta_list = []
            for split_id, points in enumerate(grid_points_split):
                glob_cond = glob_cond.repeat(1, points.shape[1], 1)
                d= deformer(points, glob_cond)
                delta_list.append(d.detach().clone())

                torch.cuda.empty_cache()
            delta = torch.cat(delta_list, dim=1)

        pred_posed = points_neutral[:, :, :3] + delta.squeeze()
        verts = pred_posed.detach().cpu().squeeze().numpy()
        mesh_deformed = trimesh.Trimesh(verts, mesh.faces, process=False)

        return mesh_deformed       
    
    def pose_transfer(self):
        base_list = ['leaf_11', 'leaf_5', 'leaf_6','leaf_19','leaf_23','leaf_42','maple4_d8','1-3','1-6','3-3',
                     '5-2','20','70', '6-3', '0006_0040_rgb', '0011_0019_rgb']
        # base_list = ['6-3','6-4','7-2']
        deform_list = [139,135,134,132,131,127,118,116,113,2,3,10,20,21,25,32,33,39,51,56,61,64,200,212,207,198,179]
        # for batch in self.dataloader:
        base_mesh_path = 'dataset/cvpr_final/base_shape_extra'
        base_mesh_list = [os.path.join(base_mesh_path, f) for f in os.listdir(base_mesh_path) if f.endswith('.obj')]
        for base_file in base_mesh_list:
            # base_file = batch['base_mesh']
            base_name = os.path.basename(base_file).split('.')[0]
            # deform_points = batch['deform_points'].to(self.device)
            # base_name = batch['base_name']

            if base_name in base_list:
                # base_file = batch['base_mesh']
                # idx = batch['idx'].to(self.device)
                # shape_idx = batch['shape_idx'].to(self.device) # start here
                # save_name = f'{base_name[0]}_deform_{int(idx)}.obj'
                # if not os.path.exists(os.path.join('results/cvpr/pose_transfer', save_name)):
                    # self.shape_codes = self.shape_codes.to(self.device)
                    # self.deform_codes = self.deform_codes.to(self.device)
                    # latent_shape = self.shape_codes[shape_idx]
                    # deform_code = self.deform_codes[idx]
                    # load base mesh   
                base_mesh = load_objs_as_meshes([base_file], device=self.device)
                    
                for batch in self.dataloader:
                    # base_name = batch['base_name']
                    # if 'leaf' in base_name[0]:  
                        # base_file  = batch['base_mesh']
                    shape_idx = batch['shape_idx'].to(self.device)
                    idx = batch['idx'].to(self.device)
                    save_name = f'{base_name}_deform_{int(idx)}.obj'

                    if idx in deform_list and not os.path.exists(os.path.join('results/cvpr/pose_transfer', save_name)):
                    
                        deform_code = self.deform_codes[idx]
                        # base_mesh = load_objs_as_meshes(base_file, device=self.device)

                        latent_shape = self.shape_codes[shape_idx]
                        # base_name = os.path.basename(base_meshfile).split('.')[0]
                        # base_mesh = load_objs_as_meshes([base_meshfile], device=self.device)
                        base_points = base_mesh.verts_packed().unsqueeze(0)
                        base_points = base_points - base_points.mean(1).unsqueeze(1)
                        # base_points = base_points - base_points.mean(1).unsqueeze(1)
                        base_faces = base_mesh.faces_packed().unsqueeze(0)
                        # deform
                        sw = self.SWpredictor(latent_code=latent_shape, centers=base_points)
                        trans = self.Transpredictor(latent_code=deform_code, centers=self.bone)
                        deformed_points = self.deform_lbs(base_points,sw,trans)
                        deformed_mesh = Meshes(deformed_points, base_faces, base_mesh.textures)
                        # save_name = f'{base_name[0]}_deform_{int(idx)}.obj'

                        IO().save_mesh(deformed_mesh, os.path.join('results/cvpr/pose_transfer', save_name))
                        print(f'save {save_name} success')
                        
        pass

    def pose_interpolation(self):
        base_list = ['leaf_11', 'leaf_5', 'leaf_6','leaf_19','leaf_23','leaf_42','maple4_d8','1-3','1-6','3-3',
                     '5-2','20','70', '6-3', '0006_0040_rgb', '0011_0019_rgb']
        deform_list = [139,135,134,132,131,127,118,116,113,2,3,10,20,21,25,32,33,39,51,56,61,64,200,212,207,198,179,10,35,23,45,68,89,67,56,32,12]
        for batch in self.dataloader:
            base_file = batch['base_mesh']
            base_name = batch['base_name']
            idx = batch['idx'].to(self.device)
            if int(idx) in deform_list:
                base_file = batch['base_mesh']
                shape_idx = batch['shape_idx'].to(self.device) # start here
                save_name = f'{base_name[0]}_deform_{int(idx)}.obj'
                self.shape_codes = self.shape_codes.to(self.device)
                self.deform_codes = self.deform_codes.to(self.device)
                latent_shape = self.shape_codes[shape_idx]
                # random interpolation of deform idx 10
                deform_code = self.deform_codes[idx]
                deform_code_neutral = self.deform_codes[15]
                interp = torch.linspace(0, 1, 10).to(self.device)
                for batch in self.dataloader:
                    base_name = batch['base_name']
                    if base_name[0] in base_list:  
                        base_file  = batch['base_mesh']
                        base_mesh = load_objs_as_meshes(base_file, device=self.device)
                        base_points = base_mesh.verts_packed().unsqueeze(0)
                        base_faces = base_mesh.faces_packed().unsqueeze(0)
                        for i in range(10):
                            save_name = f'{base_name[0]}_deform_{int(idx)}_{i}.obj'
                            code = deform_code_neutral * (1-interp[i]) + deform_code * interp[i]
                            sw = self.SWpredictor(latent_code=latent_shape, centers=base_points)
                            T = self.Transpredictor(latent_code=code, centers=self.bone)  
                            deform_points = self.deform_lbs(base_points,sw,T)   
                            deformed_mesh = Meshes(deform_points, base_faces, base_mesh.textures)
                            IO().save_mesh(deformed_mesh, os.path.join('results/cvpr/pose_interpolation', save_name))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RUN Leaf NPM')
    parser.add_argument('--gpu', type=int, default=5, help='gpu index')
    parser.add_argument('--save_folder', type=str, default='results', help='output directory')
    parser.add_argument('--config', type=str, default='scripts/configs/bashshape.yaml', help='config file')
    
    # setting
    args = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device(f'cuda:{str(args.gpu)}' if torch.cuda.is_available() else 'cpu')
    CFG = yaml.safe_load(open(args.config, 'r')) 

    # meshprocessor = MeshProcessor('dataset/deformation_cvpr_new')
    CFG_deform = yaml.safe_load(open('scripts/configs/deform.yaml', 'r'))

    # load dataset
    dataset = DeformLeafDataset()
    
    # load deformation model 
    num_bones = CFG_deform['NBS']['num_bones']
    swpredictor = SWPredictor(num_bones=num_bones).to(device)
    transpredictor = TransPredictor(num_bones=num_bones).to(device)
    checkpoint_deform = torch.load('checkpoints/cvpr/latest_neuraleaf_deform_small.pth', device)
    bone = checkpoint_deform['bone']
    bone = bone.to(device)
    deform_codes = checkpoint_deform['deform_codes']['weight']
    deform_codes = deform_codes.to(device)
    swpredictor.load_state_dict(checkpoint_deform['SWpredictor'])
    transpredictor.load_state_dict(checkpoint_deform['Transpredictor'])
    swpredictor.eval()
    transpredictor.eval()
    swpredictor.to(device)
    transpredictor.to(device)

    # load shape decoder
    decoder_base = UDFNetwork(d_in=CFG['Base']['Z_DIM'],
                         d_hidden=CFG['Base']['decoder_hidden_dim'],
                         d_out=CFG['Base']['decoder_out_dim'],
                         n_layers=CFG['Base']['decoder_nlayers'],
                         udf_type='sdf',
                         geometric_init=False) 
    decoder_base.eval()
    decoder_base.to(device)
    checkpoint_base = torch.load('checkpoints/cvpr/epoch_1000_base_deformleaf.pth.pth',map_location=device)
    decoder_base.load_state_dict(checkpoint_base['decoder'])
    if 'k' in checkpoint_base:
        k = checkpoint_base['k']
    else:
        k=1
    latent_shape =checkpoint_base['latent_shape']['weight']
    latent_shape = latent_shape.to(device)
    
    # load extra shape code
    checkpoint_extra = torch.load('checkpoints/cvpr/extra_shape.pth.pth')
    shape_code_extra = checkpoint_extra['latent_shape']['weight']
    # load texture model
    opt_texture = TestOptions().parse()
    opt_texture.gpu_ids = [args.gpu]
    model_texture = create_model(opt_texture)
    latent_tex = model_texture.setup(opt_texture)
    model_texture.eval()
    
    
    """
    functions of generator:
    1. deform_interpolation: linear interpolation between two deformation codes
    2. fitting: optimize the deformation code to fit the target point cloud
    3. generate_baseshape: generate binary mask from shape latent
    4. generate_texture: generate texture from binary mask
    5. mesh_to_mask: convert binary mask to pytorch3d mesh
    6. shape_interpolation: linear interpolation between two shape codes with fixed deformation code
    """
    generator = Generator(CFG_deform, device, decoder_base,swpredictor, transpredictor, bone, latent_shape, deform_codes,shape_code_extra,model_texture,dataset,k)
    # generator.shape_interpolation()
    # generator.deform_interpolation(shape_idx=300, idx1=0, idx2=31)
    # generator.fitting('dataset/deformation_cvpr_new/deform_train/leaf_42_deform.ply')
    start_time = time.time()
    shape_idx = np.random.randint(0,shape_code_extra.shape[0],1) # random shape idx
    deform_idx = np.random.randint(0,deform_codes.shape[0],1) # random deform idx
    # get a list of 0 to shape_code_extra.shape[0]
    # shape_idx = np.arange(shape_code_extra.shape[0])
    # deform_idx = np.arange(deform_codes.shape[0])
    # generator.pose_interpolation()
    masks = generator.generate_baseshape(shape_idx, dataset,CFG['Training']['test_result'], k, save_image=False)
    # texture generation  

    texture = generator.generate_texture(latent_tex,masks,CFG['Training']['test_result'])

    # base mesh with texture
    base_mesh = mask_to_mesh(masks,shape_idx,save_mesh=True) # return pytorch3d mesh
    time_base = time.time()
    print(f'Execution base time: {time_base-start_time}')
    # deformed_mesh =generator.generate_deformation(shape_idx,base_mesh,deform_idx,save_mesh=True)
    end_time = time.time()
    print(f'Execution full time: {end_time-start_time}')
    # uv mapping 
    generator.uv_mapping(base_mesh, texture)
    


        
        

        
    
    


    
    
    
    
    
    