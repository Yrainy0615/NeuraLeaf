import torch
import os
from scripts.data.dataset import DeformLeafDataset, DeformLeafDatasetNew
from scripts.utils.train_utils import BaseTrainer
from scripts.models.decoder import SWPredictor, TransPredictor, UDFNetwork, BoneDecoder
from scripts.models.neuralbs import NBS
from pytorch3d.io import load_ply, save_obj, IO, load_objs_as_meshes
from pytorch3d.loss import chamfer_distance, mesh_edge_loss, mesh_laplacian_smoothing,mesh_normal_consistency, point_mesh_edge_distance
from pytorch3d.structures import Meshes, pointclouds
import argparse 
import yaml
import math
import numpy as np
from pytorch3d import transforms
from scripts.utils.loss_utils import ARAPLoss, compute_loss_corresp_forward, surface_area_loss, deformation_mapping_loss
from scripts.utils.utils import mask_to_mesh, latent_to_mask
import json
# import umap
from matplotlib import pyplot as plt

torch.autograd.set_detect_anomaly(True)


class DeformTrainer():
    def __init__(self, shape_decoder,dataset,cfg, device, args):
        self.cfg = cfg['Trainer']
        self.shape_decoder = shape_decoder
        self.device = device
        self.num_bones = cfg['NBS']['num_bones']
        self.bone = None
        self.init_bones(mode='random')
        self.dataset = dataset
        self.dataloader = self.dataset.get_loader(batch_size=self.cfg['batch_size'], shuffle=False)
        self.args = args
        checkpoint_base = torch.load('checkpoints/cvpr/epoch_1000_base_deformleaf.pth.pth',map_location=device)
        self.shape_codes = checkpoint_base['latent_shape']['weight'].to(self.device)
        # self.shape_codes = torch.nn.Embedding(len(self.dataset), cfg['Base']['Z_DIM'], max_norm=1, device=device).requires_grad_(True)
        # torch.nn.init.normal_(self.shape_codes.weight.data, 0.0, 0.1/math.sqrt(cfg['NBS']['lat_dim']))
        self.shape_decoder.load_state_dict(checkpoint_base['decoder'])
        self.shape_decoder.eval()
        self.deform_codes = torch.nn.Embedding(len(self.dataset), cfg['NBS']['lat_dim'], max_norm=1, device=device).requires_grad_(True)
        # assert self.shape_codes.shape[0] == self.deform_codes.weight.size(0)
        # models
        self.shape_decoder.to(device)
        self.lbs_model = NBS(cfg['NBS'])
        self.SWpredictor = SWPredictor(num_bones=self.num_bones).to(device)
        self.Transpredictor = TransPredictor(num_bones=self.num_bones).to(device)
        deform_checkpoints = torch.load('checkpoints/cvpr/latest_deform_map.pth',map_location=device)
        # deform_codes_prior = deform_checkpoints['deform_codes']['weight']
        # with torch.no_grad():
        #     self.deform_codes.weight[:deform_codes_prior.size(0)] = deform_codes_prior        
        # self.bone = deform_checkpoints['bone'].to(device)
        self.bone_decoder = BoneDecoder(num_bones=self.num_bones).to(device)
        # with open ('chamfer_distances.json','r') as f:
        #     self.chamfer_dict = json.load(f)
        self.SWpredictor.load_state_dict(deform_checkpoints['SWpredictor'])
        self.Transpredictor.load_state_dict(deform_checkpoints['Transpredictor'])
        # optimizers
        self.optimizer_decoder = torch.optim.Adam(self.shape_decoder.parameters(), lr=self.cfg['LR_D'])
        self.optimizer_latent_deform = torch.optim.Adam(self.deform_codes.parameters(), lr=self.cfg['LR_LAT'])
        self.k = torch.nn.Parameter(torch.tensor(1.0))
        self.optimizer_k  = torch.optim.Adam([self.k], lr=0.01)
        self.optimizer_bone = torch.optim.Adam([self.bone], lr=0.001)
        # self.optimizer_bone = torch.optim.Adam(self.bone_decoder.parameters(), lr=self.cfg['LR_B'])
        self.optimizer_sw = torch.optim.Adam(self.SWpredictor.parameters(), lr=self.cfg['LR_SW'])
        self.optimizer_trans = torch.optim.Adam(self.Transpredictor.parameters(), lr=self.cfg['LR_D'])
        self.phi = torch.nn.Parameter(torch.tensor(1.0))
        self.optimizer_phi = torch.optim.Adam([self.phi], lr=0.01)

   
    def init_bones(self,mode='random'):
        if mode == 'random':
            xy = np.random.uniform(-1,1,(self.num_bones,2))
            z = np.zeros((self.num_bones,1))
            bone = np.concatenate([xy,z], axis=1)
            bone_tensor = torch.Tensor(bone)
            self.bone = torch.nn.Parameter(bone_tensor)
        elif mode=='prior':
            for batch in self.dataloader:
                base_points = batch['base_points'].to(self.device)
                bone_tensor = self.lbs_model.generate_bone()
                self.bone = torch.nn.Parameter(bone_tensor[:,:3])
                break

    def create_similarity_dictionary(self, save_file):
        similarity_dictionary = []
        for i , basefile in enumerate(self.baseshapes):
            basemesh = load_ply(basefile)
            base_points = basemesh[0].to(self.device)
            # base_points = base_points - base_points.mean(0)
            chamfer_dicts = []
            for j,extra_mesh in enumerate(self.basemesh_extra):
                extra_points = extra_mesh.verts_packed().to(self.device)
                extra_points = extra_points - extra_points.mean(0)
                chamfer_dist = chamfer_distance(base_points.unsqueeze(0), extra_points.unsqueeze(0))[0]
                chamfer_dicts.append((j, chamfer_dist))
            chamfer_dicts = sorted(chamfer_dicts, key=lambda x: x[1])
            # Get top 5 indices of the most similar shapes
            top_index = [entry[0] for entry in chamfer_dicts[-5:]]

            similarity_dictionary.append(chamfer_dicts[:5])
            print(f"Base shape {i} is most similar to shapes {top_index}")
        # save dict file
        with open(save_file, 'w') as f:
            json.dump(similarity_dictionary, f)   

    def finetune_sw(self, epoch,batch, checkpoint):
        deform_points = batch['deform_points'].to(self.device)
        idx = batch['idx'].to(self.device)
        deform_checkpoint = torch.load(checkpoint)
        self.bone = deform_checkpoint['bone'].to(self.device)
        self.SWpredictor.load_state_dict(deform_checkpoint['SWpredictor'])
        self.Transpredictor.load_state_dict(deform_checkpoint['Transpredictor'])
        self.Transpredictor.train()
        self.deform_codes = deform_checkpoint['deform_codes']['weight'].to(self.device)
        self.deform_codes.requires_grad_(True)   
        self.SWpredictor.train()
        for i,basefile in enumerate(self.baseshapes):
            self.optimizer_latent.zero_grad()   
            self.optimizer_sw.zero_grad()
            self.optimizer_trans.zero_grad()
            base_mesh = load_ply(basefile)
            # base_mesh = self.basemesh_extra[0]
            base_points = base_mesh[0].to(self.device)
            base_faces = base_mesh[1].unsqueeze(0).to(self.device)
            # base_points = base_mesh.verts_packed().to(self.device)
            # base_faces = base_mesh.faces_packed().unsqueeze(0).to(self.device)
            base_points = base_points.unsqueeze(0).repeat(1, 1, 1)
            shape_code = self.shape_codes[i]
            sw_base = self.SWpredictor(latent_code=shape_code.unsqueeze(0), centers=base_points)
            deform_code = self.deform_codes[int(idx)]
            rts_fw = self.Transpredictor(latent_code=deform_code.unsqueeze(0), centers=self.bone)
            v0 = self.deform_lbs(base_points,sw_base,rts_fw)
            new_mesh = Meshes(verts=v0, faces=base_faces)
            loss_chamfer = chamfer_distance(deform_points, v0)[0]
            loss_edge = mesh_edge_loss(new_mesh)
            loss_smooth = mesh_laplacian_smoothing(new_mesh)
            loss_dict = {'shape': loss_chamfer, 'smooth': loss_smooth, 'edge': loss_edge}
            loss_total = 0
            for key in loss_dict.keys():
                loss_total += loss_dict[key] * self.cfg['lambdas'][key]
            loss_total.backward()
            self.optimizer_sw.step()
            self.optimizer_latent.step()
            self.optimizer_trans.step()
            if i %100 == 0:
                save_name = f'base_{i}_deform_{int(idx)}.obj'
                save_dir = os.path.join('results/cvpr/finetune', save_name)
                IO().save_mesh(new_mesh,save_dir)
            printstr = "Epoch: {}, base:{},deform:{} ".format(epoch,i,int(idx))
            for k in loss_dict:
                printstr += "{}: {:.4f} ".format(k, loss_dict[k])
            print(printstr)
        return loss_dict
    
    def reduce_lr(self, epoch):
        if   self.cfg['lr_decay_interval_lat'] is not None and epoch % self.cfg['lr_decay_interval_lat'] == 0 and epoch > 0:
            # for param_group in self.optimizer_latent_shape.param_groups:
            #     param_group["lr"] = param_group["lr"] * self.cfg['lr_decay_factor_lat']
            for param_group in self.optimizer_latent_deform.param_groups:
                param_group["lr"] = param_group["lr"] * self.cfg['lr_decay_factor_lat']
            for param_group in self.optimizer_sw.param_groups:
                param_group["lr"] = param_group["lr"] * self.cfg['lr_decay_factor_lat']
            for param_group in self.optimizer_trans.param_groups:
                param_group["lr"] = param_group["lr"] * self.cfg['lr_decay_factor_lat']
            for param_group in self.optimizer_bone.param_groups:
                param_group["lr"] = param_group["lr"] * self.cfg['lr_decay_factor_lat']

            print(f"Reducing learning rate to {param_group['lr']}")

    def reinit_lr(self):
        for param_group in self.optimizer_latent_shape.param_groups:
            param_group["lr"] = self.cfg['LR_LAT']
        for param_group in self.optimizer_latent_deform.param_groups:
            param_group["lr"] = self.cfg['LR_LAT']
        for param_group in self.optimizer_sw.param_groups:
            param_group["lr"] = self.cfg['LR_LAT']
        for param_group in self.optimizer_decoder.param_groups:
            param_group["lr"] = self.cfg['LR_D']
        for param_group in self.optimizer_bone.param_groups:
            param_group["lr"] = self.cfg['LR_B']
        for param_group in self.optimizer_k.param_groups:
            param_group["lr"] = 0.01
        for param_group in self.optimizer_trans.param_groups:
            param_group["lr"] = self.cfg['LR_D']
        
    def save_checkpoint(self, checkpoint_path, save_name):
        checkpoint = {
            'deform_codes': self.deform_codes.state_dict(),
            # 'shape_codes': self.shape_codes.state_dict(),
            'bone': self.bone,
            # 'bone':self.bone_decoder.state_dict(),
            'SWpredictor': self.SWpredictor.state_dict(),
            'Transpredictor': self.Transpredictor.state_dict(),
            # 'phi': self.phi,
            }
        torch.save(checkpoint, os.path.join(checkpoint_path, save_name))

    def generate_base_mesh(self):
        for i in range(len(self.shape_codes_extra)):
            cond = self.shape_codes_extra[i]
            mask = latent_to_mask(cond.unsqueeze(0), self.shape_decoder, 128, k=100)
            base_mesh = mask_to_mesh(mask.unsqueeze(0), i,save_mesh=True)
            self.basemesh_extra.append(base_mesh)

    def train_shape_deform(self,epoch,batch):
        # load data
        batch_cuda = {k: v.to(self.device) for k, v in batch.items() if type(v) == torch.Tensor}
        idx = batch_cuda['idx']
        shape_code = self.shape_codes(idx)
        deform_code = self.deform_codes(idx)
        sdf_gt = batch_cuda['sdf']
        mask_gt = batch_cuda['mask']
        points = batch_cuda['points']
        deform_points = batch_cuda['deform_points']
        save_name = batch['save_name']
    
        if epoch < self.cfg['epoch_only_shape']:
            # optimizer zero grad
            self.optimizer_decoder.zero_grad()
            self.optimizer_latent_shape.zero_grad()
            self.optimizer_k.zero_grad()           
            
            # train base shape space
            glob_cond = torch.cat([shape_code.unsqueeze(1).expand(-1, points.shape[1], -1), points], dim=2)
            sdf_pred = self.shape_decoder(glob_cond)
            loss_sdf = torch.abs(sdf_pred.squeeze(-1) - sdf_gt).mean()
            mask_pred = latent_to_mask(shape_code, self.shape_decoder, 128, k=self.k)
            loss_sil = torch.abs(mask_pred - mask_gt).mean()
            lat_reg_shape = shape_code.norm(2, dim=1).mean()   
            loss_dict = {'sdf': loss_sdf, 'sil': loss_sil, 'reg_shape': lat_reg_shape}
            loss_shape = 0
            for key in loss_dict.keys():
                loss_shape += loss_dict[key] * self.cfg['lambdas'][key]
            loss_shape.backward()
            self.optimizer_decoder.step()
            self.optimizer_latent_shape.step()
            self.optimizer_k.step()
        
        # train deform space
        if epoch >= self.cfg['epoch_only_shape']:
            self.optimizer_latent_deform.lr = self.cfg['LR_LAT']
            self.optimizer_latent_deform.zero_grad()
            self.optimizer_sw.zero_grad()
            self.optimizer_bone.zero_grad()
            self.optimizer_trans.zero_grad()
            self.optimizer_latent_shape.zero_grad()
            self.optimizer_decoder.zero_grad()
            mask_pred = latent_to_mask(shape_code, self.shape_decoder, 128, k=self.k)
            base_mesh = mask_to_mesh(mask_pred, shape_code,save_mesh=False) # pytorch3d meshes
            base_mesh = base_mesh.to(device)
            base_points = base_mesh.verts_padded()
            base_faces = base_mesh.faces_padded()

            # forward
            sw = self.SWpredictor(latent_code=shape_code, centers=base_points) # [b, N,B]
            rts_fw = self.Transpredictor(latent_code=deform_code, centers=self.bone.to(self.device)) # [b,B,12] or [b,B,7]
            v0 = self.deform_lbs(base_points,sw,rts_fw)
            new_mesh = Meshes(verts=v0, faces=base_faces)
            # loss 
            loss_chamfer = chamfer_distance(deform_points, v0)[0]
            loss_edge = mesh_edge_loss(new_mesh)
            loss_smooth = mesh_laplacian_smoothing(new_mesh)
            lat_reg_deform = deform_code.norm(2, dim=1).mean()
            loss_dict= {'shape': loss_chamfer, 'smooth': loss_smooth, 'edge': loss_edge, 'reg_deform': lat_reg_deform}
            loss_deform = 0
            for key in loss_dict.keys():
                loss_deform += loss_dict[key] * self.cfg['lambdas'][key]
            loss_deform.backward()
            self.optimizer_sw.step()
            self.optimizer_latent_deform.step()
            self.optimizer_decoder.step()
            self.optimizer_k.step()
            self.optimizer_latent_shape.step()
            self.optimizer_bone.step()
            if epoch % 10 == 0:
                for i,mesh in enumerate(new_mesh):
                    save_file = os.path.join(self.cfg['save_result'], f'{save_name[i]}.obj')
                    IO().save_mesh(mesh,save_file)
        return loss_dict
                                        
    def train_step(self, epoch, batch):
        self.optimizer_bone.zero_grad()
        self.optimizer_latent_deform.zero_grad()
        self.optimizer_sw.zero_grad()
        self.optimizer_trans.zero_grad()
        self.optimizer_phi.zero_grad()
        deform_points = batch['deform_points'].to(self.device)
        base_name = batch['base_name']
        base_file = batch['base_mesh']
        base_mesh = load_objs_as_meshes(base_file, device=self.device)
        idx = batch['idx'].to(self.device)
        shape_idx = batch['shape_idx'].to(self.device) # start here
        
        # load base mesh

        base_points = base_mesh.verts_packed()
        base_faces = base_mesh.faces_packed()

        # latent codes
        shape_code = self.shape_codes[shape_idx]
        deform_code = self.deform_codes(idx)
        
        # forward 
        base_points = base_points.unsqueeze(0).repeat(deform_points.shape[0], 1, 1)
        shape_code = shape_code.repeat(deform_points.shape[0], 1)
        sw = self.SWpredictor(latent_code=shape_code, centers=base_points) # [b, N,B]
        rts_fw = self.Transpredictor(latent_code=deform_code, centers=self.bone.to(self.device)) # [b,B,12] or [b,B,7]
        v0 = self.deform_lbs(base_points,sw,rts_fw)

        new_mesh = Meshes(verts=v0, faces=base_faces.unsqueeze(0), textures=base_mesh.textures)

        loss_chamfer = chamfer_distance(deform_points, v0)[0]
        # smooth loss 
        # chamfer_map = self.chamfer_dict[base_name[0]]
        # loss_map = deformation_mapping_loss(chamfer_map, deform_code, phi=self.phi)
        loss_edge = mesh_edge_loss(new_mesh)
        loss_smooth = mesh_laplacian_smoothing(new_mesh)
        # surface area loss
        loss_dict = {'shape': loss_chamfer, 'smooth': loss_smooth, 'edge': loss_edge} # 'pme':loss_pme
        loss_total = 0
        for key in loss_dict.keys():
            loss_total += loss_dict[key] * self.cfg['lambdas'][key]
        loss_total.backward()
        self.optimizer_bone.step()
        self.optimizer_latent_deform.step()
        self.optimizer_sw.step()
        self.optimizer_trans.step()
        # self.optimizer_phi.step()
        if epoch % 10 == 0:
            save_folder = self.cfg['save_result']
            os.makedirs(save_folder,exist_ok=True)
            save_file = os.path.join(save_folder, f'{base_name[0]}.obj')
            IO().save_mesh(new_mesh,save_file)
        printstr = "Epoch: {}, shape:{} ".format(epoch,base_name)
        for k in loss_dict:
            printstr += "{}: {:.4f} ".format(k, loss_dict[k])
        print(printstr)
        return loss_dict

    def train_step_bone(self, epoch,batch):
        self.optimizer_bone.zero_grad()
        self.optimizer_latent_deform.zero_grad()
        self.optimizer_sw.zero_grad()
        self.optimizer_trans.zero_grad()
        deform_points = batch['deform_points'].to(self.device)
        base_name = batch['base_name']
        base_file = batch['base_mesh']
        base_mesh = load_objs_as_meshes(base_file, device=self.device)
        idx = batch['idx'].to(self.device)
        shape_idx = batch['shape_idx'].to(self.device) # start here
        
        # load base mesh

        base_points = base_mesh.verts_packed()
        base_faces = base_mesh.faces_packed()

        # latent codes
        shape_code = self.shape_codes[shape_idx]
        deform_code = self.deform_codes(idx)
        
        # forward 
        bone = self.bone_decoder(shape_code) 
        base_points = base_points.unsqueeze(0).repeat(deform_points.shape[0], 1, 1)
        shape_code = shape_code.repeat(deform_points.shape[0], 1)
        sw = self.SWpredictor(latent_code=shape_code, centers=base_points) # [b, N,B]
        rts_fw = self.Transpredictor(latent_code=deform_code, centers=self.bone.to(self.device)) # [b,B,12] or [b,B,7]
        v0 = self.deform_lbs(base_points,sw,rts_fw)

        new_mesh = Meshes(verts=v0, faces=base_faces.unsqueeze(0), textures=base_mesh.textures)
        loss_bone = 0
        if epoch < 20:
            loss_bone = chamfer_distance(base_points, bone.unsqueeze(0))[0]
        loss_chamfer = chamfer_distance(deform_points, v0)[0]
        # smooth loss 
        loss_edge = mesh_edge_loss(new_mesh)
        loss_smooth = mesh_laplacian_smoothing(new_mesh)
        # surface area loss
        loss_dict = {'shape': loss_chamfer, 'smooth': loss_smooth, 'edge': loss_edge, 'bone':loss_bone} # 'pme':loss_pme
        loss_total = 0
        for key in loss_dict.keys():
            loss_total += loss_dict[key] * self.cfg['lambdas'][key]
        loss_total.backward()
        self.optimizer_bone.step()
        self.optimizer_latent_deform.step()
        self.optimizer_sw.step()
        self.optimizer_trans.step()
        if epoch % 10 == 0:
            save_folder = self.cfg['save_result']
            os.makedirs(save_folder,exist_ok=True)
            save_file = os.path.join(save_folder, f'{base_name[0]}.obj')
            IO().save_mesh(new_mesh,save_file)
        printstr = "Epoch: {}, shape:{} ".format(epoch,base_name)
        for k in loss_dict:
            printstr += "{}: {:.4f} ".format(k, loss_dict[k])
        print(printstr)
        return loss_dict



    def deform_lbs(self, base_points, sw, rts_fw):
        """
        rts_fw:(b,K,7)
        sw: (b,N,K)
        bone: (B,K,3)
        
        """
        bone = self.bone.unsqueeze(0).expand(base_points.shape[0], -1, -1).to(self.device)
        B, N, K = sw.shape
        B, K, _ = bone.shape
        v0 = base_points.view(-1,3)
        disp = rts_fw[:,:,:3]
        rot = rts_fw[:,:,3:]
        rot = transforms.quaternion_to_matrix(rot.view(B*K, 4).contiguous()).view(B, K, 3, 3).contiguous()
        hd_disp = torch.repeat_interleave(disp, N, dim=0)
        hd_rot = torch.repeat_interleave(rot, N, dim=0)
        hd_pos = torch.repeat_interleave(bone, N, dim=0)
        per_hd_v = torch.einsum("abcd,abd->abc", hd_rot, (v0[:, None] - hd_pos)) + hd_pos + hd_disp  # (B*V, 40, 3)
        # per_hd_v = torch.einsum("abcd,abd->abc", hd_rot, (v0[:, None]))  + hd_disp  # (B*V, 40, 3)
        region_score = sw.view(-1, K)
        v = torch.sum(region_score[:, :, None] * per_hd_v, 1)  # (B*V, 3)
        return v.view(B, N, 3)
        
    def train(self):    
        loss = 0
        if self.args.continue_train:
            start = self.load_checkpoint()
        else:
            start =0
        ckpt_interval =self.cfg['ckpt_interval']
        save_name = self.cfg['save_name']
        for epoch in range(start,self.cfg['num_epochs']):
            sum_loss_dict = {k: 0.0 for k in self.cfg['lambdas']}
            sum_loss_dict.update({'loss':0.0})
            for batch in self.dataloader:
                loss_dict = self.train_step(epoch,batch)
            for k in loss_dict:
                sum_loss_dict[k] += loss_dict[k]
            
            if epoch == self.cfg['epoch_only_shape']:
                self.reinit_lr()
            self.reduce_lr(epoch)

            if epoch % ckpt_interval == 0 and epoch > 0:
                self.save_checkpoint(checkpoint_path=self.cfg['checkpoint_path'], save_name=f'epoch_{epoch}_{save_name}.pth')
            self.save_checkpoint(checkpoint_path=self.cfg['checkpoint_path'], save_name=f'latest_{save_name}.pth')
            # print result
            n_train = len(self.dataloader)
            for k in sum_loss_dict:
                sum_loss_dict[k] /= n_train
            # printstr = "Epoch: {} ".format(epoch)
            # for k in sum_loss_dict:
            #     printstr += "{}: {:.4f} ".format(k, sum_loss_dict[k])
            # print(printstr)

    def eval_latent_space(self):
        checkpoint = torch.load('checkpoints/cvpr/latest_deform_map.pth',map_location=self.device)
        self.SWpredictor.load_state_dict(checkpoint['SWpredictor'])
        self.Transpredictor.load_state_dict(checkpoint['Transpredictor'])
        self.SWpredictor.eval()
        self.Transpredictor.eval()
        bone = checkpoint['bone']
        deform_codes = checkpoint['deform_codes']['weight']
        deform_codes.to(self.device)
        # compute chamfer distance for each shape
        chamfer_dict = {}
        with open('chamfer_distances.json', 'r') as f:
            chamfer_dict = json.load(f)
        # umap for latent space
        deform_code_np = deform_codes.detach().cpu().numpy()
        # umap_model = umap.UMAP(n_neighbors=5, min_dist=0.001, n_components=2)
        # latent_2d = umap_model.fit_transform(deform_code_np)
        # distances is the norm of each latent code
        distances = np.linalg.norm(deform_code_np, axis=1)
        colors = distances/100
        # distances = np.array([chamfer_dict[k] for k in chamfer_dict])
        # min_value = np.percentile(distances, 5)  
        # max_value = np.percentile(distances, 95)  
        # colors = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
        plt.figure(figsize=(15, 15), dpi=500)        
        plt.scatter((latent_2d[:, 0] - np.mean(latent_2d[:, 0]))/np.max(latent_2d[:,0]), (latent_2d[:, 1] - np.mean(latent_2d[:, 1]))/np.max(latent_2d[:,1]), c=colors, cmap='viridis',s=90)
        cbar = plt.colorbar(label='Chamfer distance between base and deformed shapes')
        cbar.ax.tick_params(labelsize=30)  # 调整颜色条刻度标签的字体大小
        cbar.set_label('Chamfer distance between base and deformed shapes', fontsize=35, labelpad=20)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.savefig('umap_latent_space.svg',dpi=500)
        
        return chamfer_dict
    
def eval_metric(device):
    """
    eval chamfer distance and normal consistence of the deformed shapes
    """
    test_root = 'results/cvpr/fitting/train_deformleaf'
    dir =  'results/cvpr/fitting/'
    meshes = [os.path.join(test_root,f) for f in os.listdir(test_root) if f.endswith('.obj')]
    chamfer_npm_mean = 0
    chamfer_ours_mean = 0
    chamfer_pca_mean = 0
    nc_npm_mean = 0
    nc_ours_mean = 0
    nc_pca_mean = 0
    gt_num = 0
    for i,meshfile in enumerate(meshes):
        base_name = meshfile.split('/')[-1].split('.')[0]
        # read ball_pivoting
        ball_pivoting_name = base_name + '_deform.obj'
        if os.path.exists(os.path.join(dir,'ball_pivoting',ball_pivoting_name)):
            mesh_gt = load_objs_as_meshes([os.path.join(dir,'ball_pivoting',ball_pivoting_name)],device= device)
            gt_num +=1
        #  read npm
        mesh_npm = load_objs_as_meshes([meshfile], device= device)
        # read ours 
        direct_name = base_name + '_direct.obj'
        dirct_file = os.path.join(dir,'direct', direct_name)
        mesh_ours = load_objs_as_meshes([dirct_file], device= device)
        # read cpd 
        try:
            cpd_name = base_name + '.obj'
            cpd_file = os.path.join(dir,'cpd', cpd_name)
            mesh_pcd = load_objs_as_meshes([cpd_file],device= device)
        except:
            cpd_name = base_name + '_deform.obj'
            cpd_file = os.path.join(dir,'cpd', cpd_name)
            mesh_pcd = load_objs_as_meshes([cpd_file],device= device)

        # chamfer distance 
        chamfer_npm = chamfer_distance(mesh_gt.verts_packed().unsqueeze(0), mesh_npm.verts_packed().unsqueeze(0))[0]
        chamfer_ours = chamfer_distance(mesh_gt.verts_packed().unsqueeze(0), mesh_ours.verts_packed().unsqueeze(0))[0]
        chamfer_pca = chamfer_distance(mesh_gt.verts_packed().unsqueeze(0), mesh_pcd.verts_packed().unsqueeze(0))[0]
        chamfer_npm_mean += chamfer_npm
        chamfer_ours_mean += chamfer_ours
        chamfer_pca_mean += chamfer_pca
        # normal consistency
        normal_npm = mesh_normal_consistency(mesh_npm)
        normal_ours = mesh_normal_consistency(mesh_ours)
        normal_pca = mesh_normal_consistency(mesh_pcd)
        nc_npm_mean += normal_npm
        nc_ours_mean += normal_ours
        nc_pca_mean += normal_pca
        print("shape: {}, chamfer distance npm: {:.4f}, ours: {:.4f}, pca: {:.4f}".format(base_name, chamfer_npm, chamfer_ours, chamfer_pca))
    chamfer_npm_mean /= gt_num
    chamfer_ours_mean /= gt_num
    chamfer_pca_mean /= gt_num
    nc_npm_mean /= gt_num
    nc_ours_mean /= gt_num
    nc_pca_mean /= gt_num
    print(f'chamfer distance npm: {chamfer_npm_mean}, ours: {chamfer_ours_mean}, pca: {chamfer_pca_mean}')
    print(f'normal consistency npm: {nc_npm_mean}, ours: {nc_ours_mean}, pca: {nc_pca_mean}')
            
            
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RUN Leaf NPM')
    parser.add_argument('--gpu', type=int, default=2, help='gpu index')
    parser.add_argument('--continue_train', action='store_true', help='continue training from latest checkpoint')
    parser.add_argument('--config', type=str, default='scripts/configs/deform.yaml', help='config file')
    parser.add_argument('--name', type=str, default='mlp', help='experiment name')
    parser.add_argument('--use_arap', action='store_true', help='use arap loss')
    # fix seed 
    torch.manual_seed(0)
    np.random.seed(0)
    # setting
    args = parser.parse_args()
    device = torch.device(f'cuda:{str(args.gpu)}' if torch.cuda.is_available() else 'cpu')
    CFG = yaml.safe_load(open(args.config, 'r')) 
    # load model & dataset
    dataset = DeformLeafDataset()
    shape_decoder = UDFNetwork(d_in=CFG['Base']['Z_DIM'],
                         d_hidden=CFG['Base']['decoder_hidden_dim'],
                         d_out=CFG['Base']['decoder_out_dim'],
                         n_layers=CFG['Base']['decoder_nlayers'],
                         udf_type='sdf',
                         geometric_init=False) 
    
    trainer = DeformTrainer(shape_decoder,dataset,CFG, device, args)
    # trainer.generate_base_mesh()
    # trainer.create_similarity_dictionary(save_file='similarity_dict.json')
    trainer.train()
    # eval_metric(device)    
    # trainer.eval_latent_space()