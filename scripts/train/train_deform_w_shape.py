import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import torch
from scripts.data.dataset import DeformLeafDataset
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
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

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
        # load shape codes
        checkpoint_base = torch.load(cfg['Trainer']['shape_checkpoint'],map_location=device)
        shape_codes = checkpoint_base['latent_shape']['weight'].to(self.device)
        self.shape_decoder.load_state_dict(checkpoint_base['decoder'])
        self.shape_decoder.eval().to(self.device)
        # init 
        self.deform_codes = torch.nn.Embedding(len(self.dataset), cfg['NBS']['lat_dim'], max_norm=1, device=device).requires_grad_(True)
        if len(self.dataset.deform_base_masks) < len(shape_codes):
            self.shape_codes = shape_codes[:len(self.dataset.deform_base_masks),:].to(self.device)
        else:
            self.shape_codes = shape_codes.to(self.device)
        # models
        self.lbs_model = NBS(cfg['NBS'])
        latent_dim = cfg['NBS']['lat_dim']  # Get latent dimension from config
        self.SWpredictor = SWPredictor(num_bones=self.num_bones, latent_dim=cfg['Base']['Z_DIM']).to(device)
        self.Transpredictor = TransPredictor(num_bones=self.num_bones, latent_dim=latent_dim).to(device)
        self.bone_decoder = BoneDecoder(num_bones=self.num_bones).to(device)
        # optimizers
        self.optimizer_decoder = torch.optim.Adam(self.shape_decoder.parameters(), lr=self.cfg['LR_D'])
        self.optimizer_latent_deform = torch.optim.Adam(self.deform_codes.parameters(), lr=self.cfg['LR_LAT'])
        self.k = torch.nn.Parameter(torch.tensor(1.0))
        self.optimizer_k  = torch.optim.Adam([self.k], lr=0.01)
        self.optimizer_bone = torch.optim.Adam([self.bone], lr=0.001)
        self.optimizer_bone_decoder = torch.optim.Adam(self.bone_decoder.parameters(), lr=self.cfg['LR_B'])
        self.optimizer_sw = torch.optim.Adam(self.SWpredictor.parameters(), lr=self.cfg['LR_SW'])
        self.optimizer_trans = torch.optim.Adam(self.Transpredictor.parameters(), lr=self.cfg['LR_D'])
        self.phi = torch.nn.Parameter(torch.tensor(1.0))
        self.optimizer_phi = torch.optim.Adam([self.phi], lr=0.01)

        
        # Initialize TensorBoard writer
        if args.use_tensorboard:
            log_dir = os.path.join('logs', 'tensorboard', f"deform_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
            self.global_step = 0
            print(f"TensorBoard logging enabled. Log directory: {log_dir}")
        else:
            self.writer = None

   
    def init_bones(self, mode='random'):
        """
        Initialize GLOBAL bone positions on a base plane.
        
        According to the paper, bone positions are:
        - Global (shared across all shapes)
        - Located on a base plane (z=0)
        - Uniformly sampled in UV space (xy plane)
        
        Args:
            mode: 'random' - random initialization on base plane
                  'prior' - initialize from prior (not commonly used)
        """
        if mode == 'random':
            # Sample bone positions uniformly on base plane (z=0)
            xy = np.random.uniform(-1, 1, (self.num_bones, 2))
            z = np.zeros((self.num_bones, 1))
            bone = np.concatenate([xy, z], axis=1)
            bone_tensor = torch.Tensor(bone)
            self.bone = torch.nn.Parameter(bone_tensor)
        elif mode == 'prior':
            # Initialize from prior bone generation
            for batch in self.dataloader:
                base_points = batch['base_points'].to(self.device)
                bone_tensor = self.lbs_model.generate_bone()
                self.bone = torch.nn.Parameter(bone_tensor[:, :3])
                break

    
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

        
    def save_checkpoint(self, checkpoint_path, save_name):
        checkpoint = {
            'deform_codes': self.deform_codes.state_dict(),
            'bone': self.bone,
            'SWpredictor': self.SWpredictor.state_dict(),
            'Transpredictor': self.Transpredictor.state_dict(),
            'phi': self.phi,
            'bone_decoder': self.bone_decoder.state_dict(),
            }
        torch.save(checkpoint, os.path.join(checkpoint_path, save_name))

                                        
    def train_step(self, epoch, batch):
        """
        Standard training step for deformation space.
        
        Key features:
        - Uses GLOBAL bone positions (self.bone) shared across all shapes
        - Bone positions are initialized on a base plane (z=0) and optimized during training
        - Skinning weights are shape-dependent (via SWPredictor with shape_code)
        - Transformations are deformation-dependent (via TransPredictor with deform_code)
        - Includes two regularizers: latent code L2 norm and deformation mapping loss
        - Optimizes phi parameter for deformation mapping loss
        
        This follows the paper's design where:
        - Bone positions are global control points on a base plane
        - Only skinning weights adapt to different leaf shapes
        """
        self.optimizer_bone.zero_grad()
        self.optimizer_latent_deform.zero_grad()
        self.optimizer_sw.zero_grad()
        self.optimizer_trans.zero_grad()
        self.optimizer_phi.zero_grad()
        
        # Load data
        deform_points = batch['deform_points'].to(self.device)
        base_name = batch['base_name']
        base_file = batch['base_mesh']
        base_mesh = load_objs_as_meshes(base_file, device=self.device)
        idx = batch['idx'].to(self.device)
        shape_idx = batch['shape_idx'].to(self.device)
        
        base_points = base_mesh.verts_packed()
        base_faces = base_mesh.faces_packed()

        # Get latent codes
        shape_code = self.shape_codes[shape_idx]
        deform_code = self.deform_codes(idx)
        
        # Forward pass
        base_points = base_points.unsqueeze(0).repeat(deform_points.shape[0], 1, 1)
        shape_code = shape_code.repeat(deform_points.shape[0], 1)
        sw = self.SWpredictor(latent_code=shape_code, centers=base_points)  # [b, N, B] - shape-dependent
        rts_fw = self.Transpredictor(latent_code=deform_code, centers=self.bone.to(self.device))  # [b, B, 7] - uses GLOBAL bone
        v0 = self.deform_lbs(base_points, sw, rts_fw)

        new_mesh = Meshes(verts=v0, faces=base_faces.unsqueeze(0), textures=base_mesh.textures)

        loss_chamfer = chamfer_distance(deform_points, v0)[0]
        # smooth loss 
        loss_edge = mesh_edge_loss(new_mesh)
        loss_smooth = mesh_laplacian_smoothing(new_mesh)
        
        # Regularizer 1: Latent code regularization (L2 norm)
        if self.args.use_latent_reg:
            lat_reg_deform = deform_code.norm(2, dim=1).mean()
        else:
            lat_reg_deform = 0.0
        
        # Regularizer 2: Deformation mapping loss
        if self.args.use_deform_mapping:
            loss_map = deformation_mapping_loss(loss_chamfer, deform_code, phi=self.phi)
        else:
            loss_map = 0.0
        
        # surface area loss
        loss_dict = {
            'shape': loss_chamfer, 
            'smooth': loss_smooth, 
            'edge': loss_edge,
            'reg_deform': lat_reg_deform,
            'map': loss_map
        }
        loss_total = 0
        for key in loss_dict.keys():
            loss_total += loss_dict[key] * self.cfg['lambdas'].get(key, 0.0)
        loss_total.backward()
        self.optimizer_bone.step()
        self.optimizer_latent_deform.step()
        self.optimizer_sw.step()
        self.optimizer_trans.step()
        self.optimizer_phi.step()
        
        # Log to TensorBoard
        if self.writer is not None:
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    self.writer.add_scalar(f'Train/{key}', value.item(), self.global_step)
                else:
                    self.writer.add_scalar(f'Train/{key}', value, self.global_step)
            # Log total loss
            self.writer.add_scalar('Train/loss_total', loss_total.item(), self.global_step)
            self.writer.add_scalar('Train/phi', self.phi.item(), self.global_step)
            self.global_step += 1
        
        printstr = "Epoch: {}, shape:{} ".format(epoch,base_name)
        for k in loss_dict:
            printstr += "{}: {:.4f} ".format(k, loss_dict[k])
        print(printstr)
        return loss_dict

    def train_step_bone(self, epoch, batch):
        """
        Training step with bone decoder (NOT aligned with paper's standard setting).
        
        WARNING: This method does NOT follow the paper's design. The paper uses
        GLOBAL bone positions shared across all shapes. This method attempts to
        use shape-dependent bone positions via bone_decoder, but the predicted
        bone is actually NOT used in the forward pass (still uses self.bone).
        
        According to the paper:
        - Bone positions should be GLOBAL (shared across all shapes)
        - Bone positions are on a base plane (z=0)
        - Only skinning weights are shape-dependent
        
        Use train_step() instead for standard deformation training with global bone positions.
        """
        self.optimizer_bone_decoder.zero_grad()
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
        
        # Forward pass
        base_points = base_points.unsqueeze(0).repeat(deform_points.shape[0], 1, 1)
        shape_code = shape_code.repeat(deform_points.shape[0], 1)
        sw = self.SWpredictor(latent_code=shape_code, centers=base_points)  # [b, N, B]
        # use predicted bone
        bone_pred = self.bone_decoder(shape_code.repeat(deform_points.shape[0], 1))
        rts_fw = self.Transpredictor(latent_code=deform_code, centers=bone_pred)  # [b, B, 7]
        v0 = self.deform_lbs(base_points,sw,rts_fw)

        new_mesh = Meshes(verts=v0, faces=base_faces.unsqueeze(0), textures=base_mesh.textures)
        loss_chamfer = chamfer_distance(deform_points, v0)[0]
        # smooth loss 
        loss_edge = mesh_edge_loss(new_mesh)
        loss_smooth = mesh_laplacian_smoothing(new_mesh)
        # deformation mapping loss 
        if self.args.use_deform_mapping:
            loss_map = deformation_mapping_loss(loss_chamfer, deform_code, phi=self.phi)
        else:
            loss_map = 0.0
        # latent reg
        if self.args.use_latent_reg:
            lat_reg_deform = deform_code.norm(2, dim=1).mean()
        else:
            lat_reg_deform = 0.0
        loss_dict = {'shape': loss_chamfer, 'smooth': loss_smooth, 'edge': loss_edge, 'map': loss_map, 'reg_deform': lat_reg_deform} 
        loss_total = 0
        for key in loss_dict.keys():
            loss_total += loss_dict[key] * self.cfg['lambdas'][key]
        loss_total.backward()
        self.optimizer_latent_deform.step()
        self.optimizer_sw.step()
        self.optimizer_trans.step()
        self.optimizer_phi.step()
        self.optimizer_bone_decoder.step()
        
        # Log to TensorBoard
        if self.writer is not None:
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    self.writer.add_scalar(f'Train/{key}', value.item(), self.global_step)
                else:
                    self.writer.add_scalar(f'Train/{key}', value, self.global_step)
            # Log total loss
            self.writer.add_scalar('Train/loss_total', loss_total.item(), self.global_step)
            self.writer.add_scalar('Train/phi', self.phi.item(), self.global_step)
            self.global_step += 1
        
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
        
    def load_checkpoint(self):
        """Load checkpoint for continuing training"""
        checkpoint_path = os.path.join(self.cfg['checkpoint_path'], f"latest_{self.cfg['save_name']}.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.deform_codes.load_state_dict(checkpoint['deform_codes'])
            self.bone.data = checkpoint['bone'].to(self.device)
            self.SWpredictor.load_state_dict(checkpoint['SWpredictor'])
            self.Transpredictor.load_state_dict(checkpoint['Transpredictor'])
            if 'phi' in checkpoint:
                self.phi.data = checkpoint['phi'].to(self.device)
            print(f"Loaded checkpoint from {checkpoint_path}")
            return checkpoint.get('epoch', 0)
        return 0
    
    def train(self):    
        if self.args.continue_train:
            start = self.load_checkpoint()
        else:
            start = 0
        ckpt_interval =self.cfg['ckpt_interval']
        save_name = self.cfg['save_name']
        for epoch in range(start,self.cfg['num_epochs']):
            sum_loss_dict = {k: 0.0 for k in self.cfg['lambdas']}
            sum_loss_dict.update({'loss':0.0})
            for batch in self.dataloader:
                if self.args.use_bone_decoder:
                    loss_dict = self.train_step_bone(epoch,batch)
                else:
                    loss_dict = self.train_step(epoch,batch)
            for k in loss_dict:
                sum_loss_dict[k] += loss_dict[k]
            
            self.reduce_lr(epoch)

            if epoch % ckpt_interval == 0 and epoch > 0:
                self.save_checkpoint(checkpoint_path=self.cfg['checkpoint_path'], save_name=f'epoch_{epoch}_{save_name}.pth')
            self.save_checkpoint(checkpoint_path=self.cfg['checkpoint_path'], save_name=f'latest_{save_name}.pth')
            
            # Compute average losses
            n_train = len(self.dataloader)
            for k in sum_loss_dict:
                sum_loss_dict[k] /= n_train
            
            # Log epoch-level losses to TensorBoard
            if self.writer is not None:
                for k in sum_loss_dict:
                    self.writer.add_scalar(f'Epoch/{k}', sum_loss_dict[k], epoch)

    
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
    parser.add_argument('--gpu', type=int, default=6, help='gpu index')
    parser.add_argument('--continue_train', action='store_true', help='continue training from latest checkpoint')
    parser.add_argument('--config', type=str, default='scripts/configs/deform.yaml', help='config file')
    parser.add_argument('--name', type=str, default='mlp', help='experiment name')
    parser.add_argument('-- ', action='store_true', help='use arap loss')
    parser.add_argument('--use_tensorboard', action='store_true', help='use tensorboard for logging')
    parser.add_argument('--use_deform_mapping', action='store_true', help='use deform mapping loss')
    parser.add_argument('--use_latent_reg', action='store_true', help='use latent regularization loss')
    parser.add_argument('--use_bone_decoder', action='store_true', help='use bone decoder')
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
    trainer.train()
    
    # Close TensorBoard writer
    if trainer.writer is not None:
        trainer.writer.close()
        print("TensorBoard writer closed.")
    
    # eval_metric(device)    
    # trainer.eval_latent_space()