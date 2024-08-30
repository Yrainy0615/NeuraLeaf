import numpy as np
from pytorch3d import transforms
import torch
import torch.nn as nn
import sys
import os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir) 
sys.path.insert(0, parent_dir)
from utils import geom_utils as gutils
import yaml
import trimesh
from pytorch3d.io import load_obj
from pytorch3d.loss import chamfer_distance


class NBS():
    def __init__(self, opts):
        """
        bone: ...,B,10  -B gaussian ellipsoids
        pts: bs,N,3  -N points
        skin: bs,N,B -B skinning matrix
        rts: bs,B,3,4   - B ririd transforms, applied to bone coordinates (points attached to bones in world coords)
        dskin: bs,N,B - delta skinning matrix

        """
        self.latent_lbs = None
        self.bone = None 
        self.mlp_rts = None
        self.num_bones = opts['num_bones']
        self.skin_aux = None
        self.dskin = None
        self.opts = opts
        self.rts_fw = None 
        self.skin_aux = torch.Tensor([0, 2])  

        # self.intitialize(self.canonical_pts)
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def intitialize(self, pts):
        self.generate_bone(pts)
        self.dskin = torch.zeros(pts.shape[0], pts.shape[1], self.num_bones)
        R_init = torch.eye(3).repeat(self.num_bones,1,1)
        t_init = torch.zeros(self.num_bones,3,1)
        rts_fw = torch.cat([R_init,t_init],-1)
        self.rts_fw = nn.Parameter(rts_fw)
        skin_aux = torch.Tensor([0, 2])  
        self.skin_aux = nn.Parameter(skin_aux)
        self.optimizer = torch.optim.Adam(
            [
                {'params': [self.bone],'lr': 0.1},
                {'params': [self.rts_fw],'lr': 0.01},
                {'params': [self.skin_aux],'lr': 0.01},
                {'params': [self.dskin],'lr': 0.01}
             ]
        )
    
    def bone_prior(self, pts,num_bones):
        random_idx = torch.randint(0, pts.shape[1], (num_bones,))
        center = pts[:,random_idx]
        return center
    def generate_bone(self, pts, vis=False):
        """
        pts(canonical): bs,N,3 
        PCA for initial bone estimation of canonical shape
        """
        bs,N,_ = pts.shape
  
        # PCA
        pts_mean = pts.mean(dim=1, keepdim=True)
        pts_centered = pts - pts_mean
        # random select n points
        random_idx = torch.randint(0, N, (self.num_bones,))
        center = pts[:,random_idx,:]
        # center =  torch.linspace(-1, 1, self.num_bones)
        # center =torch.meshgrid(center, center, center)
        # center = torch.stack(center,0).permute(1,2,3,0).reshape(-1,3)
        # center = center[:num_bones]
        # U, S, V = torch.svd(pts_centered)
        # center = U[:,:self.num_bones]
        # other parameters
        orient = torch.Tensor([1,0,0,0])
        orient = orient.repeat(self.num_bones,1)
        scale = torch.zeros(self.num_bones,3)
        bone = torch.cat([center.squeeze(),orient,scale],-1)
        if vis:
            gutils.vis_points(bone.squeeze())
        self.bone = nn.Parameter(bone.unsqueeze(0))
    
    def warp_bw(self,pts):
        bone_rst = self.bone
        
        pass
    
    def warp_fw(self,pts, bone,):
        """
        canonical shape -> deformed shape
        """
        bone_canonical = bone
        self.optimizer.zero_grad()
        # skinning matrix
        skin_fw = self.skinning(bone_canonical, self.canonical_pts, dskin=self.dskin, skin_aux=self.skin_aux)
        # lbs 
        deformed_pred, bones_dfm = self.lbs(bone_canonical, self.rts_fw.unsqueeze(0), skin_fw, backward=False)

        # if i%10==0:
        #     save_dir = f"results/deform/lbs_test/{self.num_bones}bones"
        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir)
        #     deformed_mesh.export(os.path.join(save_dir, f"deformed_{i}_maple.ply"))
        
        
        return deformed_pred, bones_dfm
        
        
    def lbs(self,bones,pts,rts_fw,skin,backward=False):
        """
        bones: bs,B,10       - B gaussian ellipsoids indicating rest bone coordinates
        rts_fw: bs,B,12       - B rigid transforms, applied to the rest bones
        xyz_in: bs,N,3       - N 3d points after transforms in the root coordinates
        """
        B = bones.shape[-2]
        N = pts.shape[-2]
        bs = rts_fw.shape[0]
        bones = bones.view(-1,B,10)
        xyz_in = pts.view(-1,N,3)
        # rts_fw = rts_fw.view(-1,B,12)# B,12
        # rmat=rts_fw[:,:,:9]
        # rmat=rmat.view(bs,B,3,3)
        # tmat= rts_fw[:,:,9:12]
        # rts_fw = torch.cat([rmat,tmat[...,None]],-1)
        # rts_fw = rts_fw.view(-1,B,3,4)

        if backward:
            bones_dfm = self.bone_transform(bones, rts_fw) # bone coordinates after deform
            rts_bw = gutils.rts_invert(rts_fw)
            xyz = self.blend_skinning(bones_dfm, rts_bw, skin, xyz_in)
        else:
            xyz = self.blend_skinning(bones.repeat(bs,1,1), rts_fw, skin, xyz_in)
            bones_dfm = gutils.bone_transform(bones, rts_fw) # bone coordinates after deform
        return xyz, bones_dfm
    
    def blend_skinning(self, bones,rts,skin,pts):       
        """
        bone: bs,B,10   - B gaussian ellipsoids
        rts: bs,B,3,4   - B ririd transforms, applied to bone coordinates
        pts: bs,N,3     - N 3d points
        skin: bs,N,B   - skinning matrix
        apply rts to bone coordinates, while computing blending globally
        """
        chunk=pts.shape[1]
        B = rts.shape[-3]
        N = pts.shape[-2]
        bones = bones.view(-1,B,10)
        pts = pts.view(-1,N,3)
        rts = rts.view(-1,B,3,4)
        bs = pts.shape[0]

        pts_out = []
        for i in range(0,bs,chunk):
            pts_chunk = self.blend_skinning_chunk(bones[i:i+chunk], rts[i:i+chunk], 
                                            skin[i:i+chunk], pts[i:i+chunk])
            pts_out.append(pts_chunk)
        pts = torch.cat(pts_out,0)
        return pts
    
    def blend_skinning_chunk(self, bone,rts,skin,pts):
        """
        bone: bs,B,10   - B gaussian ellipsoids
        rts: bs,B,3,4   - B ririd transforms, applied to bone coordinates (points attached to bones in world coords)
        pts: bs,N,3     - N 3d points
        skin: bs,N,B   - skinning matrix
        apply rts to bone coordinates, while computing blending globally
        """
        B = rts.shape[-3]
        N = pts.shape[-2]
        pts = pts.view(-1,N,3)
        rts = rts.view(-1,B,3,4)
        Rmat = rts[:,:,:3,:3] # bs, B, 3,3
        Tmat = rts[:,:,:3,3]
        device = Tmat.device
        
        Rmat_w = (skin[...,None,None] * Rmat[:,None]).sum(2) # bs,N,B,3
        Tmat_w = (skin[...,None] * Tmat[:,None]).sum(2) # bs,N,B,3
        pts = Rmat_w.matmul(pts[...,None]) + Tmat_w[...,None] 
        pts = pts[...,0]
        return pts
            
    def skinning(self, bones, pts, dskin=None, skin_aux=None):
        """
        bone: ...,B,10  - B gaussian ellipsoids
        pts: bs,N,3    - N 3d points
        skin: bs,N,B   - skinning matrix
        """
        chunk=pts.shape[1]
        bs,N,_ = pts.shape
        B = bones.shape[-2]
        if bones.dim()==2: bones = bones[None].repeat(bs,1,1)
        bones = bones.view(-1,B,10)

        skin = []
        for i in range(0,bs,chunk):
            if dskin is None:
                dskin_chunk = None
            else: 
                dskin_chunk = dskin[i:i+chunk]
            skin_chunk = self.skinning_chunk(bones[i:i+chunk], pts[:,i:i+chunk], \
                                dskin=dskin_chunk, skin_aux=skin_aux)
            skin.append( skin_chunk )
        skin = torch.cat(skin,0)
        return skin

    def skinning_chunk(self, bones, pts, dskin=None, skin_aux=None):
        """
        bone: bs,B,10  - B gaussian ellipsoids
        pts: bs,N,3    - N 3d points, usually N=num points per ray, b~=2034
        skin: bs,N,B   - skinning matrix
        """
        device = pts.device
        log_scale= skin_aux[0]
        w_const  = skin_aux[1]
        bs,N,_ = pts.shape
        B = bones.shape[-2]
        if bones.dim()==2: bones = bones[None].repeat(bs,1,1)
        bones = bones.view(-1,B,10)
    
        center, orient, scale = gutils.vec_to_sim3(bones) 
        orient = orient.permute(0,1,3,2) # transpose R

        # mahalanobis distance [(p-v)^TR^T]S[R(p-v)]
        # transform a vector to the local coordinate
        mdis = center.view(bs,1,B,3) - pts.view(bs,N,1,3) # bs,N,B,3
        mdis = gutils.axis_rotate(orient.view(bs,1,B,3,3), mdis[...,None])
        mdis = mdis[...,0]
        mdis = scale.view(bs,1,B,3) * mdis.pow(2)
        # log_scale (being optimized) controls temporature of the skinning weight softmax 
        # multiply 1000 to make the weights more concentrated initially
        inv_temperature = 1000 * log_scale.exp()
        mdis = (-inv_temperature * mdis.sum(3)) # bs,N,B

        if dskin is not None:
            mdis = mdis+dskin

        skin = mdis.softmax(2)
        return skin

if __name__ == '__main__':
    # load data
    canonical_file = "dataset/deformation_eccv/maple4_d1_aligned.obj"
    deformed_file = "dataset/deformation_eccv/maple4_d8_aligned.obj"
    opts_file = "scripts/configs/deform.yaml"
    cfg = yaml.load(open(opts_file, 'r'), Loader=yaml.FullLoader)
    canonical_mesh = load_obj(canonical_file)
    deformed_mesh = load_obj(deformed_file)
    canonical_points = canonical_mesh[0]
    deformed_points = deformed_mesh[0]
    
    # load model
    nbs_model = NBS(cfg['NBS'], canonical_points)
    
    # forward
    deformed_pred, bones_dfm = nbs_model.warp_fw(deformed_points, epoch=100)
    """
    TO-DO:
    1. add surface constraint during optimization
    2. optimization -> training 
    3. learn latent space
    4. neuralbs + displacement field 
    """
    
    pass

    