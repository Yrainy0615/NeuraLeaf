import os
import trimesh
import numpy as np
from data_utils import data_processor
import cv2
from pytorch3d.io import load_ply, load_obj, save_ply
from pytorch3d.structures import Meshes
from pytorch3d.loss import chamfer_distance
import torch
from probreg import cpd
from pytorch3d.transforms import RotateAxisAngle, Translate, Scale

class MeshProcessor(data_processor):
    def __init__(self, root_dir):
        super().__init__(root_dir)
        self.all_deformed_mesh = [os.path.join(root_dir, 'deformation_final', i) for i in os.listdir(os.path.join(root_dir, 'deformation_final')) if 'canonical' not in i]
        self.all_deformed_mesh.sort()
    
    def normalize(self, verts:torch.tensor):
        centroid = verts.mean(dim=0)
        # translate and sclae verts to canonical space
        normalized_verts = verts - centroid
        scale = torch.max(torch.abs(normalized_verts)).item()
        scale_factor = 1.0 / scale
        scale_transform = Scale(scale_factor)
        normalized_verts = scale_transform.transform_points(normalized_verts)

        return normalized_verts
    
    def uv_mapping(self,mesh, texture_image):
        uvs = np.zeros((mesh.vertices.shape[0], 2), dtype=np.float32)
        texture_height, texture_width, _ = texture_image.shape

        # 映射顶点坐标到纹理图像坐标
        for i, (x, y, _) in enumerate(mesh.vertices):
            u = int(np.clip(x, 0, texture_width - 1)) / texture_width
            v = int(np.clip(y, 0, texture_height - 1)) / texture_height
            uvs[i] = [u, 1 - v]  
        texture = trimesh.visual.TextureVisuals(uv=uvs, image=texture_image)
        mesh.visual = texture
        return mesh
    
    def find_paired_deformed_mesh(self,basepoints,num=3):
        chamfer_list = []
        for i, deformed_file in enumerate(self.all_deformed_mesh):
            deformed = load_obj(deformed_file)
            deformed_points = deformed[0]
            chamfer_list.append(chamfer_distance(basepoints.unsqueeze(0), deformed_points.unsqueeze(0))[0])
        chamfer_list = torch.stack(chamfer_list)
        _, indices = torch.topk(chamfer_list, num)
        return indices

        
    def nonrigid_cpd_cuda(self, basepoints:torch.tensor, deformed_points:torch.tensor, use_cuda=True):
        import cupy as cp
        if use_cuda:
            to_cpu = cp.asnumpy
            cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
        else: 
            cp = np
            to_cpu = lambda x: x
        source_pt = cp.asarray(basepoints)
        target_pt = cp.asarray(deformed_points)
        acpd = cpd.NonRigidCPD(source_pt, use_cuda=use_cuda)
        tf_param, _ ,_ = acpd.registration(target_pt)
        result = tf_param.transform(source_pt)
        # registrated_mesh = Meshes(verts=[result], faces=[base.faces_packed()])
        return torch.tensor(result)


if __name__ == "__main__":
    root = 'dataset/'
    device = 'cuda:0'
    processor = MeshProcessor(root)
    # test uv mapping 
    mesh = trimesh.load(processor.all_base_mesh[0])
    texture = cv2.imread(processor.all_rgb[0])
    save_folder = os.path.join(root,'deformation_pairs')
    canonical_folder = os.path.join(save_folder,'canonical')
    deformed_folder = os.path.join(save_folder,'deformed')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(canonical_folder):
        os.makedirs(canonical_folder)
    if not os.path.exists(deformed_folder):
        os.makedirs(deformed_folder)
    # textured_mesh = processor.uv_mapping(mesh, texture)
    # get base-deform pair
    for i in range(len(processor.all_base_mesh)):
        base = load_ply(processor.all_base_mesh[i])
        base_points = base[0]
        base_points_normalized = processor.normalize(base_points)
        base_points_normalized[torch.isnan(base_points_normalized)] = 0
        new_base = Meshes(verts=[base_points_normalized.to(device)], faces=[base[1].to(device)])
        save_ply(os.path.join(canonical_folder,f'base_{i}.ply'), new_base.verts_list()[0], new_base.faces_list()[0])
        shape_idx = processor.find_paired_deformed_mesh(base_points_normalized)

        
        for j in shape_idx:
            # non-rigid registration
            deformed = load_obj(processor.all_deformed_mesh[j])
            # rotate deformed mesh
            
            deformed_points = deformed[0]
            deformed_points_rot = RotateAxisAngle(90,axis='z').transform_points(deformed_points)
            # test rotate
            # deformed_rot_mesh = Meshes(verts=[deformed_points_rot.to(device)], faces=[deformed[1].verts_idx.to(device)])
            # save_ply(f'deformed_rot_{i}.ply', deformed_rot_mesh.verts_list()[0], deformed_rot_mesh.faces_list()[0])
            registrated_points = processor.nonrigid_cpd_cuda(base_points_normalized, deformed_points)
            registrated_mesh = Meshes(verts=[registrated_points.to(device)], faces=[base[1].to(device)])
        
            # save registrated mesh
            save_ply(os.path.join(deformed_folder,f'{i}_{j}.ply'), registrated_mesh.verts_list()[0], registrated_mesh.faces_list()[0])
            print(f"save {i}_{j}.ply")
        
    pass
        