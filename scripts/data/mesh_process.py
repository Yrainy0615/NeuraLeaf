import os
import trimesh
import numpy as np
from data_utils import data_processor
import cv2
from pytorch3d.io import load_ply
from pytorch3d.structures import Meshes
from pytorch3d.loss import chamfer_distance
import torch
from probreg import cpd
import cupy as cp

class MeshProcessor(data_processor):
    def __init__(self, root_dir):
        super().__init__(root_dir)
        self.all_deformed_mesh = None
    
    
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
    
    def find_paired_deformed_mesh(self,baseshape,num=3):
        base = load_ply(baseshape)
        base_points = base[0]
        chamfer_list = []
        for i, deformed in enumerate(self.all_deformed_mesh):
            deformed_points = deformed.verts_packed()
            chamfer_distance.append(chamfer_distance(base_points, deformed_points))
        chamfer_list = torch.stack(chamfer_list)
        _, indices = torch.topk(chamfer_list, num)
        return indices

        
    def nonrigid_cpd_cuda(self, base, deformed, use_cuda=True):
        if use_cuda:
            to_cpu = cp.asnumpy
            cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
        else: 
            cp = np
            to_cpu = lambda x: x
        source_pt = cp.asarray(base.verts_packed())
        target_pt = cp.asarray(deformed.verts_packed())
        acpd = cpd.NonRigidCPD(source_pt, use_cuda=use_cuda)
        tf_param, _ ,_ = acpd.registration(target_pt)
        result = tf_param.transform(source_pt)
        registrated_mesh = Meshes(verts=[result], faces=[base.faces_packed()])
        return registrated_mesh


if __name__ == "__main__":
    root = 'dataset/LeafData'
    processor = MeshProcessor(root)
    # test uv mapping 
    mesh = trimesh.load(processor.all_base_mesh[0])
    texture = cv2.imread(processor.all_rgb[0])
    # textured_mesh = processor.uv_mapping(mesh, texture)
    # get base-deform pair
    for i in range(len(processor.all_base_mesh)):
        shape_idx = processor.find_paired_deformed_mesh(processor.all_base_mesh[i])
        base = load_ply(processor.all_base_mesh[i])
        # non-rigid registration
        for j in shape_idx:
            deformed = load_ply(processor.all_deformed_mesh[j])
            registrated_mesh = processor.nonrigid_cpd_cuda(base, deformed)
        
        
    pass
        