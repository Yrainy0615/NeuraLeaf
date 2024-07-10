import os
import trimesh
import numpy as np
from data_utils import data_processor
import cv2
from pytorch3d.io import load_ply
from pytorch3d.structures import Meshes

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
        
        pass
        



if __name__ == "__main__":
    root = 'dataset/LeafData'
    processor = MeshProcessor(root)
    # test uv mapping 
    mesh = trimesh.load(processor.all_base_mesh[0])
    texture = cv2.imread(processor.all_rgb[0])
    # textured_mesh = processor.uv_mapping(mesh, texture)
    # get base-deform pair
    for i in range(len(processor.all_base_mesh)):
        processor.find_paired_deformed_mesh(processor.all_base_mesh[i])
        
    pass
        