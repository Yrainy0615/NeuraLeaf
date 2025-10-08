import pyrender
import trimesh
import numpy as np
import torch
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PointLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    SoftSilhouetteShader,
    )
from pytorch3d.renderer.cameras import OrthographicCameras
from matplotlib import pyplot as plt

class Mesh_visualizer:
    def __init__(self, img_size=512):
        self.img_size = img_size    
        self.renderer = pyrender.OffscreenRenderer(img_size, img_size)
        self.mesh = None
        self.scene = pyrender.Scene()
        self.camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
        s = np.sqrt(2)/2
        self.camera_pose = np.array([
            [0.0, -s,   s,   2],
            [1.0,  0, 0, 0.0],
            [0.0,  s,   s,   2],
            [0.0,  0.0, 0.0, 1.0],
        ])
      
        self.light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                            innerConeAngle=np.pi/16.0,
                         outerConeAngle=np.pi/6.0)
        self.scene.add(self.camera, pose=self.camera_pose)
        self.scene.add(self.light, pose=self.camera_pose)
    
    def set_input(self, verts, faces,bones):
        verts = verts.detach().cpu().numpy()
        faces = faces.detach().cpu().numpy()
        bones = bones.detach().cpu().numpy()
        if verts.shape[0]==1:
            verts = verts.squeeze()
        if bones.shape[0]==1:
            bones = bones.squeeze()
        mesh = trimesh.Trimesh(vertices=verts, faces=faces.squeeze())
        mesh.visual.vertex_colors = np.ones((mesh.vertices.shape[0], 4), dtype=np.uint8) * 255
        mesh.visual.vertex_colors[:,-1] = 128
        mesh_render = pyrender.Mesh.from_trimesh(mesh, smooth=True)
        colors = np.array([255, 0, 0, 255] * bones.shape[0]).reshape(bones.shape[0], 4)  
        point_cloud = pyrender.Mesh.from_points(points=bones, colors=colors)
        
        
        mesh_node = pyrender.Node(mesh=mesh_render, matrix=np.eye(4))
        bones_node = pyrender.Node(mesh=point_cloud, matrix=np.eye(4))
        self.scene.add_node(mesh_node) 
        self.scene.add_node(bones_node)
    
    def render(self):
        color, depth = self.renderer.render(self.scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
        depth = depth.astype(np.uint8)
        return color, depth
    
        
        
