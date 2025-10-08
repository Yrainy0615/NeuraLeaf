import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import torch
import cv2
import pymesh

# define color 

COLORS_HSV = np.zeros([64, 3])
ci = 0
for s in [170, 100]:
    for h in np.linspace(0, 179, 16):
        for v in [220, 128]:
            COLORS_HSV[ci] = [h, s, v]
            ci += 1
COLORS = cv2.cvtColor(COLORS_HSV[None].astype(np.uint8), cv2.COLOR_HSV2RGB)[0].astype(np.float32) / 255
COLORS_ALPHA = np.concatenate((COLORS, np.ones_like(COLORS[:, :1])), 1)

def generate_distinct_colors(n):
    hsv_colors = np.zeros((n, 3))
    hsv_colors[:, 0] = np.linspace(0, 1, n, endpoint=False)  # Hue
    hsv_colors[:, 1] = 1.0  # Saturation
    hsv_colors[:, 2] = 1.0  # Value
    rgb_colors = cv2.cvtColor((hsv_colors * 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    return rgb_colors / 255.0  # Normalize to [0, 1]

def generate_color_based_on_xy(handle_pos):
    """
    根据控制点的XY坐标生成颜色
    """
    # 归一化XY坐标到[0, 179]用于HSV色相
    normalized_xy = (handle_pos[:, :2] - handle_pos[:, :2].min(axis=0)) / (handle_pos[:, :2].max(axis=0) - handle_pos[:, :2].min(axis=0))
    hsv_colors = np.zeros((handle_pos.shape[0], 3), dtype=np.uint8)
    hsv_colors[:, 0] = (normalized_xy[:, 0] * 179).astype(np.uint8)  # Hue
    hsv_colors[:, 1] = 255  # Saturation
    hsv_colors[:, 2] = 255  # Value
    
    # 转换为RGB颜色
    rgb_colors = cv2.cvtColor(hsv_colors.reshape(-1, 1, 3), cv2.COLOR_HSV2RGB).reshape(-1, 3)
    return rgb_colors / 255.0  # 归一化到[0,1]

def compute_vertex_colors(skinning_weights, control_colors, top_n=30):
    """
    计算每个顶点的颜色，基于其影响最大的N个控制点的颜色和权重。

    参数：
    - skinning_weights: (N, B) 的数组，表示N个顶点和B个控制点的皮肤权重。
    - control_colors: (B, 3) 的数组，表示B个控制点的RGB颜色。
    - top_n: int，选择影响最大的前N个控制点。

    返回：
    - vertex_colors: (N, 3) 的数组，表示每个顶点的RGB颜色，范围在[0, 1]之间。
    """
    N, B = skinning_weights.shape
    # 获取每个顶点影响最大的前N个控制点的索引
    top_n_indices = np.argsort(skinning_weights, axis=1)[:, -top_n:]
    # 获取对应的权重
    top_n_weights = np.take_along_axis(skinning_weights, top_n_indices, axis=1)
    
    # 归一化权重
    top_n_weights_normalized = top_n_weights / (top_n_weights.sum(axis=1, keepdims=True) + 1e-8)
    
    # 获取对应的颜色
    top_n_colors = control_colors[top_n_indices]  # (N, top_n, 3)
    
    # 计算加权平均颜色
    vertex_colors = np.sum(top_n_weights_normalized[:, :, np.newaxis] * top_n_colors, axis=1)  # (N, 3)
    
    return vertex_colors

def visualize_handle(v, f, handle_pos=None, heatmap=None, save_path=None, norm=True):
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    if isinstance(f, torch.Tensor):
        f = f.detach().cpu().numpy()
    if isinstance(handle_pos, torch.Tensor):
        handle_pos = handle_pos.detach().cpu().numpy()
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()
    cube_v = np.array([[-1, -1, -1], [-1, -1, 1], [1, -1, 1], [1, -1, -1],
                       [-1, 1, -1], [-1, 1, 1], [1, 1, 1], [1, 1, -1],], dtype=np.float32) * 0.01
    cube_f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [7, 4, 6],
                       [2, 5, 1], [2, 6, 5], [0, 4, 3], [4, 7, 3],
                       [1, 5, 0], [5, 4, 0], [3, 6, 2], [3, 7, 6]])
    cube_c = np.tile(np.array([[2, 0.3, 0.3]]), (len(cube_v), 1))
    if heatmap is None:
        base_c = np.ones([v.shape[0], 4])
        if handle_pos is not None:
            base_c[:, 3] = 0.5
    else:
        if norm:
            heatmap = heatmap / np.max(heatmap)
        heatmap = heatmap * 10 - 5
        heatmap = 1/(np.exp(-heatmap)+1) # sigmoid
        heatmap = (1 - heatmap) * 120
        heatmap_hsv = np.ones([heatmap.shape[0], 3], dtype=np.uint8) * 255
        heatmap_hsv[:, 0] = heatmap.astype(np.uint8)
        heatmap_bgr = cv2.cvtColor(heatmap_hsv[None], cv2.COLOR_HSV2BGR)[0]
        base_c = heatmap_bgr.astype(float) / 255
        base_c = np.concatenate((base_c, np.ones_like(base_c[:, :1])*0.5), 1)
    vv, ff, vcc = [v],  [f], [base_c]
    if handle_pos is not None:
        for i, hp in enumerate(handle_pos):
            vv.append(cube_v + hp[None])
            ff.append(cube_f)
            if len(handle_pos) > 1:
                vcc.append(COLORS_ALPHA[i%1000])
            else:
                vcc.append(np.array([1., 1., 1., 1.]))
    if save_path is None:
        return vv, ff, vcc
    else:
        vv, ff, vcc = merge_mesh(vv, ff, vcc)
        MeshO3d(v=vv, f=ff, vc=vcc[:, :3]).write_ply(save_path)
        return vv, ff, vcc
    
def visualize_deform_part(v, f, handle_pos, score, save_path=None):
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().squeeze().numpy()
    if isinstance(f, torch.Tensor):
        f = f.detach().cpu().squeeze().numpy()
    if isinstance(handle_pos, torch.Tensor):
        handle_pos = handle_pos.detach().cpu().numpy()
    if isinstance(score, torch.Tensor):
        score = score.detach().cpu().squeeze().numpy()
    if not isinstance(score, np.ndarray):
        score = np.array(score.todense())
    
    # Small cubes to represent control points
    cube_v = np.array([[-1, -1, -1], [-1, -1, 1], [1, -1, 1], [1, -1, -1],
                    [-1, 1, -1], [-1, 1, 1], [1, 1, 1], [1, 1, -1]], dtype=np.float32) * 0.01
    cube_f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [7, 4, 6],
                    [2, 5, 1], [2, 6, 5], [0, 4, 3], [4, 7, 3],
                    [1, 5, 0], [5, 4, 0], [3, 6, 2], [3, 7, 6]])
    cube_c = np.tile(np.array([[2, 0.3, 0.3]]), (len(cube_v), 1))
    
    num_part = score.shape[1]
    if num_part >= 64:
        # concatenate for 16 times
        # colors = COLORS_ALPHA
        # for i in range(15):
        #     colors = np.concatenate((colors, COLORS_ALPHA), 0)[:num_part]
        # colors = np.concatenate((colors, COLORS_ALPHA), 0)[:num_part]
        colors = generate_color_based_on_xy(handle_pos)
    #     # add a alpha channel 
    #     colors = np.concatenate((colors, np.ones_like(colors[:, :1])), 1)
    # else:
    #     colors = COLORS_ALPHA[:num_part]
    # base_c = np.sum(score[:, :, None] * colors[None], 1)  # [N, 4]
    # base_c[:, 3] = 0.5
    vertex_color = compute_vertex_colors(score, colors, top_n=30)
    vv, ff, vcc = [v], [f], vertex_color
    # if handle_pos is not None:
    #     for i, hp in enumerate(handle_pos):
    #         vv.append(cube_v + hp[None])
    #         ff.append(cube_f)
    #         if len(handle_pos) > 1:
    #             vcc.append(COLORS_ALPHA[i % 64])
    #         else:
    #             vcc.append(np.array([1., 1., 1., 1.]))
    # if save_path is None:
    #     return vv, ff, vcc
    # else:
    #     vv, ff, vcc = merge_mesh(vv, ff, vcc)
    MeshO3d(v=v, f=f,vc=vcc).write_ply(save_path)
    return vv, ff, vcc

def merge_mesh(vs, fs, vcs):
    v_num = 0
    new_fs = [fs[0]]
    new_vcs = []
    for i in range(len(vs)):
        if i >= 1:
            v_num += vs[i-1].shape[0]
            new_fs.append(fs[i]+v_num)
        if vcs is not None:
            if vcs[i].ndim == 1:
                new_vcs.append(np.tile(np.expand_dims(vcs[i], 0), [vs[i].shape[0], 1]))
            else:
                new_vcs.append(vcs[i])
    vs = np.concatenate(vs, 0)
    new_fs = np.concatenate(new_fs, 0)
    if vcs is not None:
        new_vcs = np.concatenate(new_vcs, 0)
    return vs, new_fs, new_vcs


class MeshO3d(object):
    def __init__(self, v=None, f=None, vc=None,filename=None):
        self.m = o3d.geometry.TriangleMesh()
        if v is not None:
            self.m.vertices = o3d.utility.Vector3dVector(v.astype(np.float32))
            if f is not None:
                self.m.triangles = o3d.utility.Vector3iVector(f.astype(np.int32))
            if vc is not None:
                self.m.vertex_colors = o3d.utility.Vector3dVector(vc[:, :3].astype(np.float32))
        elif filename is not None:
            v, f = read_obj(filename)
            self.m = o3d.geometry.TriangleMesh()
            self.m.vertices = o3d.utility.Vector3dVector(v.astype(np.float32))
            if f is not None:
                self.m.triangles = o3d.utility.Vector3iVector(f.astype(np.int32))

    @property
    def v(self):
        return np.asarray(self.m.vertices)

    @v.setter
    def v(self, value):
        self.m.vertices = o3d.utility.Vector3dVector(value.astype(np.float32))
    @property
    def vc(self):
        if self.m.has_vertex_colors():
            return np.asarray(self.m.vertex_colors)
        else:
            return np.zeros((len(self.m.vertices), 3))
        
    @vc.setter
    def vc(self, value):
        self.m.vertex_colors = o3d.utility.Vector3dVector(value.astype(np.float32))
        
    @property
    def f(self):
        return np.asarray(self.m.triangles)

    @f.setter
    def f(self, value):import trimesh
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from scripts.utils.utils import save_tensor_image, mask_to_mesh, mask_to_mesh_distancemap
import cv2
import torch
from probreg import cpd
import meshlib.mrmeshpy as mm
import meshlib.mrmeshnumpy as mn
from pytorch3d.io import load_ply, load_obj, save_ply, IO, save_obj, load_objs_as_meshes
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss import chamfer_distance
from pytorch3d.transforms import RotateAxisAngle, Translate, Scale
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.renderer import Textures
import torchvision.transforms as transforms
import pymeshlab
import open3d as o3d

from geomdl import construct
from geomdl import fitting
from geomdl import exchange


class MeshProcessor():
    def __init__(self, root_dir):
        self.all_base_img = [os.path.join(root_dir, 'rgb', f) for f in os.listdir(os.path.join(root_dir, 'rgb')) ]
        self.all_base_mask = [os.path.join(root_dir, 'mask', f) for f in os.listdir(os.path.join(root_dir, 'mask')) ]
        # self.all_base_shape = [os.path.join(root_dir, 'base_shape', f) for f in os.listdir(os.path.join(root_dir, 'base_shape')) if f.endswith('.obj')]        
        # self.all_deform_shape_denoise = [os.path.join(root_dir, 'deform_raw_denoise', f) for f in os.listdir(os.path.join(root_dir, 'deform_raw_denoise')) if f.endswith('.ply')]
        # self.all_deform_train = [os.path.join(root_dir, 'deform_train', f) for f in os.listdir(os.path.join(root_dir, 'deform_train'))]
        # # self.all_deform_train = [os.path.join(root_dir, 'deform_train_new', f) for f in os.listdir(os.path.join(root_dir, 'deform_train_new'))]
        # self.all_base_img.sort()
        # self.all_base_mask.sort()
        # self.all_base_shape.sort()
        # self.all_deform_train.sort()
        # self.all_deform_shape_denoise.sort()

    def raw_to_canonical(self,path,rotate_x=False,export=True):
        """
        raw mesh to canonical space
        """
        if type(path) == str:
            mesh = trimesh.load_mesh(path)
        else:
            mesh = path
        t = -mesh.centroid
        mesh.apply_translation(t)
        max_extent = mesh.extents.max()
        scale_factor = 1 / max_extent
        mesh.apply_scale(scale_factor)
        if rotate_x:
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]))
            # mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0]))
        if export:
            mesh.export(path)
        return torch.tensor(mesh.vertices)
    
    def uv_mapping(self,meshes:Meshes, texture_image:torch.tensor, save_mesh=False, return_mesh=False):
        texture_lsit = []
        for i in range(len(meshes)):
            mesh = meshes[i]
            texture_img = texture_image[i]
            verts= mesh.verts_packed()
            faces = mesh.faces_packed().unsqueeze(0)
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

        if return_mesh:
            return meshes
        else:
            return uvs[:,:,:2].squeeze(), faces.squeeze(), texture_img.permute(1,2,0)
        return meshes
    def img_to_tensor(self, img:np.array):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float() / 255
        return img_tensor
    
    def nonrigid_cpd_cuda(self, basepoints:torch.tensor, deformed_points:torch.tensor,use_cuda=True):
        import cupy as cp
        # set cuda device
        if use_cuda:
            to_cpu = cp.asnumpy
            cp.cuda.Device(3).use()

            cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
        else: 
            cp = np
            to_cpu = lambda x: x
        source_pt = cp.asarray(basepoints)
        target_pt = cp.asarray(deformed_points)     
        if target_pt.shape[0] > 10000:
            target_index = cp.random.choice(target_pt.shape[0], 10000, replace=False)
            target_sample = target_pt[target_index] 
        else:
            target_sample = target_pt  
        # then non-rigd
        acpd = cpd.NonRigidCPD(source_pt, use_cuda=use_cuda)
        tf_param_nrgd, _ ,_ = acpd.registration(target_sample)
        result_nrgd = tf_param_nrgd.transform(source_pt)
        # registrated_mesh = Meshes(verts=[result], faces=[base.faces_packed()])
        return torch.tensor(result_nrgd)

    def pointcloud_reconstruction(self,obj_file, method='ball_pivoting'):
        """
        conventional methods for point cloud reconstruction
        input: denoised & aligned deformed point cloud 
        output: reconstructed mesh
        save_path: results/cvpr/fitting/{baseline name} 
        """
        pcd = o3d.io.read_point_cloud(obj_file)
        pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  
        pcd.estimate_normals()
        file_name = os.path.basename(obj_file).split('.')[0]
        if method == 'ball_pivoting':
            radii =  o3d.utility.DoubleVector([0.005, 0.01, 0.02, 0.04])        
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,radii)
        elif method == 'poisson':
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        elif method == 'bspline':
            pass
        else:
            raise ValueError('method not supported')
        # export mesh
        save_folder = f'results/cvpr/fitting/{method}'
        save_name = os.path.join(save_folder, f'{file_name}.obj')
        o3d.io.write_triangle_mesh(save_name, mesh)
        print(f'{save_name} is saved')



    def rigid_cpd_cuda(self, basepoints:torch.tensor, deformed_points:torch.tensor,use_cuda=False):
        # import cupy as cp
        if use_cuda:
            to_cpu = cp.asnumpy
            cp.cuda.Device(3).use()
            cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
        else: 
            cp = np
            to_cpu = lambda x: x
        source_pt = cp.asarray(basepoints)
        target_pt = cp.asarray(deformed_points)
        
        # subsample points
        # if source_pt.shape[0] > 100000:
        #     source_index = cp.random.choice(source_pt.shape[0], 10000, replace=False)
        #     source_sample = source_pt[source_index]
        # else:
        #     source_sample = source_pt
        # if target_pt.shape[0] > 100000:
        #     target_sample = cp.random.choice(target_pt.shape[0], 10000, replace=False)
        #     target_sample = target_pt[target_sample]
        # else:
        #     target_sample = target_pt
        # first perform rigid
        rcpd = cpd.RigidCPD(target_pt, use_cuda=use_cuda)
        tf_param_rgd, _, _ = rcpd.registration(source_pt)
        target_rgd = tf_param_rgd.transform(target_pt)
        source_rgd = tf_param_rgd.transform(source_pt)
        # output and test 
        base_ori = trimesh.Trimesh(vertices=to_cpu(source_pt))
        # defrom_ori = trimesh.Trimesh(vertices=to_cpu(target_pt))
        # base_trans = trimesh.Trimesh(vertices=to_cpu(source_rgd))
        return base_ori, to_cpu(target_rgd)
    
    def parametric_surface_fitting(self,pointfile):
        pointcloud = load_ply(pointfile)
        points = pointcloud[0]
        points = points.numpy()
        size_u = int(np.sqrt(points.shape[0]))
        size_v = int(size_u)
        degree_u = 3
        degree_v = 3

        # Do global surface approximation
        # surf = fitting.approximate_surface(points, size_u, size_v, degree_u, degree_v)
        surf = fitting.approximate_surface(points, size_u, size_v, degree_u, degree_v, ctrlpts_size_u=size_u-5, ctrlpts_size_v=size_v-5)
        # export surface
        save_folder = 'results/cvpr/fitting/p-surface'
        os.makedirs(save_folder,exist_ok=True)
        save_name = os.path.join(save_folder, os.path.basename(pointfile).split('.')[0]+'.obj')
        surf.delta = 0.01
        exchange.export_obj(surf, save_name)
        print(f'{save_name} is saved')
    
    def triangulate_points(self,point_file):
        pointcloud = load_ply(point_file)
        points = pointcloud[0]
        # triangulate points
        pc = mn.pointCloudFromPoints(points.numpy())
        pc.validPoints = mm.pointUniformSampling(pc, 1e-3)
        pc.invalidateCaches()
        mesh = mm.triangulatePointCloud(pc)
        out_faces = mn.getNumpyFaces(mesh.topology)
        verts = mn.getNumpyVerts(mesh)
        mesh_save = trimesh.Trimesh(verts, out_faces, process=False)
        save_name = os.path.join('results/cvpr/fitting/p-surface',os.path.basename(point_file).split('.')[0]+'.obj')
        mesh_save.export(save_name)
        
    def clean_point_cloud(self,file,save_folder):
        save_folder = os.path.join(save_folder,os.path.basename(file))
        # if not os.path.exists(save_folder):
        pcd = o3d.io.read_point_cloud(file)
        pcd_down = pcd.voxel_down_sample(voxel_size=0.003)
        cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=4000,std_ratio=1.0)
        clean_pcd = pcd_down.select_by_index(ind)
        o3d.io.write_point_cloud(save_folder, clean_pcd)
        print(f'{save_folder} is cleaned')
    
    def crop_img_mask(self, mask:np.array, img:np.array,vein:np.array,vein_flag =False,mask_size=512):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # mask vein to 3 channel 

        x, y, w, h = cv2.boundingRect(mask)
        if w > h:
            pad = (w - h) // 2
            crop = mask[max(0, y - pad):y + h + pad, x:x + w]
            crop_img = img[max(0, y - pad):y + h + pad, x:x + w]
            if vein_flag:
                crop_vein = vein[max(0, y - pad):y + h + pad, x:x + w]
        else:
            pad = (h - w) // 2
            crop = mask[y:y + h, max(0, x - pad):x + w + pad]
            crop_img = img[y:y + h, max(0, x - pad):x + w + pad]
            if vein_flag:
                crop_vein = vein[y:y + h, max(0, x - pad):x + w + pad]

        crop_resized = cv2.resize(crop, (mask_size, mask_size))
        mask_tensor = self.img_to_tensor(crop_resized)
        # crop img accoding to mask
        img_resized = cv2.resize(crop_img, (mask_size, mask_size))
        img_tensor = self.img_to_tensor(img_resized)
        # set mask==0 area to 0 in img
        img_tensor = img_tensor * mask_tensor
        # for vein 
        if vein_flag:
            vein_resized = cv2.resize(crop_vein, (mask_size, mask_size))
            vein_tensor = self.img_to_tensor(vein_resized)
            vein_tensor = vein_tensor * mask_tensor
            
            return mask_tensor, img_tensor, vein_tensor
        return mask_tensor, img_tensor, None

    def clean_mesh(self,mesh_file,save_folder):
        """
        meshlab api for mesh cleaning
        """
        # read by pymeshlab
        save_name = os.path.join(save_folder,os.path.basename(mesh_file))
        if not os.path.exists(save_name):  
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(mesh_file)
            ms.meshing_remove_connected_component_by_diameter()
            ms.save_current_mesh(save_name)
            print(f"clean {save_name}")
        else:
            print(f"{save_name} already exists")
            
    
    
if __name__ == "__main__":
    root = 'dataset/2D_Datasets/leaf_vein'
    # set visible environment device to cuda：3 
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    device = torch.device("cuda:3")


    # task
    regis = True
    non_rigid = False
    clean = False
    base_process = False
    simplify = False
    gen_bone = False
    densify = False
    poisson = False
    parametric_fitting = False
    
    if parametric_fitting:
        point_dir = 'results/cvpr/neuraleaf'
        point_file = [os.path.join(point_dir,f) for f in os.listdir(point_dir) if f.endswith('.ply')]
        for file in point_file:
            processor = MeshProcessor(root)
            processor.parametric_surface_fitting(file)
        
    
    if clean:   # clean and normalize raw deform shape
        processor = MeshProcessor(root)

        mesh_path = "dataset/cvpr_final/deform_raw"
        point_file = [f for f in os.listdir(mesh_path) if f.endswith('.ply')]
        point_file.sort()
        for file in point_file:
            test_mesh = os.path.join(mesh_path,file)
            processor.clean_point_cloud(test_mesh, 'dataset/cvpr_final/deform_raw_denoise')
        mesh_path_cleaned = "dataset/cvpr_final/deform_raw_denoise"
        mesh_file_cleaned = [f for f in os.listdir(mesh_path_cleaned) if f.endswith('.ply')]
        mesh_file_cleaned.sort()
        for file in mesh_file_cleaned:
            clean_file = os.path.join(mesh_path_cleaned,file)
            normalized_mesh = processor.raw_to_canonical(clean_file)
    
    if base_process:       # process base shape 
        processor = MeshProcessor(root)
        resize_transform = transforms.Resize((256, 256))
        # setting
        crop = False
        test_normalize = False
        # data process
        base_img = processor.all_base_img
        base_mask = processor.all_base_mask
        for i in range(len(base_mask)):     
            if i % 30 ==0:       
                mask_file = base_mask[i]
                img_file = base_img[i]
                img_name= os.path.basename(img_file).split('.')[0]
                # mask_file = os.path.join(root, 'base_mask', f'{img_name}.png')
                mask = cv2.imread(mask_file)
                texture = cv2.imread(img_file)
                
                if not mask.shape[0]==mask.shape[1]:
                    mask, img,_  = processor.crop_img_mask(mask, img,vein=None, vein_flag=False)
                    # save mask and img 
                    save_tensor_image(mask,mask_file )
                    save_tensor_image(img,img_file)       
                base_mesh_path = os.path.join(root, 'base_shape', f'{img_name}.obj')  
                # base_mesh_path = 'dataset/template.obj'
                if not os.path.exists(base_mesh_path):
                    mask = cv2.imread(mask_file)
                    texture  = cv2.imread(img_file)
                    texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
                    texture_tensor = processor.img_to_tensor(texture)
                    mask_tensor = processor.img_to_tensor(mask)
                    base_mesh = mask_to_mesh(resize_transform(mask_tensor), device)
                    # base_mesh = mask_to_mesh_distancemap(mask_file)
                    verts_uv, faces_uv , map= processor.uv_mapping(base_mesh, texture_tensor, save_mesh=False)

                    save_obj(base_mesh_path, base_mesh.verts_packed(), base_mesh.faces_packed(),verts_uvs=verts_uv,faces_uvs=faces_uv,texture_map = map)
                    print(f'{base_mesh_path} saved')
                    verts_canonical = processor.raw_to_canonical(base_mesh_path,rotate_x=False)
                    save_obj(base_mesh_path, base_mesh.verts_packed(), base_mesh.faces_packed(),verts_uvs=verts_uv,faces_uvs=faces_uv,texture_map = map)

                else:
                    print(f'{base_mesh_path} already exists')

    # poisson reconstruction
    if poisson:
        processor = MeshProcessor(root)
        deform_mesh = processor.all_deform_train
        for file in deform_mesh:
            processor.pointcloud_reconstruction(file, method='ball_pivoting')

    # rigid registration
    if regis:
        processor = MeshProcessor(root)
        # base_mesh = processor.all_base_shape
        deform_mesh_path = 'dataset/denseleaf/plant_real_data/test_thesis/data_00001/split/'
        deform_mesh = [os.path.join(deform_mesh_path,f) for f in os.listdir(deform_mesh_path) if f.endswith('.ply')]        
        for i in range(len(deform_mesh)):
            # if 'regis' in deform_mesh[i]:
            # base = load_obj(base_mesh[i])
                processor.raw_to_canonical(deform_mesh[i],rotate_x=False)
                # deform = load_objs_as_meshes([deform_mesh[i]])
                deform = load_ply(deform_mesh[i])
                # base_name = os.path.basename(deform_mesh[i]).split('.')[0]
                deform_name = os.path.basename(deform_mesh[i]).split('.')[0]
        
                # bath_path = 'dataset/cvpr_final/base_shape/'
                deform_train_path = 'dataset/denseleaf/plant_real_data/test_thesis/data_00001/split_regis'

                # base_file = os.path.join(bath_path, deform_name+'.obj')
                # base= load_objs_as_meshes([base_file])
                base = load_obj('dataset/cvpr_final/base_shape/1-6.obj')
                # base = load_obj('dataset/cvpr_final/base_shape/88.obj')

                deform_save_path = os.path.join(deform_train_path,deform_name+'.obj')
                if not os.path.exists(deform_save_path):
                    # base_points = base.verts_packed()
                    base_points = base[0]
                    base_points = base_points - base_points.mean(dim=0)
                    # deform_points = deform.verts_packed()
                    deform_points = deform[0]
                    # sample 20000 points
                    # deform_points = deform_points[torch.randperm(base_points.shape[0])[:10000]]
                    base_ori, deform_trans = processor.rigid_cpd_cuda(base_points,deform_points, use_cuda=False)
                    deform_verts=torch.tensor(deform_trans).unsqueeze(0)
                    # raw to canonical scaling and mean 
                    deform_verts  = deform_verts - deform_verts.mean(dim=1)
                    deform_verts = deform_verts / deform_verts.abs().max()

                    # tri_pts = trimesh.PointCloud(vertices=deform_trans)
                    # tri_pts.export(deform_save_path)
                    deform_regis = Meshes(verts=deform_verts, faces=deform[1][0].unsqueeze(0))
                    # save_ply(deform_save_path, deform_regis)
                    # deform_save_path = os.path.join(regis_path,deform_name+'_regis.ply')
                    IO().save_mesh(deform_regis,deform_save_path)
                    print(f'{deform_save_path} saved')
                else:
                    print(f'{deform_save_path} already exists')
    
    # non-rigid registration
    if non_rigid:
        processor = MeshProcessor(root)
        deform_mesh = processor.all_deform_train
        for i in range(len(deform_mesh)):
            if deform_mesh[i].endswith('.ply'): 
                deform = load_ply(deform_mesh[i])
            elif deform_mesh[i].endswith('.obj'):
                deform = load_obj(deform_mesh[i])
            
            deform_name = os.path.basename(deform_mesh[i]).split('.')[0]
            if '_deform' in deform_name:
                base_name = deform_name.replace('_deform','')
            else:
                base_name = deform_name
            base_path = os.path.join(root, 'base_shape', f'{base_name}.obj')
            deform_path = deform_mesh[i]
            regis_path = 'results/cvpr/fitting/cpd'
            os.makedirs(regis_path,exist_ok=True)
            deform_save_path = os.path.join(regis_path,f'{deform_name}.obj')
            if not os.path.exists(deform_save_path):
                base = load_objs_as_meshes([base_path])
                print(f'processing {deform_mesh[i]} and {base_path}')
                base_points = base.verts_packed()
                deform_points = deform[0]
                base_trans = processor.nonrigid_cpd_cuda(base_points,deform_points, use_cuda=True)
                base_regis = Meshes(verts=base_trans.unsqueeze(0), faces=base.faces_packed().unsqueeze(0),textures=base.textures)
                # save registration with texture 
                IO().save_mesh(base_regis,deform_save_path)
                print(f'{deform_save_path} saved')
            else:
                print(f'{deform_save_path} already exists')      
    
    # mesh simplification
    if simplify:
        processor = MeshProcessor(root)
        all_base_mesh = processor.all_base_shape
        for i in range(len(all_base_mesh)):
            # open3d simplification
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(all_base_mesh[i])
            ms.meshing_decimation_clustering()
            ms.save_current_mesh(all_base_mesh[i])
    
    if densify:
        processor = MeshProcessor(root)
        all_base_mesh = processor.all_base_shape
        for i in range(len(all_base_mesh)):
            # open3d simplification
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(all_base_mesh[i])
            ms.meshing_tri_to_quad_by_4_8_subdivision()
            ms.save_current_mesh(all_base_mesh[i])
          import trimesh
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from scripts.utils.utils import save_tensor_image, mask_to_mesh, mask_to_mesh_distancemap
import cv2
import torch
from probreg import cpd
import meshlib.mrmeshpy as mm
import meshlib.mrmeshnumpy as mn
from pytorch3d.io import load_ply, load_obj, save_ply, IO, save_obj, load_objs_as_meshes
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss import chamfer_distance
from pytorch3d.transforms import RotateAxisAngle, Translate, Scale
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.renderer import Textures
import torchvision.transforms as transforms
import pymeshlab
import open3d as o3d

from geomdl import construct
from geomdl import fitting
from geomdl import exchange


class MeshProcessor():
    def __init__(self, root_dir):
        self.all_base_img = [os.path.join(root_dir, 'rgb', f) for f in os.listdir(os.path.join(root_dir, 'rgb')) ]
        self.all_base_mask = [os.path.join(root_dir, 'mask', f) for f in os.listdir(os.path.join(root_dir, 'mask')) ]
        # self.all_base_shape = [os.path.join(root_dir, 'base_shape', f) for f in os.listdir(os.path.join(root_dir, 'base_shape')) if f.endswith('.obj')]        
        # self.all_deform_shape_denoise = [os.path.join(root_dir, 'deform_raw_denoise', f) for f in os.listdir(os.path.join(root_dir, 'deform_raw_denoise')) if f.endswith('.ply')]
        # self.all_deform_train = [os.path.join(root_dir, 'deform_train', f) for f in os.listdir(os.path.join(root_dir, 'deform_train'))]
        # # self.all_deform_train = [os.path.join(root_dir, 'deform_train_new', f) for f in os.listdir(os.path.join(root_dir, 'deform_train_new'))]
        # self.all_base_img.sort()
        # self.all_base_mask.sort()
        # self.all_base_shape.sort()
        # self.all_deform_train.sort()
        # self.all_deform_shape_denoise.sort()

    def raw_to_canonical(self,path,rotate_x=False,export=True):
        """
        raw mesh to canonical space
        """
        if type(path) == str:
            mesh = trimesh.load_mesh(path)
        else:
            mesh = path
        t = -mesh.centroid
        mesh.apply_translation(t)
        max_extent = mesh.extents.max()
        scale_factor = 1 / max_extent
        mesh.apply_scale(scale_factor)
        if rotate_x:
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]))
            # mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0]))
        if export:
            mesh.export(path)
        return torch.tensor(mesh.vertices)
    
    def uv_mapping(self,meshes:Meshes, texture_image:torch.tensor, save_mesh=False, return_mesh=False):
        texture_lsit = []
        for i in range(len(meshes)):
            mesh = meshes[i]
            texture_img = texture_image[i]
            verts= mesh.verts_packed()
            faces = mesh.faces_packed().unsqueeze(0)
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

        if return_mesh:
            return meshes
        else:
            return uvs[:,:,:2].squeeze(), faces.squeeze(), texture_img.permute(1,2,0)
        return meshes
    def img_to_tensor(self, img:np.array):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float() / 255
        return img_tensor
    
    def nonrigid_cpd_cuda(self, basepoints:torch.tensor, deformed_points:torch.tensor,use_cuda=True):
        import cupy as cp
        # set cuda device
        if use_cuda:
            to_cpu = cp.asnumpy
            cp.cuda.Device(3).use()

            cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
        else: 
            cp = np
            to_cpu = lambda x: x
        source_pt = cp.asarray(basepoints)
        target_pt = cp.asarray(deformed_points)     
        if target_pt.shape[0] > 10000:
            target_index = cp.random.choice(target_pt.shape[0], 10000, replace=False)
            target_sample = target_pt[target_index] 
        else:
            target_sample = target_pt  
        # then non-rigd
        acpd = cpd.NonRigidCPD(source_pt, use_cuda=use_cuda)
        tf_param_nrgd, _ ,_ = acpd.registration(target_sample)
        result_nrgd = tf_param_nrgd.transform(source_pt)
        # registrated_mesh = Meshes(verts=[result], faces=[base.faces_packed()])
        return torch.tensor(result_nrgd)

    def pointcloud_reconstruction(self,obj_file, method='ball_pivoting'):
        """
        conventional methods for point cloud reconstruction
        input: denoised & aligned deformed point cloud 
        output: reconstructed mesh
        save_path: results/cvpr/fitting/{baseline name} 
        """
        pcd = o3d.io.read_point_cloud(obj_file)
        pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  
        pcd.estimate_normals()
        file_name = os.path.basename(obj_file).split('.')[0]
        if method == 'ball_pivoting':
            radii =  o3d.utility.DoubleVector([0.005, 0.01, 0.02, 0.04])        
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,radii)
        elif method == 'poisson':
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        elif method == 'bspline':
            pass
        else:
            raise ValueError('method not supported')
        # export mesh
        save_folder = f'results/cvpr/fitting/{method}'
        save_name = os.path.join(save_folder, f'{file_name}.obj')
        o3d.io.write_triangle_mesh(save_name, mesh)
        print(f'{save_name} is saved')



    def rigid_cpd_cuda(self, basepoints:torch.tensor, deformed_points:torch.tensor,use_cuda=False):
        # import cupy as cp
        if use_cuda:
            to_cpu = cp.asnumpy
            cp.cuda.Device(3).use()
            cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
        else: 
            cp = np
            to_cpu = lambda x: x
        source_pt = cp.asarray(basepoints)
        target_pt = cp.asarray(deformed_points)
        
        # subsample points
        # if source_pt.shape[0] > 100000:
        #     source_index = cp.random.choice(source_pt.shape[0], 10000, replace=False)
        #     source_sample = source_pt[source_index]
        # else:
        #     source_sample = source_pt
        # if target_pt.shape[0] > 100000:
        #     target_sample = cp.random.choice(target_pt.shape[0], 10000, replace=False)
        #     target_sample = target_pt[target_sample]
        # else:
        #     target_sample = target_pt
        # first perform rigid
        rcpd = cpd.RigidCPD(target_pt, use_cuda=use_cuda)
        tf_param_rgd, _, _ = rcpd.registration(source_pt)
        target_rgd = tf_param_rgd.transform(target_pt)
        source_rgd = tf_param_rgd.transform(source_pt)
        # output and test 
        base_ori = trimesh.Trimesh(vertices=to_cpu(source_pt))
        # defrom_ori = trimesh.Trimesh(vertices=to_cpu(target_pt))
        # base_trans = trimesh.Trimesh(vertices=to_cpu(source_rgd))
        return base_ori, to_cpu(target_rgd)
    
    def parametric_surface_fitting(self,pointfile):
        pointcloud = load_ply(pointfile)
        points = pointcloud[0]
        points = points.numpy()
        size_u = int(np.sqrt(points.shape[0]))
        size_v = int(size_u)
        degree_u = 3
        degree_v = 3

        # Do global surface approximation
        # surf = fitting.approximate_surface(points, size_u, size_v, degree_u, degree_v)
        surf = fitting.approximate_surface(points, size_u, size_v, degree_u, degree_v, ctrlpts_size_u=size_u-5, ctrlpts_size_v=size_v-5)
        # export surface
        save_folder = 'results/cvpr/fitting/p-surface'
        os.makedirs(save_folder,exist_ok=True)
        save_name = os.path.join(save_folder, os.path.basename(pointfile).split('.')[0]+'.obj')
        surf.delta = 0.01
        exchange.export_obj(surf, save_name)
        print(f'{save_name} is saved')
    
    def triangulate_points(self,point_file):
        pointcloud = load_ply(point_file)
        points = pointcloud[0]
        # triangulate points
        pc = mn.pointCloudFromPoints(points.numpy())
        pc.validPoints = mm.pointUniformSampling(pc, 1e-3)
        pc.invalidateCaches()
        mesh = mm.triangulatePointCloud(pc)
        out_faces = mn.getNumpyFaces(mesh.topology)
        verts = mn.getNumpyVerts(mesh)
        mesh_save = trimesh.Trimesh(verts, out_faces, process=False)
        save_name = os.path.join('results/cvpr/fitting/p-surface',os.path.basename(point_file).split('.')[0]+'.obj')
        mesh_save.export(save_name)
        
    def clean_point_cloud(self,file,save_folder):
        save_folder = os.path.join(save_folder,os.path.basename(file))
        # if not os.path.exists(save_folder):
        pcd = o3d.io.read_point_cloud(file)
        pcd_down = pcd.voxel_down_sample(voxel_size=0.003)
        cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=4000,std_ratio=1.0)
        clean_pcd = pcd_down.select_by_index(ind)
        o3d.io.write_point_cloud(save_folder, clean_pcd)
        print(f'{save_folder} is cleaned')
    
    def crop_img_mask(self, mask:np.array, img:np.array,vein:np.array,vein_flag =False,mask_size=512):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # mask vein to 3 channel 

        x, y, w, h = cv2.boundingRect(mask)
        if w > h:
            pad = (w - h) // 2
            crop = mask[max(0, y - pad):y + h + pad, x:x + w]
            crop_img = img[max(0, y - pad):y + h + pad, x:x + w]
            if vein_flag:
                crop_vein = vein[max(0, y - pad):y + h + pad, x:x + w]
        else:
            pad = (h - w) // 2
            crop = mask[y:y + h, max(0, x - pad):x + w + pad]
            crop_img = img[y:y + h, max(0, x - pad):x + w + pad]
            if vein_flag:
                crop_vein = vein[y:y + h, max(0, x - pad):x + w + pad]

        crop_resized = cv2.resize(crop, (mask_size, mask_size))
        mask_tensor = self.img_to_tensor(crop_resized)
        # crop img accoding to mask
        img_resized = cv2.resize(crop_img, (mask_size, mask_size))
        img_tensor = self.img_to_tensor(img_resized)
        # set mask==0 area to 0 in img
        img_tensor = img_tensor * mask_tensor
        # for vein 
        if vein_flag:
            vein_resized = cv2.resize(crop_vein, (mask_size, mask_size))
            vein_tensor = self.img_to_tensor(vein_resized)
            vein_tensor = vein_tensor * mask_tensor
            
            return mask_tensor, img_tensor, vein_tensor
        return mask_tensor, img_tensor, None

    def clean_mesh(self,mesh_file,save_folder):
        """
        meshlab api for mesh cleaning
        """
        # read by pymeshlab
        save_name = os.path.join(save_folder,os.path.basename(mesh_file))
        if not os.path.exists(save_name):  
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(mesh_file)
            ms.meshing_remove_connected_component_by_diameter()
            ms.save_current_mesh(save_name)
            print(f"clean {save_name}")
        else:
            print(f"{save_name} already exists")
            
    
    
if __name__ == "__main__":
    root = 'dataset/2D_Datasets/leaf_vein'
    # set visible environment device to cuda：3 
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    device = torch.device("cuda:3")


    # task
    regis = True
    non_rigid = False
    clean = False
    base_process = False
    simplify = False
    gen_bone = False
    densify = False
    poisson = False
    parametric_fitting = False
    
    if parametric_fitting:
        point_dir = 'results/cvpr/neuraleaf'
        point_file = [os.path.join(point_dir,f) for f in os.listdir(point_dir) if f.endswith('.ply')]
        for file in point_file:
            processor = MeshProcessor(root)
            processor.parametric_surface_fitting(file)
        
    
    if clean:   # clean and normalize raw deform shape
        processor = MeshProcessor(root)

        mesh_path = "dataset/cvpr_final/deform_raw"
        point_file = [f for f in os.listdir(mesh_path) if f.endswith('.ply')]
        point_file.sort()
        for file in point_file:
            test_mesh = os.path.join(mesh_path,file)
            processor.clean_point_cloud(test_mesh, 'dataset/cvpr_final/deform_raw_denoise')
        mesh_path_cleaned = "dataset/cvpr_final/deform_raw_denoise"
        mesh_file_cleaned = [f for f in os.listdir(mesh_path_cleaned) if f.endswith('.ply')]
        mesh_file_cleaned.sort()
        for file in mesh_file_cleaned:
            clean_file = os.path.join(mesh_path_cleaned,file)
            normalized_mesh = processor.raw_to_canonical(clean_file)
    
    if base_process:       # process base shape 
        processor = MeshProcessor(root)
        resize_transform = transforms.Resize((256, 256))
        # setting
        crop = False
        test_normalize = False
        # data process
        base_img = processor.all_base_img
        base_mask = processor.all_base_mask
        for i in range(len(base_mask)):     
            if i % 30 ==0:       
                mask_file = base_mask[i]
                img_file = base_img[i]
                img_name= os.path.basename(img_file).split('.')[0]
                # mask_file = os.path.join(root, 'base_mask', f'{img_name}.png')
                mask = cv2.imread(mask_file)
                texture = cv2.imread(img_file)
                
                if not mask.shape[0]==mask.shape[1]:
                    mask, img,_  = processor.crop_img_mask(mask, img,vein=None, vein_flag=False)
                    # save mask and img 
                    save_tensor_image(mask,mask_file )
                    save_tensor_image(img,img_file)       
                base_mesh_path = os.path.join(root, 'base_shape', f'{img_name}.obj')  
                # base_mesh_path = 'dataset/template.obj'
                if not os.path.exists(base_mesh_path):
                    mask = cv2.imread(mask_file)
                    texture  = cv2.imread(img_file)
                    texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
                    texture_tensor = processor.img_to_tensor(texture)
                    mask_tensor = processor.img_to_tensor(mask)
                    base_mesh = mask_to_mesh(resize_transform(mask_tensor), device)
                    # base_mesh = mask_to_mesh_distancemap(mask_file)
                    verts_uv, faces_uv , map= processor.uv_mapping(base_mesh, texture_tensor, save_mesh=False)

                    save_obj(base_mesh_path, base_mesh.verts_packed(), base_mesh.faces_packed(),verts_uvs=verts_uv,faces_uvs=faces_uv,texture_map = map)
                    print(f'{base_mesh_path} saved')
                    verts_canonical = processor.raw_to_canonical(base_mesh_path,rotate_x=False)
                    save_obj(base_mesh_path, base_mesh.verts_packed(), base_mesh.faces_packed(),verts_uvs=verts_uv,faces_uvs=faces_uv,texture_map = map)

                else:
                    print(f'{base_mesh_path} already exists')

    # poisson reconstruction
    if poisson:
        processor = MeshProcessor(root)
        deform_mesh = processor.all_deform_train
        for file in deform_mesh:
            processor.pointcloud_reconstruction(file, method='ball_pivoting')

    # rigid registration
    if regis:
        processor = MeshProcessor(root)
        # base_mesh = processor.all_base_shape
        deform_mesh_path = 'dataset/denseleaf/plant_real_data/test_thesis/data_00001/split/'
        deform_mesh = [os.path.join(deform_mesh_path,f) for f in os.listdir(deform_mesh_path) if f.endswith('.ply')]        
        for i in range(len(deform_mesh)):
            # if 'regis' in deform_mesh[i]:
            # base = load_obj(base_mesh[i])
                processor.raw_to_canonical(deform_mesh[i],rotate_x=False)
                # deform = load_objs_as_meshes([deform_mesh[i]])
                deform = load_ply(deform_mesh[i])
                # base_name = os.path.basename(deform_mesh[i]).split('.')[0]
                deform_name = os.path.basename(deform_mesh[i]).split('.')[0]
        
                # bath_path = 'dataset/cvpr_final/base_shape/'
                deform_train_path = 'dataset/denseleaf/plant_real_data/test_thesis/data_00001/split_regis'

                # base_file = os.path.join(bath_path, deform_name+'.obj')
                # base= load_objs_as_meshes([base_file])
                base = load_obj('dataset/cvpr_final/base_shape/1-6.obj')
                # base = load_obj('dataset/cvpr_final/base_shape/88.obj')

                deform_save_path = os.path.join(deform_train_path,deform_name+'.obj')
                if not os.path.exists(deform_save_path):
                    # base_points = base.verts_packed()
                    base_points = base[0]
                    base_points = base_points - base_points.mean(dim=0)
                    # deform_points = deform.verts_packed()
                    deform_points = deform[0]
                    # sample 20000 points
                    # deform_points = deform_points[torch.randperm(base_points.shape[0])[:10000]]
                    base_ori, deform_trans = processor.rigid_cpd_cuda(base_points,deform_points, use_cuda=False)
                    deform_verts=torch.tensor(deform_trans).unsqueeze(0)
                    # raw to canonical scaling and mean 
                    deform_verts  = deform_verts - deform_verts.mean(dim=1)
                    deform_verts = deform_verts / deform_verts.abs().max()

                    # tri_pts = trimesh.PointCloud(vertices=deform_trans)
                    # tri_pts.export(deform_save_path)
                    deform_regis = Meshes(verts=deform_verts, faces=deform[1][0].unsqueeze(0))
                    # save_ply(deform_save_path, deform_regis)
                    # deform_save_path = os.path.join(regis_path,deform_name+'_regis.ply')
                    IO().save_mesh(deform_regis,deform_save_path)
                    print(f'{deform_save_path} saved')
                else:
                    print(f'{deform_save_path} already exists')
    
    # non-rigid registration
    if non_rigid:
        processor = MeshProcessor(root)
        deform_mesh = processor.all_deform_train
        for i in range(len(deform_mesh)):
            if deform_mesh[i].endswith('.ply'): 
                deform = load_ply(deform_mesh[i])
            elif deform_mesh[i].endswith('.obj'):
                deform = load_obj(deform_mesh[i])
            
            deform_name = os.path.basename(deform_mesh[i]).split('.')[0]
            if '_deform' in deform_name:
                base_name = deform_name.replace('_deform','')
            else:
                base_name = deform_name
            base_path = os.path.join(root, 'base_shape', f'{base_name}.obj')
            deform_path = deform_mesh[i]
            regis_path = 'results/cvpr/fitting/cpd'
            os.makedirs(regis_path,exist_ok=True)
            deform_save_path = os.path.join(regis_path,f'{deform_name}.obj')
            if not os.path.exists(deform_save_path):
                base = load_objs_as_meshes([base_path])
                print(f'processing {deform_mesh[i]} and {base_path}')
                base_points = base.verts_packed()
                deform_points = deform[0]
                base_trans = processor.nonrigid_cpd_cuda(base_points,deform_points, use_cuda=True)
                base_regis = Meshes(verts=base_trans.unsqueeze(0), faces=base.faces_packed().unsqueeze(0),textures=base.textures)
                # save registration with texture 
                IO().save_mesh(base_regis,deform_save_path)
                print(f'{deform_save_path} saved')
            else:
                print(f'{deform_save_path} already exists')      
    
    # mesh simplification
    if simplify:
        processor = MeshProcessor(root)
        all_base_mesh = processor.all_base_shape
        for i in range(len(all_base_mesh)):
            # open3d simplification
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(all_base_mesh[i])
            ms.meshing_decimation_clustering()
            ms.save_current_mesh(all_base_mesh[i])
    
    if densify:
        processor = MeshProcessor(root)
        all_base_mesh = processor.all_base_shape
        for i in range(len(all_base_mesh)):
            # open3d simplificationimport trimesh
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from scripts.utils.utils import save_tensor_image, mask_to_mesh, mask_to_mesh_distancemap
import cv2
import torch
from probreg import cpd
import meshlib.mrmeshpy as mm
import meshlib.mrmeshnumpy as mn
from pytorch3d.io import load_ply, load_obj, save_ply, IO, save_obj, load_objs_as_meshes
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss import chamfer_distance
from pytorch3d.transforms import RotateAxisAngle, Translate, Scale
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.renderer import Textures
import torchvision.transforms as transforms
import pymeshlab
import open3d as o3d

from geomdl import construct
from geomdl import fitting
from geomdl import exchange


class MeshProcessor():
    def __init__(self, root_dir):
        self.all_base_img = [os.path.join(root_dir, 'rgb', f) for f in os.listdir(os.path.join(root_dir, 'rgb')) ]
        self.all_base_mask = [os.path.join(root_dir, 'mask', f) for f in os.listdir(os.path.join(root_dir, 'mask')) ]
        # self.all_base_shape = [os.path.join(root_dir, 'base_shape', f) for f in os.listdir(os.path.join(root_dir, 'base_shape')) if f.endswith('.obj')]        
        # self.all_deform_shape_denoise = [os.path.join(root_dir, 'deform_raw_denoise', f) for f in os.listdir(os.path.join(root_dir, 'deform_raw_denoise')) if f.endswith('.ply')]
        # self.all_deform_train = [os.path.join(root_dir, 'deform_train', f) for f in os.listdir(os.path.join(root_dir, 'deform_train'))]
        # # self.all_deform_train = [os.path.join(root_dir, 'deform_train_new', f) for f in os.listdir(os.path.join(root_dir, 'deform_train_new'))]
        # self.all_base_img.sort()
        # self.all_base_mask.sort()
        # self.all_base_shape.sort()
        # self.all_deform_train.sort()
        # self.all_deform_shape_denoise.sort()

    def raw_to_canonical(self,path,rotate_x=False,export=True):
        """
        raw mesh to canonical space
        """
        if type(path) == str:
            mesh = trimesh.load_mesh(path)
        else:
            mesh = path
        t = -mesh.centroid
        mesh.apply_translation(t)
        max_extent = mesh.extents.max()
        scale_factor = 1 / max_extent
        mesh.apply_scale(scale_factor)
        if rotate_x:
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]))
            # mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0]))
        if export:
            mesh.export(path)
        return torch.tensor(mesh.vertices)
    
    def uv_mapping(self,meshes:Meshes, texture_image:torch.tensor, save_mesh=False, return_mesh=False):
        texture_lsit = []
        for i in range(len(meshes)):
            mesh = meshes[i]
            texture_img = texture_image[i]
            verts= mesh.verts_packed()
            faces = mesh.faces_packed().unsqueeze(0)
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

        if return_mesh:
            return meshes
        else:
            return uvs[:,:,:2].squeeze(), faces.squeeze(), texture_img.permute(1,2,0)
        return meshes
    def img_to_tensor(self, img:np.array):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float() / 255
        return img_tensor
    
    def nonrigid_cpd_cuda(self, basepoints:torch.tensor, deformed_points:torch.tensor,use_cuda=True):
        import cupy as cp
        # set cuda device
        if use_cuda:
            to_cpu = cp.asnumpy
            cp.cuda.Device(3).use()

            cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
        else: 
            cp = np
            to_cpu = lambda x: x
        source_pt = cp.asarray(basepoints)
        target_pt = cp.asarray(deformed_points)     
        if target_pt.shape[0] > 10000:
            target_index = cp.random.choice(target_pt.shape[0], 10000, replace=False)
            target_sample = target_pt[target_index] 
        else:
            target_sample = target_pt  
        # then non-rigd
        acpd = cpd.NonRigidCPD(source_pt, use_cuda=use_cuda)
        tf_param_nrgd, _ ,_ = acpd.registration(target_sample)
        result_nrgd = tf_param_nrgd.transform(source_pt)
        # registrated_mesh = Meshes(verts=[result], faces=[base.faces_packed()])
        return torch.tensor(result_nrgd)

    def pointcloud_reconstruction(self,obj_file, method='ball_pivoting'):
        """
        conventional methods for point cloud reconstruction
        input: denoised & aligned deformed point cloud 
        output: reconstructed mesh
        save_path: results/cvpr/fitting/{baseline name} 
        """
        pcd = o3d.io.read_point_cloud(obj_file)
        pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  
        pcd.estimate_normals()
        file_name = os.path.basename(obj_file).split('.')[0]
        if method == 'ball_pivoting':
            radii =  o3d.utility.DoubleVector([0.005, 0.01, 0.02, 0.04])        
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,radii)
        elif method == 'poisson':
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        elif method == 'bspline':
            pass
        else:
            raise ValueError('method not supported')
        # export mesh
        save_folder = f'results/cvpr/fitting/{method}'
        save_name = os.path.join(save_folder, f'{file_name}.obj')
        o3d.io.write_triangle_mesh(save_name, mesh)
        print(f'{save_name} is saved')



    def rigid_cpd_cuda(self, basepoints:torch.tensor, deformed_points:torch.tensor,use_cuda=False):
        # import cupy as cp
        if use_cuda:
            to_cpu = cp.asnumpy
            cp.cuda.Device(3).use()
            cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
        else: 
            cp = np
            to_cpu = lambda x: x
        source_pt = cp.asarray(basepoints)
        target_pt = cp.asarray(deformed_points)
        
        # subsample points
        # if source_pt.shape[0] > 100000:
        #     source_index = cp.random.choice(source_pt.shape[0], 10000, replace=False)
        #     source_sample = source_pt[source_index]
        # else:
        #     source_sample = source_pt
        # if target_pt.shape[0] > 100000:
        #     target_sample = cp.random.choice(target_pt.shape[0], 10000, replace=False)
        #     target_sample = target_pt[target_sample]
        # else:
        #     target_sample = target_pt
        # first perform rigid
        rcpd = cpd.RigidCPD(target_pt, use_cuda=use_cuda)
        tf_param_rgd, _, _ = rcpd.registration(source_pt)
        target_rgd = tf_param_rgd.transform(target_pt)
        source_rgd = tf_param_rgd.transform(source_pt)
        # output and test 
        base_ori = trimesh.Trimesh(vertices=to_cpu(source_pt))
        # defrom_ori = trimesh.Trimesh(vertices=to_cpu(target_pt))
        # base_trans = trimesh.Trimesh(vertices=to_cpu(source_rgd))
        return base_ori, to_cpu(target_rgd)
    
    def parametric_surface_fitting(self,pointfile):
        pointcloud = load_ply(pointfile)
        points = pointcloud[0]
        points = points.numpy()
        size_u = int(np.sqrt(points.shape[0]))
        size_v = int(size_u)
        degree_u = 3
        degree_v = 3

        # Do global surface approximation
        # surf = fitting.approximate_surface(points, size_u, size_v, degree_u, degree_v)
        surf = fitting.approximate_surface(points, size_u, size_v, degree_u, degree_v, ctrlpts_size_u=size_u-5, ctrlpts_size_v=size_v-5)
        # export surface
        save_folder = 'results/cvpr/fitting/p-surface'
        os.makedirs(save_folder,exist_ok=True)
        save_name = os.path.join(save_folder, os.path.basename(pointfile).split('.')[0]+'.obj')
        surf.delta = 0.01
        exchange.export_obj(surf, save_name)
        print(f'{save_name} is saved')
    
    def triangulate_points(self,point_file):
        pointcloud = load_ply(point_file)
        points = pointcloud[0]
        # triangulate points
        pc = mn.pointCloudFromPoints(points.numpy())
        pc.validPoints = mm.pointUniformSampling(pc, 1e-3)
        pc.invalidateCaches()
        mesh = mm.triangulatePointCloud(pc)
        out_faces = mn.getNumpyFaces(mesh.topology)
        verts = mn.getNumpyVerts(mesh)
        mesh_save = trimesh.Trimesh(verts, out_faces, process=False)
        save_name = os.path.join('results/cvpr/fitting/p-surface',os.path.basename(point_file).split('.')[0]+'.obj')
        mesh_save.export(save_name)
        
    def clean_point_cloud(self,file,save_folder):
        save_folder = os.path.join(save_folder,os.path.basename(file))
        # if not os.path.exists(save_folder):
        pcd = o3d.io.read_point_cloud(file)
        pcd_down = pcd.voxel_down_sample(voxel_size=0.003)
        cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=4000,std_ratio=1.0)
        clean_pcd = pcd_down.select_by_index(ind)
        o3d.io.write_point_cloud(save_folder, clean_pcd)
        print(f'{save_folder} is cleaned')
    
    def crop_img_mask(self, mask:np.array, img:np.array,vein:np.array,vein_flag =False,mask_size=512):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # mask vein to 3 channel 

        x, y, w, h = cv2.boundingRect(mask)
        if w > h:
            pad = (w - h) // 2
            crop = mask[max(0, y - pad):y + h + pad, x:x + w]
            crop_img = img[max(0, y - pad):y + h + pad, x:x + w]
            if vein_flag:
                crop_vein = vein[max(0, y - pad):y + h + pad, x:x + w]
        else:
            pad = (h - w) // 2
            crop = mask[y:y + h, max(0, x - pad):x + w + pad]
            crop_img = img[y:y + h, max(0, x - pad):x + w + pad]
            if vein_flag:
                crop_vein = vein[y:y + h, max(0, x - pad):x + w + pad]

        crop_resized = cv2.resize(crop, (mask_size, mask_size))
        mask_tensor = self.img_to_tensor(crop_resized)
        # crop img accoding to mask
        img_resized = cv2.resize(crop_img, (mask_size, mask_size))
        img_tensor = self.img_to_tensor(img_resized)
        # set mask==0 area to 0 in img
        img_tensor = img_tensor * mask_tensor
        # for vein 
        if vein_flag:
            vein_resized = cv2.resize(crop_vein, (mask_size, mask_size))
            vein_tensor = self.img_to_tensor(vein_resized)
            vein_tensor = vein_tensor * mask_tensor
            
            return mask_tensor, img_tensor, vein_tensor
        return mask_tensor, img_tensor, None

    def clean_mesh(self,mesh_file,save_folder):
        """
        meshlab api for mesh cleaning
        """
        # read by pymeshlab
        save_name = os.path.join(save_folder,os.path.basename(mesh_file))
        if not os.path.exists(save_name):  
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(mesh_file)
            ms.meshing_remove_connected_component_by_diameter()
            ms.save_current_mesh(save_name)
            print(f"clean {save_name}")
        else:
            print(f"{save_name} already exists")
            
    
    
if __name__ == "__main__":
    root = 'dataset/2D_Datasets/leaf_vein'
    # set visible environment device to cuda：3 
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    device = torch.device("cuda:3")


    # task
    regis = True
    non_rigid = False
    clean = False
    base_process = False
    simplify = False
    gen_bone = False
    densify = False
    poisson = False
    parametric_fitting = False
    
    if parametric_fitting:
        point_dir = 'results/cvpr/neuraleaf'
        point_file = [os.path.join(point_dir,f) for f in os.listdir(point_dir) if f.endswith('.ply')]
        for file in point_file:
            processor = MeshProcessor(root)
            processor.parametric_surface_fitting(file)
        
    
    if clean:   # clean and normalize raw deform shape
        processor = MeshProcessor(root)

        mesh_path = "dataset/cvpr_final/deform_raw"
        point_file = [f for f in os.listdir(mesh_path) if f.endswith('.ply')]
        point_file.sort()
        for file in point_file:
            test_mesh = os.path.join(mesh_path,file)
            processor.clean_point_cloud(test_mesh, 'dataset/cvpr_final/deform_raw_denoise')
        mesh_path_cleaned = "dataset/cvpr_final/deform_raw_denoise"
        mesh_file_cleaned = [f for f in os.listdir(mesh_path_cleaned) if f.endswith('.ply')]
        mesh_file_cleaned.sort()
        for file in mesh_file_cleaned:
            clean_file = os.path.join(mesh_path_cleaned,file)
            normalized_mesh = processor.raw_to_canonical(clean_file)
    
    if base_process:       # process base shape 
        processor = MeshProcessor(root)
        resize_transform = transforms.Resize((256, 256))
        # setting
        crop = False
        test_normalize = False
        # data process
        base_img = processor.all_base_img
        base_mask = processor.all_base_mask
        for i in range(len(base_mask)):     
            if i % 30 ==0:       
                mask_file = base_mask[i]
                img_file = base_img[i]
                img_name= os.path.basename(img_file).split('.')[0]
                # mask_file = os.path.join(root, 'base_mask', f'{img_name}.png')
                mask = cv2.imread(mask_file)
                texture = cv2.imread(img_file)
                
                if not mask.shape[0]==mask.shape[1]:
                    mask, img,_  = processor.crop_img_mask(mask, img,vein=None, vein_flag=False)
                    # save mask and img 
                    save_tensor_image(mask,mask_file )
                    save_tensor_image(img,img_file)       
                base_mesh_path = os.path.join(root, 'base_shape', f'{img_name}.obj')  
                # base_mesh_path = 'dataset/template.obj'
                if not os.path.exists(base_mesh_path):
                    mask = cv2.imread(mask_file)
                    texture  = cv2.imread(img_file)
                    texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
                    texture_tensor = processor.img_to_tensor(texture)
                    mask_tensor = processor.img_to_tensor(mask)
                    base_mesh = mask_to_mesh(resize_transform(mask_tensor), device)
                    # base_mesh = mask_to_mesh_distancemap(mask_file)
                    verts_uv, faces_uv , map= processor.uv_mapping(base_mesh, texture_tensor, save_mesh=False)

                    save_obj(base_mesh_path, base_mesh.verts_packed(), base_mesh.faces_packed(),verts_uvs=verts_uv,faces_uvs=faces_uv,texture_map = map)
                    print(f'{base_mesh_path} saved')
                    verts_canonical = processor.raw_to_canonical(base_mesh_path,rotate_x=False)
                    save_obj(base_mesh_path, base_mesh.verts_packed(), base_mesh.faces_packed(),verts_uvs=verts_uv,faces_uvs=faces_uv,texture_map = map)

                else:
                    print(f'{base_mesh_path} already exists')

    # poisson reconstruction
    if poisson:
        processor = MeshProcessor(root)
        deform_mesh = processor.all_deform_train
        for file in deform_mesh:
            processor.pointcloud_reconstruction(file, method='ball_pivoting')

    # rigid registration
    if regis:
        processor = MeshProcessor(root)
        # base_mesh = processor.all_base_shape
        deform_mesh_path = 'dataset/denseleaf/plant_real_data/test_thesis/data_00001/split/'
        deform_mesh = [os.path.join(deform_mesh_path,f) for f in os.listdir(deform_mesh_path) if f.endswith('.ply')]        
        for i in range(len(deform_mesh)):
            # if 'regis' in deform_mesh[i]:
            # base = load_obj(base_mesh[i])
                processor.raw_to_canonical(deform_mesh[i],rotate_x=False)
                # deform = load_objs_as_meshes([deform_mesh[i]])
                deform = load_ply(deform_mesh[i])
                # base_name = os.path.basename(deform_mesh[i]).split('.')[0]
                deform_name = os.path.basename(deform_mesh[i]).split('.')[0]
        
                # bath_path = 'dataset/cvpr_final/base_shape/'
                deform_train_path = 'dataset/denseleaf/plant_real_data/test_thesis/data_00001/split_regis'

                # base_file = os.path.join(bath_path, deform_name+'.obj')
                # base= load_objs_as_meshes([base_file])
                base = load_obj('dataset/cvpr_final/base_shape/1-6.obj')
                # base = load_obj('dataset/cvpr_final/base_shape/88.obj')

                deform_save_path = os.path.join(deform_train_path,deform_name+'.obj')
                if not os.path.exists(deform_save_path):
                    # base_points = base.verts_packed()
                    base_points = base[0]
                    base_points = base_points - base_points.mean(dim=0)
                    # deform_points = deform.verts_packed()
                    deform_points = deform[0]
                    # sample 20000 points
                    # deform_points = deform_points[torch.randperm(base_points.shape[0])[:10000]]
                    base_ori, deform_trans = processor.rigid_cpd_cuda(base_points,deform_points, use_cuda=False)
                    deform_verts=torch.tensor(deform_trans).unsqueeze(0)
                    # raw to canonical scaling and mean 
                    deform_verts  = deform_verts - deform_verts.mean(dim=1)
                    deform_verts = deform_verts / deform_verts.abs().max()

                    # tri_pts = trimesh.PointCloud(vertices=deform_trans)
                    # tri_pts.export(deform_save_path)
                    deform_regis = Meshes(verts=deform_verts, faces=deform[1][0].unsqueeze(0))
                    # save_ply(deform_save_path, deform_regis)
                    # deform_save_path = os.path.join(regis_path,deform_name+'_regis.ply')
                    IO().save_mesh(deform_regis,deform_save_path)
                    print(f'{deform_save_path} saved')
                else:
                    print(f'{deform_save_path} already exists')
    
    # non-rigid registration
    if non_rigid:
        processor = MeshProcessor(root)
        deform_mesh = processor.all_deform_train
        for i in range(len(deform_mesh)):
            if deform_mesh[i].endswith('.ply'): 
                deform = load_ply(deform_mesh[i])
            elif deform_mesh[i].endswith('.obj'):
                deform = load_obj(deform_mesh[i])
            
            deform_name = os.path.basename(deform_mesh[i]).split('.')[0]
            if '_deform' in deform_name:
                base_name = deform_name.replace('_deform','')
            else:
                base_name = deform_name
            base_path = os.path.join(root, 'base_shape', f'{base_name}.obj')
            deform_path = deform_mesh[i]
            regis_path = 'results/cvpr/fitting/cpd'
            os.makedirs(regis_path,exist_ok=True)
            deform_save_path = os.path.join(regis_path,f'{deform_name}.obj')
            if not os.path.exists(deform_save_path):
                base = load_objs_as_meshes([base_path])
                print(f'processing {deform_mesh[i]} and {base_path}')
                base_points = base.verts_packed()
                deform_points = deform[0]
                base_trans = processor.nonrigid_cpd_cuda(base_points,deform_points, use_cuda=True)
                base_regis = Meshes(verts=base_trans.unsqueeze(0), faces=base.faces_packed().unsqueeze(0),textures=base.textures)
                # save registration with texture 
                IO().save_mesh(base_regis,deform_save_path)
                print(f'{deform_save_path} saved')
            else:
                print(f'{deform_save_path} already exists')      
    
    # mesh simplification
    if simplify:
        processor = MeshProcessor(root)
        all_base_mesh = processor.all_base_shape
        for i in range(len(all_base_mesh)):
            # open3d simplification
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(all_base_mesh[i])
            ms.meshing_decimation_clustering()
            ms.save_current_mesh(all_base_mesh[i])
    
    if densify:
        processor = MeshProcessor(root)
        all_base_mesh = processor.all_base_shape
        for i in range(len(all_base_mesh)):
            # open3d simplification
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(all_base_mesh[i])
            ms.meshing_tri_to_quad_by_4_8_subdivision()
            ms.save_current_mesh(all_base_mesh[i])
          
    if gen_bone:
        processor = MeshProcessor(root)
        base_shape = processor.all_base_shape[0]
        processor.generate_bone_from_vein(base_shape, 'dataset/2D_Datasets/leaf_vein_new/vein_train/C_1_1_3_bot.png')
        


[i])
            ms.meshing_tri_to_quad_by_4_8_subdivision()
            ms.save_current_mesh(all_base_mesh[i])
          
    if gen_bone:
        processor = MeshProcessor(root)
        base_shape = processor.all_base_shape[0]
        processor.generate_bone_from_vein(base_shape, 'dataset/2D_Datasets/leaf_vein_new/vein_train/C_1_1_3_bot.png')
        



    if gen_bone:
        processor = MeshProcessor(root)
        base_shape = processor.all_base_shape[0]
        processor.generate_bone_from_vein(base_shape, 'dataset/2D_Datasets/leaf_vein_new/vein_train/C_1_1_3_bot.png')
        



        self.m.triangles = o3d.utility.Vector3iVector(value.astype(np.int32))

    def write_obj(self, fpath):
        if not fpath.endswith('.obj'):
            fpath = fpath + '.obj'
        o3d.io.write_triangle_mesh(fpath, self.m, write_ascii=True)

    def write_ply(self, fpath):
        if not fpath.endswith('.ply'):
            fpath = fpath + '.ply'
        o3d.io.write_triangle_mesh(fpath, self.m, write_ascii=False, compressed=True)


class MeshPyMesh(object):
    def __init__(self, v=None, f=None, vc=None, filename=None):
        if v is not None:
            self.m = pymesh.form_mesh(v, f)
        elif filename is not None:
            self.m = pymesh.load_mesh(filename)
        self.m.add_attribute('vertex_red')
        self.m.add_attribute('vertex_green')
        self.m.add_attribute('vertex_blue')
        if vc is not None:
            self.vc = vc

    @property
    def v(self):
        return np.copy(self.m.vertices)

    @v.setter
    def v(self, value):
        self.m.vertices = value

    @property
    def f(self):
        return np.copy(self.m.faces)

    @f.setter
    def f(self, value):
        self.m.faces = value

    @property
    def vc(self):
        return np.stack((self.m.get_attribute('vertex_red'), self.m.get_attribute('vertex_green'),
                         self.m.get_attribute('vertex_blue')), 1)/255

    @vc.setter
    def vc(self, value):
        value = np.copy(value) * 255
        self.m.set_attribute('vertex_red', value[:, 0])
        self.m.set_attribute('vertex_green', value[:, 1])
        self.m.set_attribute('vertex_blue', value[:, 2])

    def write_obj(self, fpath):
        if not fpath.endswith('.obj'):
            fpath = fpath + '.obj'
        pymesh.save_mesh(fpath, self.m)

    def write_ply(self, fpath):
        if not fpath.endswith('.ply'):
            fpath = fpath + '.ply'
        if self.vc.size == 0:
            pymesh.save_mesh(fpath, self.m)
        else:
            # import IPython; IPython.embed()
            pymesh.save_mesh(fpath, self.m, 'vertex_red', 'vertex_green', 'vertex_blue')