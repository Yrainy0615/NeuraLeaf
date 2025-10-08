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

    def raw_to_canonical(self,path,rotate_x=False,export=False):
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
            cp.cuda.Device(4).use()

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
        acpd = cpd.NonRigidCPD(source_pt, use_cuda=use_cuda)
        tf_param_nrgd, _ ,_ = acpd.registration(target_sample)
        result_nrgd = tf_param_nrgd.transform(source_pt)
        # registrated_mesh = Meshes(verts=[result], faces=[base.faces_packed()])
        return torch.tensor(result_nrgd)

    def pointcloud_reconstruction(self,obj_file, point_dir,method='ball_pivoting'):
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
        elif method == 'ps':
            self.triangulate_points(obj_file,point_dir)
        elif method == "nonrigid":
            # rigid + non-rigid registration
            base_mesh = load_obj('dataset/cvpr_final/base_shape/leaf_5.obj')
            base_points = base_mesh[0]
            deform_files = [os.path.join(point_dir,f) for f in os.listdir(point_dir) if f.endswith('.ply')]
            save_path = os.path.join(point_dir,'fit_pca')
            os.makedirs(save_path,exist_ok=True)
            for file in deform_files:
                save_name = os.path.join(save_path,os.path.basename(file).split('.')[0]+'.obj')
                if not os.path.exists(save_name):
                    deform_points = load_ply(file)[0]
                    base_points_rigid = self.rigid_cpd_cuda(deform_points,base_points,use_cuda=True)
                    base_points_nonrigid = self.nonrigid_cpd_cuda(base_points_rigid,deform_points,use_cuda=True)
                    base_points_rigid = torch.tensor(base_points_rigid).unsqueeze(0)
                    base_points_save = torch.tensor(base_points_nonrigid).unsqueeze(0)
                    new_mesh_rigid = Meshes(verts=base_points_rigid, faces=base_mesh[1][0].unsqueeze(0))
                    new_mesh = Meshes(verts=base_points_save, faces=base_mesh[1][0].unsqueeze(0))
                    IO().save_mesh(new_mesh,save_name)
                    print(f'{save_name} saved')
                else:
                    print(f'{save_name} already exists')
                
            
        else:
            raise ValueError('method not supported')
        # export mesh
        save_folder = os.path.join(point_dir,'fit_ps')
        save_name = os.path.join(save_folder, f'{file_name}.obj')
        # o3d.io.write_triangle_mesh(save_name, mesh)
        print(f'{save_name} is saved')



    def rigid_cpd_cuda(self,basepoints:torch.tensor, deformed_points:torch.tensor,use_cuda=True):
        import cupy as cp
        if use_cuda:
            to_cpu = cp.asnumpy
            cp.cuda.Device(3).use()
            cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
        else: 
            cp = np
            to_cpu = lambda x: x
        source_pt = cp.asarray(basepoints)
        target_pt = cp.asarray(deformed_points)
        if source_pt.shape[0]>10000:
            rabdom_index = np.random.choice(source_pt.shape[0], 10000, replace=False)
            source_pt = source_pt[rabdom_index]
        rcpd = cpd.RigidCPD(target_pt, use_cuda=use_cuda)
        tf_param_rgd, _, _ = rcpd.registration(source_pt)
        target_rgd = tf_param_rgd.transform(target_pt)
        return to_cpu(target_rgd)
    
    def parametric_surface_fitting(self,pointfile,point_path):
        pointcloud = load_ply(pointfile)
        points = pointcloud[0]
        points = points.numpy()
        size_u = int(np.sqrt(points.shape[0]))
        size_v = int(size_u)
        degree_u = 2
        degree_v = 2

        # Do global surface approximation
        # surf = fitting.approximate_surface(points, size_u, size_v, degree_u, degree_v)
        surf = fitting.approximate_surface(points, size_u, size_v, degree_u, degree_v, ctrlpts_size_u=size_u-5, ctrlpts_size_v=size_v-5)
        # export surface
        save_folder = os.path.join(point_path,'fit_ps')

        os.makedirs(save_folder,exist_ok=True)
        save_name = os.path.join(save_folder, os.path.basename(pointfile).split('.')[0]+'.obj')
        surf.delta = 0.01
        exchange.export_obj(surf, save_name)
        print(f'{save_name} is saved')
    
    def triangulate_points(self,point_file,point_dir):
        pointcloud = load_ply(point_file)
        points = pointcloud[0]
        # triangulate points
        if points.shape[0] > 200:
            random_index = torch.randperm(points.shape[0])[:100]
            points = points[random_index]
        pc = mn.pointCloudFromPoints(points.numpy())
    
        # pc.validPoints = mm.pointUniformSampling(pc, 1e-3)
        # pc.invalidateCaches()
        mesh = mm.triangulatePointCloud(pc)
        # hole_loops = mm.findHoleLoops(mesh.topology)
        # for loop in hole_loops:
        #     mm.fillHole(mesh, loop)
        out_faces = mn.getNumpyFaces(mesh.topology)
        verts = mn.getNumpyVerts(mesh)
        mesh_save = trimesh.Trimesh(verts, out_faces, process=False)
        save_path = os.path.join(point_dir,'fit_ps')
        os.makedirs(save_path,exist_ok=True)
        save_name = os.path.join(save_path, os.path.basename(point_file).split('.')[0]+'.obj')
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
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    device = torch.device("cuda:2")


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
        processor = MeshProcessor(root)
        point_dir = 'dataset/denseleaf/data_00001/split/'
        point_file = [os.path.join(point_dir,f) for f in os.listdir(point_dir) if f.endswith('.obj')]
        for file in point_file:
            processor.parametric_surface_fitting(file,point_dir)
        
    
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
        point_dir = 'dataset/denseleaf/data_00001/split/'
        processor = MeshProcessor(root)
        point_file = [os.path.join(point_dir,f) for f in os.listdir(point_dir) if f.endswith('.ply')]
        for file in point_file:
            processor.pointcloud_reconstruction(file, point_dir,method='nonrigid')

    # rigid registration
    if regis:
        processor = MeshProcessor(root)
        # base_mesh = processor.all_base_shape
        deform_mesh_path = 'dataset/denseleaf/data_00001/split/'
        deform_mesh = [os.path.join(deform_mesh_path,f) for f in os.listdir(deform_mesh_path) if f.endswith('.ply')]        
        for i in range(len(deform_mesh)):

                # processor.raw_to_canonical(deform_mesh[i],rotate_x=False)
                # deform = load_objs_as_meshes([deform_mesh[i]])
                deform = load_ply(deform_mesh[i])
                # base_name = os.path.basename(deform_mesh[i]).split('.')[0]
                deform_name = os.path.basename(deform_mesh[i]).split('.')[0]
        
                bath_path = 'dataset/denseleaf/data_00001/split_regis/fit_pca'
                deform_train_path = os.path.join(bath_path,'fit_pca')
                os.makedirs(deform_train_path,exist_ok=True)

                base_file = os.path.join(bath_path, deform_name+'.obj')
                base = load_objs_as_meshes([base_file])
                # base = load_objs_as_meshes(['dataset/cvpr_final/base_shape/leaf_5.obj'])

                deform_save_path = os.path.join(deform_train_path,deform_name+'.obj')
                if not os.path.exists(deform_save_path):
                    # base_points = base.verts_packed()
                    base_points = base.verts_packed()
                    # base_points = base_points - base_points.mean(dim=0)
                    # deform_points = deform.verts_packed()
                    deform_points = deform[0]
                    # sample 20000 points
                    # deform_points = deform_points[torch.randperm(base_points.shape[0])[:10000]]
                    deform_trans = processor.rigid_cpd_cuda(deform_points, base_points.squeeze(),use_cuda=True)
                    deform_verts=torch.tensor(deform_trans).unsqueeze(0)
                    new_mesh = Meshes(verts=deform_verts, faces=base.faces_packed().unsqueeze(0),textures=base.textures)
                    # raw to canonical scaling and mean 


                    # tri_pts.export(deform_save_path)
                    # deform_regis = Meshes(verts=deform_verts, faces=deform[1][0].unsqueeze(0))
                    # tri_pts = trimesh.PointCloud(vertices=deform_trans)
                    # tri_pts.export(deform_save_path)
                    # save_ply(deform_save_path, deform_regis)
                    # deform_save_path = os.path.join(regis_path,deform_name+'_regis.ply')
                    IO().save_mesh(new_mesh,deform_save_path)
                    print(f'{deform_save_path} saved')
                else:
                    print(f'{deform_save_path} already exists')
    
    # non-rigid registration
    if non_rigid:
        processor = MeshProcessor(root)
        # deform_mesh = processor.all_deform_train
        deform_mesh_path = 'dataset/denseleaf/data_00001/split_regis/'
        deform_mesh = [os.path.join(deform_mesh_path,f) for f in os.listdir(deform_mesh_path) if f.endswith('.ply')]  
        base = load_objs_as_meshes(['dataset/cvpr_final/base_shape/leaf_5.obj']) # 0003_0050 for 0002
        base_points = base.verts_packed().unsqueeze(0)
        base_faces = base.faces_packed().unsqueeze(0)
        for i in range(len(deform_mesh)):
                    # if 'regis' in deform_mesh[i]:
                    # base = load_obj(base_mesh[i])
                        deform_points = load_ply(deform_mesh[i])[0]
                        # deform_points = processor.raw_to_canonical(deform_mesh[i],rotate_x=False)
                        deform_name = os.path.basename(deform_mesh[i]).split('.')[0]
                        deform_train_path = os.path.join(deform_mesh_path,'fit_pca')
                        # base_file = os.path.join(deform_train_path,deform_name+'_rigid.obj')
                        # base = load_objs_as_meshes([base_file])
                        base_points = base.verts_packed()
                        deform_save_path = os.path.join(deform_train_path,deform_name+'.obj')
                        os.makedirs(deform_train_path,exist_ok=True)
                        if not os.path.exists(deform_save_path):
                            # deform_points = deform.verts_packed()
                            # sample 20000 points
                            deform_trans = processor.nonrigid_cpd_cuda(base_points.squeeze(),deform_points,use_cuda=True)
                            deform_verts=torch.tensor(deform_trans).unsqueeze(0)
                            deform_mesh_save = Meshes(verts=deform_verts, faces=base.faces_packed().unsqueeze(0), textures=base.textures)
                            # tri_pts = trimesh.PointCloud(vertices=deform_trans)
                            # tri_pts.export(deform_save_path)
                            # save_ply(deform_save_path, deform_regis)
                            # deform_save_path = os.path.join(regis_path,deform_name+'_regis.ply')
                            IO().save_mesh(deform_mesh_save,deform_save_path)
                            print(f'{deform_save_path} saved')
                        else:
                            print(f'{deform_save_path} already exists')
        # for i in range(len(deform_mesh)):
        #     if deform_mesh[i].endswith('.ply'): 
        #         deform = load_ply(deform_mesh[i])
        #     elif deform_mesh[i].endswith('.obj'):
        #         deform = load_obj(deform_mesh[i])
            
        #     deform_name = os.path.basename(deform_mesh[i]).split('.')[0]
        #     if '_deform' in deform_name:
        #         base_name = deform_name.replace('_deform','')
        #     else:
        #         base_name = deform_name
        #     base_path = os.path.join(root, 'base_shape', f'{base_name}.obj')
        #     deform_path = deform_mesh[i]
        #     regis_path = 'results/cvpr/fitting/cpd'
        #     os.makedirs(regis_path,exist_ok=True)
        #     deform_save_path = os.path.join(regis_path,f'{deform_name}.obj')
        #     if not os.path.exists(deform_save_path):
        #         base = load_objs_as_meshes([base_path])
        #         print(f'processing {deform_mesh[i]} and {base_path}')
        #         base_points = base.verts_packed()
        #         deform_points = deform[0]
        #         base_trans = processor.nonrigid_cpd_cuda(base_points,deform_points, use_cuda=True)
        #         base_regis = Meshes(verts=base_trans.unsqueeze(0), faces=base.faces_packed().unsqueeze(0),textures=base.textures)
        #         # save registration with texture 
        #         IO().save_mesh(base_regis,deform_save_path)
        #         print(f'{deform_save_path} saved')
        #     else:
        #         print(f'{deform_save_path} already exists')      
    
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
        


