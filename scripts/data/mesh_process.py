import os
import trimesh
import numpy as np
from data_utils import data_processor
import cv2
from pytorch3d.io import load_ply, load_obj, save_ply, IO, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.loss import chamfer_distance
import torch
from probreg import cpd
from pytorch3d.transforms import RotateAxisAngle, Translate, Scale
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures
from utils.utils import save_tensor_image, mask_to_mesh
import torchvision.transforms as transforms




class MeshProcessor(data_processor):
    def __init__(self, root_dir):
        super().__init__(root_dir)
        self.all_deformed_mesh = [os.path.join(root_dir, 'deformation_eccv', i) for i in os.listdir(os.path.join(root_dir, 'deformation_eccv')) if 'canonical' not in i]
        self.all_deformed_mesh.sort()
        self.base_img_cvpr = [os.path.join(root_dir, 'deformation_cvpr/base_img', i) for i in os.listdir(os.path.join(root_dir, 'deformation_cvpr/base_img'))]
        self.base_mask_cvpr = [os.path.join(root_dir, 'deformation_cvpr/base_mask', i) for i in os.listdir(os.path.join(root_dir, 'deformation_cvpr/base_mask'))]
        self.base_shape =None
        self.deform_shape = [os.path.join(root_dir, 'deformation_cvpr/deform_shape', i) for i in os.listdir(os.path.join(root_dir, 'deformation_cvpr/deform_shape')) if not 'registrated' in i]
        self.base_img_cvpr.sort()
        self.base_mask_cvpr.sort()
        self.deform_shape.sort()
        
    def normalize(self, verts:torch.tensor):
        centroid = verts.mean(dim=0)
        # translate and sclae verts to canonical space
        normalized_verts = verts - centroid
        scale = torch.max(torch.abs(normalized_verts)).item()
        scale_factor = 1.0 / scale
        scale_transform = Scale(scale_factor)
        normalized_verts = scale_transform.transform_points(normalized_verts)

        return normalized_verts
    
    def uv_mapping(self,meshes:Meshes, texture_image:torch.tensor, save_mesh=False):
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


        return meshes
    
    def sample_points(self, mesh:Meshes, num_points=10000):
        verts = mesh.verts_packed()
        faces = mesh.faces_packed()
        sampled_points = Meshes(verts=[verts], faces=[faces]).sample_points(num_points)
        return sampled_points
    
    def find_paired_deformed_mesh(self,basepoints,num=3):
        chamfer_list = []
        for i, deformed_file in enumerate(self.all_deformed_mesh):
            deformed = load_obj(deformed_file)
            deformed_points = deformed[0]
            chamfer_list.append(chamfer_distance(basepoints.unsqueeze(0), deformed_points.unsqueeze(0))[0])
        chamfer_list = torch.stack(chamfer_list)
        _, indices = torch.topk(chamfer_list, num)
        return indices

    def img_to_tensor(self, img:np.array):
        img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float() / 255
        return img_tensor
    
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

    def crop_img_mask(self, mask:np.array, img:np.array,mask_size=512):
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x, y, w, h = cv2.boundingRect(gray)
        if w > h:
            pad = (w - h) // 2
            crop = mask[max(0, y - pad):y + h + pad, x:x + w]
            crop_img = img[max(0, y - pad):y + h + pad, x:x + w]
        else:
            pad = (h - w) // 2
            crop = mask[y:y + h, max(0, x - pad):x + w + pad]
            crop_img = img[y:y + h, max(0, x - pad):x + w + pad]

        crop_resized = cv2.resize(crop, (mask_size, mask_size))
        mask_tensor = self.img_to_tensor(crop_resized)
        # crop img accoding to mask
        img_resized = cv2.resize(crop_img, (mask_size, mask_size))
        img_tensor = self.img_to_tensor(img_resized)
        # set mask==0 area to 0 in img
        img_tensor = img_tensor * mask_tensor
        return mask_tensor, img_tensor


if __name__ == "__main__":
    root = 'dataset/'
    # set device to gpu 1
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    processor = MeshProcessor(root)
    # new registration test
    resize_transform = transforms.Resize((256, 256))

    crop = False
    base_img = processor.base_img_cvpr
    base_mask = processor.base_mask_cvpr
    for i in range(len(base_mask)):
        mask_file = base_mask[i]
        img_file = base_img[i]
        mask = cv2.imread(mask_file)
        img = cv2.imread(img_file)
        if not mask.shape[0]==mask.shape[1]:
            mask, img  = processor.crop_img_mask(mask, img)
            # save mask and img 
            save_tensor_image(mask,mask_file )
            save_tensor_image(img,img_file)       
        mask = cv2.imread(mask_file)
        texture  = cv2.imread(img_file)
        texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
        texture_tensor = processor.img_to_tensor(texture)
        mask_tensor = processor.img_to_tensor(mask)
        base_mesh = mask_to_mesh(resize_transform(mask_tensor), device)
        # textured_mesh= processor.uv_mapping(base_mesh, texture_tensor, save_mesh=True)
        deformed_shape = load_ply(processor.deform_shape[i])
        deformed_shape = Meshes(verts=[deformed_shape[0].to(device)], faces=[deformed_shape[1].to(device)])
        base_sampled = sample_points_from_meshes(base_mesh, 10000)
        deformed_sampled = sample_points_from_meshes(deformed_shape, 10000)
        registration = processor.nonrigid_cpd_cuda(processor.normalize(base_mesh.verts_packed()),processor.normalize(deformed_sampled.cpu().squeeze()), use_cuda=True)
        base_registration = Meshes(verts=[registration.to(device)], faces=[base_mesh.faces_packed().to(device)])
        save_obj('registration.obj', registration, base_mesh.faces_packed())
        pass

    # test uv mapping 
    # mesh = trimesh.load(processor.all_base_mesh[0])
    # texture = cv2.imread(processor.all_rgb[0])
    # save_folder = os.path.join(root,'deformation_pairs')
    # canonical_folder = os.path.join(save_folder,'canonical')
    # deformed_folder = os.path.join(save_folder,'deformed')
    # if not os.path.exists(save_folder):
    #     os.makedirs(save_folder)
    # if not os.path.exists(canonical_folder):
    #     os.makedirs(canonical_folder)
    # if not os.path.exists(deformed_folder):
    #     os.makedirs(deformed_folder)
    
    # get base-deform pair
    # for i in range(len(processor.all_base_mesh)):
    #     base = load_ply(processor.all_base_mesh[i])
    #     base_points = base[0]
    #     base_points_normalized = processor.normalize(base_points)
    #     base_points_normalized[torch.isnan(base_points_normalized)] = 0
    #     new_base = Meshes(verts=[base_points_normalized.to(device)], faces=[base[1].to(device)])
    #     save_ply(os.path.join(canonical_folder,f'base_{i}.ply'), new_base.verts_list()[0], new_base.faces_list()[0])
    #     shape_idx = processor.find_paired_deformed_mesh(base_points_normalized)

        
    #     for j in shape_idx:
    #         # non-rigid registration
    #         deformed = load_obj(processor.all_deformed_mesh[j])
    #         # rotate deformed mesh
            
    #         deformed_points = deformed[0]
    #         deformed_points_rot = RotateAxisAngle(90,axis='z').transform_points(deformed_points)
    #         # test rotate
    #         # deformed_rot_mesh = Meshes(verts=[deformed_points_rot.to(device)], faces=[deformed[1].verts_idx.to(device)])
    #         # save_ply(f'deformed_rot_{i}.ply', deformed_rot_mesh.verts_list()[0], deformed_rot_mesh.faces_list()[0])
    #         registrated_points = processor.nonrigid_cpd_cuda(base_points_normalized, deformed_points)
    #         registrated_mesh = Meshes(verts=[registrated_points.to(device)], faces=[base[1].to(device)])
        
    #         # save registrated mesh
    #         save_ply(os.path.join(deformed_folder,f'{i}_{j}.ply'), registrated_mesh.verts_list()[0], registrated_mesh.faces_list()[0])
    #         print(f"save {i}_{j}.ply")
        

        