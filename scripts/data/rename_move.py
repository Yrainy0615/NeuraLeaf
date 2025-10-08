import os
import shutil
import argparse



def copy_and_rename(source_folder,base_folder,target_shape, target_image,target_mask):
    subfolders = [f.path for f in os.scandir(source_folder) if f.is_dir()]
    subfolders.sort()  
    base_imgs = [f for f in os.listdir(base_folder) if f.endswith('.JPG')]
    base_imgs.sort()
    
    existing_files = [f for f in os.scandir(target_shape) if f.is_file()]
    existing_indices = [int(f.name.split('_')[-1].split('.')[0]) for f in existing_files]
    start_index = max(existing_indices) + 1 if existing_indices else 0
    
    for i, folder in enumerate(subfolders):
        obj_file = os.path.join(folder, 'model.obj')
        ply_file = os.path.join(folder, 'point_cloud.ply')
        base_img = os.path.join(base_folder, base_imgs[i])
        base_mask = os.path.join(folder, 'mask',base_imgs[i].replace('JPG','png'))
        shutil.copy(obj_file, os.path.join(target_shape, f'leaf_{start_index + i}.obj'))
        shutil.copy(ply_file, os.path.join(target_shape, f'leaf_{start_index + i}.ply'))
        shutil.copy(base_img, os.path.join(target_image, f'leaf_{start_index + i}.png'))
        shutil.copy(base_mask, os.path.join(target_mask, f'leaf_{start_index + i}.png'))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess')
    parser.add_argument('--source_folder', type=str, default='dataset/deformation_cvpr/raw_capture/240906', help='source folder')
    parser.add_argument('--base_folder', type=str, default='dataset/deformation_cvpr/base', help='base folder')
    parser.add_argument('--target_shape', type=str, default='dataset/deformation_cvpr/shape', help='target shape folder')
    parser.add_argument('--target_image', type=str, default='dataset/deformation_cvpr/image', help='target image folder')
    parser.add_argument('--target_mask', type=str, default='dataset/deformation_cvpr/mask', help='target mask folder')

    args = parser.parse_args()
    copy_and_rename(args.source_folder, args.base_folder, args.target_shape, args.target_image, args.target_mask)
  