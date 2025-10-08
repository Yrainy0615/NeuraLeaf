import os
import shutil
import argparse
import subprocess


def group_images(source_folder,  group_size=40):
    # 确保源文件夹存在
    if not os.path.exists(source_folder):
        raise ValueError(f"Source folder {source_folder} does not exist.")

    # 获取所有.jpg文件
    jpg_files = [f for f in os.listdir(source_folder) if f.lower().endswith('.jpg')]
    jpg_files.sort()  # 按文件名排序
    folder_name = os.path.basename(source_folder)
  
    # 分组保存图片
    for i in range(0, len(jpg_files), group_size):
        # 创建新的文件夹
        group_folder = os.path.join(source_folder, f'leaf_{i // group_size + 1}')
        print(group_folder)
        while os.path.exists(group_folder):
            i += 1
            group_folder = os.path.join(source_folder, f'leaf_{i // group_size + 1}')
        os.makedirs(group_folder, exist_ok=True)
        
        # 把当前组的文件复制到新文件夹
        for j in range(i, min(i + group_size, len(jpg_files))):
            source_file = os.path.join(source_folder, jpg_files[j])
            target_file = os.path.join(group_folder, jpg_files[j])
            shutil.move(source_file, target_file)



def run_workflow(root_folder, mask_file):
    if not os.path.exists(root_folder):
        raise ValueError(f"Root folder {root_folder} does not exist.")

    for subdir, dirs, files in os.walk(root_folder):
        dirs.sort()
        for dir in dirs:
            folder_path = os.path.join(subdir, dir)
            point_cloud_file = os.path.join(folder_path, 'point_cloud.ply')
            project_file = os.path.join(folder_path, 'project.psx')
            project_folder = os.path.join(folder_path, 'project.files')

            # 检查是否存在点云文件，如果存在，则跳过
            if os.path.exists(point_cloud_file):
                print(f"Skipping {folder_path}, point cloud file exists.")
                continue

            # 检查是否存在项目文件但不存在点云文件
            if os.path.exists(project_file) and not os.path.exists(point_cloud_file):
                print(f"Deleting old project file and reprocessing: {project_file}")
                os.remove(project_file)
                os.remove(project_folder)

            print(f"Running workflow for {folder_path}")
            try:
                subprocess.run([
                    'python', 'general_workflow.py',
                    '--image_folder', folder_path,
                    '--output_folder', folder_path,
                    '--mask_file', mask_file
                ], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error processing {folder_path}: {e}")
                continue  # Skip to the next folder after logging the error

def copy_and_rename(source_folder,base_folder,target_shape, target_image,target_mask):
    subfolders = [f.path for f in os.scandir(source_folder) if f.is_dir()]
    subfolders.sort()  
    base_imgs = [f for f in os.scandir(base_folder) if f.is_file()]
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
        shutil.copy(base_img, os.path.join(target_image, f'leaf_{start_index + i}.jpg'))
        shutil.copy(base_mask, os.path.join(target_mask, f'leaf_{start_index + i}.png'))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess')
    parser.add_argument('--source_folder', type=str, default='dataset/deformation_cvpr/raw_capture/240906', help='source folder')
    parser.add_argument('--mask_file', type=str, default=None, help='mask file')
    args = parser.parse_args()
    group_images(args.source_folder)
    run_workflow(args.source_folder, args.mask_file)
    