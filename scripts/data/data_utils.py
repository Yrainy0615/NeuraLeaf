import os
import sys
sys.path.append('scripts')
from utils.utils import mask_to_mesh
from meshlib import mrmeshpy
import trimesh
import cv2
import numpy as np

class data_processor():
    def __init__(self, root_dir):
        self.all_masks= []
        self.all_rgb = []
        self.healthy_masks = []
        self.diseased_masks = []
        self.healthy_rgb = []
        self.diseased_rgb = []
        self.extra_mask = []
        self.extra_sdf = []
        self.all_sdf = []
        self.all_base_mesh = []
        self.all_deformed_mesh = None
        for root, dirs, files in os.walk(os.path.join(root_dir,'Leaf_RGB')):
            for file in files:
                if file.endswith('.png'):
                    if 'mask' in file:
                            self.all_masks.append(os.path.join(root, file))
                    else:   
                        self.all_rgb.append(os.path.join(root, file.replace('mask','rgb')))
        
        
                elif file.endswith('ply'):
                    self.all_base_mesh.append(os.path.join(root, file))
                elif file.endswith('.npy'):
                    self.all_sdf.append(os.path.join(root, file))
        
        for root, dirs, files in os.walk(os.path.join(root_dir,'canonical_mask')):
            for file in files:
                if file.endswith('.jpg'):
                    self.extra_mask.append(os.path.join(root, file))
                elif file.endswith('.npy'):
                    self.extra_sdf.append(os.path.join(root, file))
        self.healthy_rgb = [x for x in self.all_rgb if 'healthy' in x]
        self.healthy_masks = [x for x in self.all_masks if 'healthy' in x]
        self.diseased_rgb = [x for x in self.all_rgb if 'diseased' in x]
        self.diseased_masks = [x for x in self.all_masks if 'diseased' in x]
        self.healthy_sdf = [x for x in self.all_sdf if 'healthy' in x]
        self.diseased_sdf = [x for x in self.all_sdf if 'diseased' in x]
        self.all_rgb.sort()
        self.all_masks.sort()
        self.extra_mask.sort()
        self.healthy_masks.sort()
        self.healthy_rgb.sort()
        self.diseased_masks.sort()
        self.diseased_rgb.sort()
        self.healthy_sdf.sort()
        self.diseased_sdf.sort()
        self.all_sdf.sort()
        self.extra_sdf.sort()
        self.base_mask = self.healthy_masks + self.extra_mask
        self.base_sdf = self.healthy_sdf + self.extra_sdf
        self.base_mask.sort()
        self.base_sdf.sort()
        # save base_mask 
        if not os.path.exists(os.path.join(root_dir,'base_sdf.txt')):
            with open(os.path.join(root_dir,'base_sdf.txt'), 'w') as f:
                for item in self.base_sdf:
                    f.write("%s\n" % item)


    def mask_binary(self, mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return binary

    def align_mask(self, image_path, rgb_path):
        # 加载图像并转换为灰度图
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        rgb_image = cv2.imread(rgb_path)
        # rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        
        # 增加图像边界
        border_size = max(image.shape)
        image_with_border = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=0)
        rgb_with_border = cv2.copyMakeBorder(rgb_image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        # 二值化图像
        _, binary = cv2.threshold(image_with_border, 127, 255, cv2.THRESH_BINARY)
        
        # 找到叶子的轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 获取最大轮廓（假设只有一个叶子）
        contour = max(contours, key=cv2.contourArea)
        
        # 计算轮廓的质心
        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        # 主成分分析 (PCA)
        data = contour.reshape(-1, 2)
        data = data - np.mean(data, axis=0)
        cov = np.cov(data, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        principal_axis = eigenvectors[np.argmax(eigenvalues)]
        
        # 计算旋转角度，将主轴旋转到竖直方向
        angle = np.arctan2(principal_axis[1], principal_axis[0]) * 180 / np.pi
        if angle < 0:
            angle += 90
        else:
            angle -= 90

        # 旋转图像
        (h, w) = binary.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        rotated_rgb = cv2.warpAffine(rgb_with_border, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=[0, 0, 0])
        
        # 再次二值化以确保只有0和255
        _, rotated_binary = cv2.threshold(rotated, 127, 255, cv2.THRESH_BINARY)
        
        # 提取旋转后的轮廓
        contours, _ = cv2.findContours(rotated_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)

        # 检查叶子的方向，通过比较上半部分和下半部分的像素数
        top_half = rotated_binary[:h//2, :]
        bottom_half = rotated_binary[h//2:, :]
        
        if cv2.countNonZero(top_half) < cv2.countNonZero(bottom_half):
            # 如果顶部的白色像素比底部少，翻转图像
            rotated_binary = cv2.flip(rotated_binary, 0)
            rotated_rgb = cv2.flip(rotated_rgb, 0)

        # 获取mask的非零像素的坐标
        y, x = np.where(rotated_binary > 0)

        # 获取bounding box
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)

        # 裁剪图像
        cropped_mask = rotated_binary[y_min:y_max+1, x_min:x_max+1]
        cropped_image = rotated_rgb[y_min:y_max+1, x_min:x_max+1]

        # 计算新的大小，保持长宽比
        h, w = cropped_mask.shape
        aspect_ratio = w / h
        if h > w:
            new_h = 512
            new_w = int(new_h * aspect_ratio)
        else:
            new_w = 512
            new_h = int(new_w / aspect_ratio)

        # 调整大小
        resized_mask = cv2.resize(cropped_mask, (new_w, new_h), interpolation=cv2.INTER_AREA)
        resized_image = cv2.resize(cropped_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 在需要的地方填充0，以获得512x512的图像
        final_mask = np.zeros((512, 512), dtype=resized_mask.dtype)
        final_image = np.zeros((512, 512, 3), dtype=resized_image.dtype)
        y_offset = (512 - new_h) // 2
        x_offset = (512 - new_w) // 2
        final_mask[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_mask
        final_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_image
        final_image[final_mask == 0] = [0, 0, 0]
        # final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
        
        return final_mask, final_image
    
    def sample_deformation(self):
        pass
        


if __name__ == "__main__":
    root = 'dataset/2D_Datasets'
    processor = data_processor(root)
    for i in range(len(processor.extra_mask)):
        mask_binary = processor.mask_binary(processor.extra_mask[i])
        cv2.imwrite(processor.extra_mask[i], mask_binary)
        pass
        
    
      