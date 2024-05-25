from torch.utils.data import Dataset, DataLoader
import cv2
import os

class BaseShapeDataset(Dataset):
    def __init__(self, data_dir):
        self.masks = []
        for dirpath, dirnames, filenames in os.walk(data_dir):
            for filename in [f for f in filenames if f.endswith(".JPG") and 'mask' in f]:
                self.masks.append(os.path.join(dirpath, filename))
        self.labels = [self._get_label(mask) for mask in self.masks]
        self.label_to_index = self._create_label_to_index_map(self.labels)
    
    def _get_label(self, mask):
        return mask.split('/')[-1].split('_')[0]
    
    def _create_label_to_index_map(self, labels):
        unique_labels = list(set(labels))
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        return label_to_index
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        mask = cv2.imread(self.masks[idx])
        mask = cv2.resize(mask, (512, 512))
        label = self.masks[idx].split('/')[-1].split('_')[0]
        label_index = self.label_to_index[label]
        data = {
            'mask': mask,
            'label': label_index
        }
        return data


if __name__ == '__main__':
    dataset = BaseShapeDataset('dataset/LeafData')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for i, data in enumerate(dataloader):
        print(i, data['mask'].shape, data['label'])
        