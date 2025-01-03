import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
from utils.preprocessing import ColorPreprocessor


class ColorHarmonyDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train', n_colors=64):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.n_colors = n_colors
        self.preprocessor = ColorPreprocessor(n_colors=self.n_colors)  # 只传递n_colors
        self.data = self._load_data()
        
    def _load_data(self):
        data = []
        split_file = os.path.join(self.root_dir, f'{self.split}.txt')
        
        with open(split_file, 'r') as f:
            for line in f:
                img_path, harmony_score = line.strip().split(',')
                # data.append({
                #     'image': os.path.join(self.root_dir, img_path),
                #     'score': float(harmony_score)
                # })
                data.append([
                    os.path.join(self.root_dir, img_path),
                    float(harmony_score)
                ])
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 从data中获取图片路径和分数
        image_path, score = self.data[idx]
        # 打开并转换图片
        image = Image.open(image_path).convert('RGB')
        
        # 获取颜色直方图
        colors, hist, labels, _ = self.preprocessor.quantize_colors(image)
        color_hist = np.bincount(labels, minlength=self.n_colors)
        
        # 应用transform如果存在
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(color_hist, dtype=torch.float32), torch.tensor(score, dtype=torch.float32), colors