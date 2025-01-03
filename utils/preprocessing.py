import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from sklearn.cluster import KMeans

class ColorPreprocessor:
    def __init__(self, n_colors=64):
        self.n_colors = n_colors
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 固定大小
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
    def quantize_colors(self, image):
        # 转换为Lab颜色空间
        image_lab = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB)
        h, w, _ = image_lab.shape
        pixels = image_lab.reshape(-1, 3)
        
        # KMeans颜色量化
        kmeans = KMeans(n_clusters=self.n_colors, random_state=42)
        labels = kmeans.fit_predict(pixels)
        
        # 获取量化后的颜色和直方图
        colors = kmeans.cluster_centers_
        color_hist = np.bincount(labels, minlength=self.n_colors) / len(labels)
        
        return colors, color_hist, labels, (h, w, 3)
    
    def reconstruct_image(self, labels, colors, image_shape):
        quantized_pixels = colors[labels].astype('uint8')
        quantized_image_lab = quantized_pixels.reshape(image_shape)
        quantized_image_rgb = cv2.cvtColor(quantized_image_lab, cv2.COLOR_LAB2RGB)
        return quantized_image_rgb
    
    def preprocess_image(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self.image_transform(image)
    
    def prepare_lda_data(self, image_color_indices):
    # 将每张图像的颜色聚类标签转换为颜色词计数
        k = self.n_colors
        data = np.bincount(image_color_indices, minlength=k)
        data = np.array(data)
        # LDA 通常需要计数数据，而不是归一化的概率分布
        return data
