import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torchvision.utils import make_grid
import cv2
import os

class Visualizer:
    def __init__(self, config):
        self.config = config
        
    def plot_attention_maps(self, original_image, spatial_attention, save_path):
        """
        绘制热力图叠加在原始图像上
        Args:
            original_image: 原始图像 numpy array [H,W,3], 值范围 [0,1]
            spatial_attention: 注意力图 numpy array [H,W], 值范围 [0,1]
            save_path: 保存路径
        """
        plt.figure(figsize=(15, 5))
        
        # 1. 显示原始图像 - 归一化到[0,1]范围
        plt.subplot(131)
        original_image = np.clip((original_image - original_image.min()) / 
                               (original_image.max() - original_image.min()), 0, 1)
        plt.imshow(original_image)
        plt.title('Original Image')
        plt.axis('off')
        
        # 2. 显示注意力热力图
        plt.subplot(132)
        attention_map = spatial_attention
        if torch.is_tensor(attention_map):
            attention_map = attention_map.detach().cpu().numpy()
        if len(attention_map.shape) > 2:
            attention_map = attention_map.squeeze()
        
        # 归一化注意力图到 [0,1] 范围
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

        # 对注意力图进行反转
        attention_map = 1 - attention_map

        attention_map_resized = cv2.resize(attention_map, 
                                           (original_image.shape[1], original_image.shape[0]))
        
        plt.imshow(attention_map_resized, cmap='jet')
        plt.title('Attention Heatmap')
        plt.colorbar()
        plt.axis('off')
        
        # 3. 显示叠加后的图像
        plt.subplot(133)
        
        plt.imshow(original_image)
        plt.imshow(attention_map_resized, cmap='jet', alpha=0.35)
        plt.title('Overlay with Transparency')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_color_distribution(self, colors, weights, save_path=None):
        """绘制颜色分布
        Args:
            colors: numpy array，形状为(n_colors,) 或 (n_colors, 3)
            weights: numpy array，形状为(n_colors,)
            save_path: str，保存路径
        """
        plt.figure(figsize=(10, 2))
    
    # 将weights归一化为概率分布
        weights = np.array(weights)
        if weights.sum() != 0:  # 避免除零错误
            weights = weights / weights.sum()
        
        if len(colors.shape) == 1:
            # 如果输入是一维数组，直接用索引作为x轴
            plt.bar(np.arange(len(colors)), weights)
        else:
            # 确保颜色数组形状正确
            n_colors = len(weights)
            colors_reshaped = colors.reshape(-1, 3)
            
            # 将颜色值归一化到[0,1]范围
            colors_normalized = (colors_reshaped - colors_reshaped.min()) / (colors_reshaped.max() - colors_reshaped.min())
            
            # 转换颜色空间
            try:
                colors_rgb = cv2.cvtColor(
                    (colors_normalized.reshape(1, -1, 3) * 255).astype(np.uint8), 
                    cv2.COLOR_LAB2RGB
                )[0]
                # 使用转换后的RGB颜色绘制柱状图
                plt.bar(np.arange(n_colors), weights, color=colors_rgb/255.0)
            except cv2.error:
                # 如果颜色转换失败，用默认颜色
                plt.bar(np.arange(n_colors), weights)
        
        plt.title('Color Distribution')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    def plot_training_history(self, train_losses, val_losses, metrics_history, save_path=None):
        """绘制训练和验证损失，以及评估指标曲线"""
        epochs = range(1, len(train_losses) + 1)

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # 绘制损失曲线
        axes[0].plot(epochs, train_losses, label='train_loss')
        axes[0].plot(epochs, val_losses, label='val_loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and validating loss curves')
        axes[0].legend()
        axes[0].grid(True)

        # 绘制评估指标曲线（如 MSE 和 R²）
        axes[1].plot(epochs, metrics_history['mse'], label='verify MSE')
        axes[1].plot(epochs, metrics_history['r2'], label='verify R²')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Metric')
        axes[1].set_title('Validation Set Evaluation Metrics Curve')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def get_topic_colors(self, lda_model, n_topics):
        """
        获取每个topic下的主要颜色
        
        参数:
        - lda_model: 训练好的LDA模型
        - colors: 颜色中心点
        - n_topics: 主题数量
        """
        # 获取主题-颜色分布矩阵
        # topic_color_dist = lda_model.lda.components_  # shape: (n_topics, n_colors)
        topic_word_prob = lda_model.lda.components_ / lda_model.lda.components_.sum(axis=1, keepdims=True)
        
        # 获取每个主题下最重要的颜色索引
        topic_main_colors = []
        for topic_dist in topic_word_prob:
            # 取前3个最重要的颜色索引
            top_color_indices = np.argsort(topic_dist)[-8:][::-1]
            topic_main_colors.append({
                'indices': top_color_indices,
                'weights': topic_dist[top_color_indices]
            })
        return topic_main_colors


    def visualize_lda_classification(self, topic_dist, lda_model, colors, save_path=None, show_frame=False):
        n_topics = len(topic_dist)
        fig = plt.figure(figsize=(15, 8))
        topic_main_colors = self.get_topic_colors(lda_model, n_topics)

        # 创建x轴标签
        x_labels = [f'Topic {i+1}' for i in range(n_topics)]
        x_positions = np.arange(n_topics)

        # 主题分布条形图
        ax1 = plt.subplot(2, 1, 1)
        bars = ax1.bar(x_positions, topic_dist)
        ax1.set_title('Topic Distribution')
        # ax1.set_xlabel('Subject No.')
        ax1.set_ylabel('Scale')
        
        # 设置x轴刻度和标签
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels(x_labels)
        
        # 调整x轴标签的角度，避免重叠
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

        for bar, prop in zip(bars, topic_dist):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{prop:.2%}', ha='center', va='bottom')

        # 每个主题的主要颜色
        ax2 = plt.subplot(2, 1, 2)
        colors = colors.cpu().numpy()
        colors = colors[0]
        colors_uint8 = np.clip(colors, 0, 255).astype(np.uint8)
        colors_rgb = cv2.cvtColor(colors_uint8.reshape(1, -1, 3), cv2.COLOR_LAB2RGB)[0]

        # 其他参数设置保持不变
        square_size = 0.04
        y_spacing = 0.06
        max_colors = 5
        base_y = 0.7
        frame_color = 'lightblue'

        total_width = 1.0
        topic_width = total_width / n_topics
        
        for i in range(n_topics):
            topic_colors = topic_main_colors[i]
            indices = topic_colors['indices'][:max_colors]
            
            topic_center = i * topic_width + topic_width/2
            
            # 创建颜色方块
            for j, color_idx in enumerate(indices):
                color = colors_rgb[color_idx] / 255
                
                x = topic_center - square_size/2
                y = base_y - (j * y_spacing)
                
                if show_frame:
                    frame = plt.Rectangle((x-0.002, y-0.002), 
                                    square_size+0.004, square_size+0.004,
                                    facecolor='none',
                                    edgecolor=frame_color,
                                    linewidth=1)
                    ax2.add_patch(frame)
                
                rect = plt.Rectangle((x, y), square_size, square_size,
                                facecolor=color,
                                edgecolor='black',
                                linewidth=1)
                ax2.add_patch(rect)

            # 添加主题标签
            plt.text(topic_center, base_y + 0.05, f'Topic {i+1}',
                    ha='center', va='bottom', fontsize=8)

            plt.text(topic_center, base_y - max_colors * y_spacing + 0.04, '...',
                    ha='center', va='top', fontsize=10)

        ax2.set_xlim(-0.02, 1.02)
        ax2.set_ylim(0.2, 0.9)
        ax2.axis('off')
        # ax2.set_title('Main Colors for Each Topic')

        # 调整子图之间的间距
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=900)
            plt.close()
        else:
            plt.show()