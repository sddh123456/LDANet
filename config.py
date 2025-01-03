import torch
import os

class Config:
    # Data
    DATA_ROOT = "./data"  # 替换为实际数据路径
    IMAGE_SIZE = 224
    NUM_COLORS = 64
    BATCH_SIZE = 32
    NUM_WORKERS = 0
    
    # Model
    NUM_TOPICS = 20
    FEATURE_DIM = 2048
    
    # Training
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Logging
    LOG_INTERVAL = 10
    CHECKPOINT_DIR = "checkpoints"
    
    # Visualization
    ENABLE_LDA_VISUALIZATION = True  # Set to True to enable LDA visualization
    VISUALIZATION_INTERVAL = 100     # Interval for visualization (visualize every N batches)
    VISUALIZATION_DIR = "visualizations"
    
    # New configurations
    SAVE_LDA_MODEL = True  # 是否保存 LDA 模型的开关
    LDA_MODEL_SAVE_PATH = './models/lda_model.pkl'  # LDA 模型的保存路径
    
    # 添加预训练模型相关配置
    RESUME_TRAINING = True  # 是否继续训练
    PRETRAINED_MODEL_PATH = 'checkpoints/checkpoint_20.pth'  # 预训练模型路径

    LAMBDA_FID = 0.025 # FID 损失的权重
