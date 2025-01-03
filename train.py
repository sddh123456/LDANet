import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 必须在导入 pyplot 之前设置后端
import matplotlib.pyplot as plt  # 添加 matplotlib 库

from PIL import Image
import numpy as np
import joblib  # 用于保存 LDA 模型

from config import Config
from models.color_model import ColorHarmonyNet, ColorLDAModel
from data.dataset import ColorHarmonyDataset
from utils.preprocessing import ColorPreprocessor
from utils.metrics import calculate_metrics
from utils.visualization import Visualizer
from utils.getFID import FIDCalculator


def train(model, lda_model, train_loader, criterion, optimizer, device, epoch, visualizer, config):
    model.train()
    total_loss = 0
    batch_losses = []
    FID = FIDCalculator(config)
    
    # 使用 tqdm 添加训练进度条
    progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config.EPOCHS}] Training", leave=False)

    for batch_idx, (images, infrared_images, color_hist, targets, color) in enumerate(progress_bar):
    # for batch_idx, (images, color_hist, targets, color) in enumerate(progress_bar):
        images = images.to(device)
        infrared_images = infrared_images.to(device)
        color_hist = color_hist.to(device).float()
        targets = targets.to(device).view(-1, 1)
        
        optimizer.zero_grad()
        
        # 模型前向传播，得到彩色图像的输出和特征
        outputs_color, deep_features_color, lda_features_color, lda_topics = model(images, color_hist)
        
        # 模型前向传播，得到红外图像的特征（无需计算输出）
        with torch.no_grad():
            _, deep_features_infrared, _, _ = model(infrared_images, color_hist)
        
        # 计算图像质量预测损失
        loss_color = criterion(outputs_color, targets)
        
        # 计算特征分布损失（FID）
        fid_loss = FID.calculate_fid_loss(deep_features_color, deep_features_infrared)
        
        # 总损失
        total_loss = 0.5 * loss_color + config.LAMBDA_FID * fid_loss
        # total_loss = loss_color 
        
        total_loss.backward()
        optimizer.step()
        
        total_loss_item = total_loss.item()
        total_loss += total_loss_item
        batch_losses.append(total_loss_item)
        
        # 更新进度条
        progress_bar.set_postfix({
            'Total Loss': total_loss.item(),
            'Color Loss': loss_color.item(),
            'FID Loss': fid_loss.item()
        })

        # progress_bar.set_postfix({
        #     'Total Loss': total_loss.item(),
        #     'Color Loss': loss_color.item()
        # })
        
        # 根据配置，决定是否进行可视化
        if config.ENABLE_LDA_VISUALIZATION and batch_idx % config.VISUALIZATION_INTERVAL == 0:
            # 保存原始图像
            # 反归一化
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(device)
            orig_image = images[0] * std + mean
            
            # 转换为numpy并调整通道顺序
            orig_image = orig_image.cpu().numpy().transpose(1, 2, 0)
            
            # 将值域调整到[0,1]
            orig_image = np.clip(orig_image, 0, 1)
            
            # 转为PIL图像并保存
            orig_image = Image.fromarray((orig_image * 255).astype(np.uint8))
            orig_image.save(os.path.join(config.VISUALIZATION_DIR, 
                                        f'original_image_epoch{epoch}_batch{batch_idx}.png'))
                    
            # 可视化颜色直方图
            visualizer.plot_color_distribution(
                colors=np.arange(len(color_hist[0])),
                weights=color_hist[0].cpu().numpy(),
                save_path=os.path.join(config.VISUALIZATION_DIR, f'color_hist_epoch{epoch}_batch{batch_idx}.png')
            )
            
            # 可视化 LDA 分类后颜色块图像
            lda_topic_distribution = lda_topics[0]
            visualizer.visualize_lda_classification(
                lda_topic_distribution, lda_model, color,
                save_path=os.path.join(config.VISUALIZATION_DIR, f'lda_classification_epoch{epoch}_batch{batch_idx}.png')
            )
            
    average_loss = total_loss / len(train_loader)
    return average_loss, batch_losses


def validate(model, val_loader, criterion, device, epoch, visualizer, config):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    # 使用 tqdm 添加验证进度条
    progress_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{config.EPOCHS}] Validation", leave=False)

    with torch.no_grad():
        for images, color_hist, targets, _ in progress_bar:
            images = images.to(device)
            color_hist = color_hist.to(device)
            targets = targets.to(device).view(-1, 1) 
            
            outputs, deep_features, lda_features, _ = model(images, color_hist)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # 更新进度条后缀，显示当前损失
            progress_bar.set_postfix({'Loss': loss.item()})
    
    average_loss = total_loss / len(val_loader)
    metrics = calculate_metrics(torch.tensor(all_preds), torch.tensor(all_targets))
    
   # 绘制验证集评估指标
    visualizer.plot_color_distribution(
        deep_features[0].cpu().numpy(),
        outputs[0].cpu().numpy(),
        save_path=os.path.join(config.VISUALIZATION_DIR, f'color_dist_epoch{epoch}.png')
    )
    
    # visualizer.plot_color_distribution(
    #     color_hist[0].cpu().numpy(),
    #     lda_features[0].cpu().numpy(),
    #     save_path=os.path.join(config.VISUALIZATION_DIR, f'lda_features_epoch{epoch}.png')
    # )
    
    return average_loss, metrics


def main():
    config = Config()
    device = torch.device(config.DEVICE)
    
    # 创建可视化目录
    os.makedirs(config.VISUALIZATION_DIR, exist_ok=True)
    
    # 如果需要保存 LDA 模型的目录不存在，则创建
    lda_model_dir = os.path.dirname(config.LDA_MODEL_SAVE_PATH)
    if config.SAVE_LDA_MODEL and not os.path.exists(lda_model_dir):
        os.makedirs(lda_model_dir, exist_ok=True)
    
    # 初始化可视化工具
    visualizer = Visualizer(config)
    
    # Data loading
    preprocessor = ColorPreprocessor(config.NUM_COLORS)
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = ColorHarmonyDataset(config.DATA_ROOT, transform=transform, split='train', n_colors=config.NUM_COLORS)
    val_dataset = ColorHarmonyDataset(config.DATA_ROOT, transform=transform, split='val', n_colors=config.NUM_COLORS)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                            shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                          shuffle=False, num_workers=config.NUM_WORKERS)
    
    # 将原来的LDA模型训练部分替换为:
    if os.path.exists(config.LDA_MODEL_SAVE_PATH):
        # 如果存在预训练模型，直接加载
        print(f"正在加载预训练的LDA模型: {config.LDA_MODEL_SAVE_PATH}")
        lda_model = joblib.load(config.LDA_MODEL_SAVE_PATH)
    else:
        # 如果不存在预训练模型，训练新模型
        print("未找到预训练LDA模型，开始训练新模型...")
        all_color_hists = []
        for img_path, _ in train_dataset.data:
            image = Image.open(img_path).convert('RGB')
            _, _, labels, _ = preprocessor.quantize_colors(np.array(image))
            color_hist = np.bincount(labels, minlength=config.NUM_COLORS)
            all_color_hists.append(color_hist)
        
        lda_model = ColorLDAModel(n_topics=config.NUM_TOPICS)
        lda_model.fit(all_color_hists)
        
        # 如果配置为保存LDA模型，则保存
        if config.SAVE_LDA_MODEL:
            os.makedirs(os.path.dirname(config.LDA_MODEL_SAVE_PATH), exist_ok=True)
            joblib.dump(lda_model, config.LDA_MODEL_SAVE_PATH)
            print(f"LDA模型已保存: {config.LDA_MODEL_SAVE_PATH}")
    
    # 将训练好的LDA模型传递给ColorHarmonyNet
    model = ColorHarmonyNet(config, lda_model)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE,
                               weight_decay=config.WEIGHT_DECAY)
    
    start_epoch = 0
    # 如果配置了继续训练
    if config.RESUME_TRAINING and config.PRETRAINED_MODEL_PATH:
        if os.path.exists(config.PRETRAINED_MODEL_PATH):
            print(f"正在加载预训练模型: {config.PRETRAINED_MODEL_PATH}")
            checkpoint = torch.load(config.PRETRAINED_MODEL_PATH)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"将从 epoch {start_epoch} 开始继续训练")
        else:
            print(f"警告：预训练模型路径 {config.PRETRAINED_MODEL_PATH} 不存在")

    # Training history
    train_losses = []
    val_losses = []
    metrics_history = {
        'mse': [],
        'r2': []
    }
    epochs_list = []
    
    # 修改训练循环的起始位置
    for epoch in range(start_epoch, config.EPOCHS):
        train_loss, batch_losses = train(model, lda_model, train_loader, criterion, optimizer, 
                                       device, epoch, visualizer, config)
        val_loss, metrics = validate(model, val_loader, criterion, device, 
                                   epoch, visualizer, config)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        metrics_history['mse'].append(metrics['mse'])
        metrics_history['r2'].append(metrics['r2'])
        epochs_list.append(epoch + 1)
        
        # 实时绘制训练和验证损失曲线
        plt.figure(figsize=(10, 4))
        plt.plot(epochs_list, train_losses, label='Training loss')
        plt.plot(epochs_list, val_losses, label='Proof loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train and verify loss curves')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(config.VISUALIZATION_DIR, 'loss_curve.png'))
        plt.close()

        # 实时绘制指���曲线（如 MSE）
        plt.figure(figsize=(10, 4))
        plt.plot(epochs_list, metrics_history['mse'], label='verify MSE')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('Verify the set MSE curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(config.VISUALIZATION_DIR, 'mse_curve.png'))
        plt.close()
        
        # 在训练循环中，收集并绘制损失和指标
        visualizer.plot_training_history(
            train_losses, val_losses, metrics_history,
            save_path=os.path.join(config.VISUALIZATION_DIR, 'training_history.png')
        )
        
        print(f'Epoch {epoch+1}/{config.EPOCHS}:')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Metrics: {metrics}')
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, f'{config.CHECKPOINT_DIR}/checkpoint_{epoch+1}.pth')

if __name__ == '__main__':
    main()
