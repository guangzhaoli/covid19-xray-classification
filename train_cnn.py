import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np
from models.autoencoder import Autoencoder
from models.simplecnn import SimpleCNN
from data.dataset import LungXrayDataset

def add_noise(images, noise_factor=0.3):
    """添加高斯噪声"""
    noisy_images = images + noise_factor * torch.randn_like(images)
    return torch.clamp(noisy_images, 0., 1.)

def plot_confusion_matrix(cm, classes, output_path):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def train_cnn(cnn_model, autoencoder, lr, train_loader, test_loader, num_epochs=100, 
              device='cuda', output_dir='results_cnn', noise_factor=0.3):
    """
    训练CNN模型
    Args:
        cnn_model: CNN模型
        autoencoder: 预训练的autoencoder模型
        lr: 学习率
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        num_epochs: 训练轮数
        device: 使用的设备
        output_dir: 输出目录
        noise_factor: 噪声因子
    """
    # 创建输出目录
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    tensorboard_dir = os.path.join(output_dir, 'tensorboard')
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # 初始化TensorBoard writer
    writer = SummaryWriter(tensorboard_dir)
    
    # 将模型移至指定设备
    cnn_model = cnn_model.to(device)
    autoencoder = autoencoder.to(device)
    autoencoder.eval()  # 设置autoencoder为评估模式
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=lr)
    
    # 记录最佳模型
    best_test_acc = 0.0
    
    # 用于记录训练历史
    history = {
        'train_loss': [],
        'test_loss': [],
        'train_acc': [],
        'test_acc': []
    }
    
    # 训练开始时间
    start_time = time.time()
    global_step = 0
    
    # 类别名称
    classes = ['Covid', 'Normal', 'Viral Pneumonia']
    
    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        cnn_model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        train_pbar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}] Training')
        
        for batch_idx, (data, targets) in enumerate(train_pbar):
            data, targets = data.to(device), targets.to(device)
            
            # 添加噪声
            noisy_data = add_noise(data, noise_factor)
            
            # 通过autoencoder降噪
            with torch.no_grad():
                denoised_data = autoencoder(noisy_data)
            
            # 前向传播
            outputs = cnn_model(denoised_data)
            loss = criterion(outputs, targets)
            
            # 计算准确率
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录损失
            train_loss += loss.item()
            
            # 记录到TensorBoard
            writer.add_scalar('Loss/train_step', loss.item(), global_step)
            
            # 更新进度条
            train_pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * train_correct / train_total
            })
            global_step += 1
        
        # 计算平均训练指标
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # 测试阶段
        cnn_model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc=f'Epoch [{epoch+1}/{num_epochs}] Testing')
            for data, targets in test_pbar:
                data, targets = data.to(device), targets.to(device)
                
                # 添加噪声并通过autoencoder降噪
                noisy_data = add_noise(data, noise_factor)
                denoised_data = autoencoder(noisy_data)
                
                outputs = cnn_model(denoised_data)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
                
                # 收集预测结果用于混淆矩阵
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                test_pbar.set_postfix({
                    'loss': loss.item(),
                    'acc': 100. * test_correct / test_total
                })
        
        # 计算平均测试指标
        test_loss = test_loss / len(test_loader)
        test_acc = 100. * test_correct / test_total
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        # 记录到TensorBoard
        writer.add_scalars('Loss/epoch', {
            'train': train_loss,
            'test': test_loss
        }, epoch)
        
        writer.add_scalars('Accuracy/epoch', {
            'train': train_acc,
            'test': test_acc
        }, epoch)
        
        # 每个epoch结束时绘制混淆矩阵
        cm = confusion_matrix(all_targets, all_predictions)
        plot_confusion_matrix(cm, classes, 
                            os.path.join(plot_dir, f'confusion_matrix_epoch_{epoch+1}.png'))
        
        # 打印分类报告
        report = classification_report(all_targets, all_predictions, target_names=classes)
        print(f"\nClassification Report - Epoch {epoch+1}:")
        print(report)
        
        # 打印进度
        elapsed_time = time.time() - start_time
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, '
              f'Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, '
              f'Time: {elapsed_time:.2f}s')
        
        # 保存最佳模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(cnn_model.state_dict(), 
                      os.path.join(checkpoint_dir, 'best_model.pth'))
        
        # 每10个epoch保存检查点和绘制图表
        if (epoch + 1) % 10 == 0:
            # 保存检查点
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': cnn_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
                'train_acc': train_acc,
                'test_acc': test_acc
            }
            torch.save(checkpoint, 
                      os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            
            # 绘制损失和准确率曲线
            plt.figure(figsize=(12, 5))
            
            # 损失曲线
            plt.subplot(1, 2, 1)
            plt.plot(history['train_loss'], label='Train Loss')
            plt.plot(history['test_loss'], label='Test Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Training and Testing Losses')
            
            # 准确率曲线
            plt.subplot(1, 2, 2)
            plt.plot(history['train_acc'], label='Train Acc')
            plt.plot(history['test_acc'], label='Test Acc')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.title('Training and Testing Accuracies')
            
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'metrics_epoch_{epoch+1}.png'))
            plt.close()
    
    # 保存最终模型
    torch.save(cnn_model.state_dict(), 
              os.path.join(checkpoint_dir, 'final_model.pth'))
    
    # 绘制最终的损失和准确率曲线
    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Testing Losses')
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['test_acc'], label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Testing Accuracies')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'final_metrics.png'))
    plt.close()
    
    # 记录总训练时间
    total_time = time.time() - start_time
    print(f'Training completed in {total_time:.2f} seconds')
    
    # 关闭TensorBoard writer
    writer.close()
    
    return history

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    
    # 检查是否可以使用GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 创建数据加载器
    train_dataset = LungXrayDataset(root_dir="/home/lgz/Code/class/ML/e1/covid19", is_train=True)
    test_dataset = LungXrayDataset(root_dir="/home/lgz/Code/class/ML/e1/covid19", is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 加载预训练的autoencoder
    autoencoder = Autoencoder()
    autoencoder.load_state_dict(torch.load('results/checkpoints/best_model.pth'))
    
    # 创建CNN模型
    cnn_model = SimpleCNN()
    
    # 训练CNN模型
    history = train_cnn(
        cnn_model=cnn_model,
        autoencoder=autoencoder,
        lr=1e-3,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=100,
        device=device,
        noise_factor=0.3
    )