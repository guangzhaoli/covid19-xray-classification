import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import os
import matplotlib.pyplot as plt
from data.dataset import LungXrayDataset
from models.autoencoder import Autoencoder
from tqdm import tqdm
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter


def train_autoencoder(model, lr, train_loader, test_loader, num_epochs=100, device='cuda', output_dir='results_autoencoder'):
    """
    训练自编码器, 使用TensorBoard进行可视化
    
    Args:
        model: 自编码器模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        num_epochs: 训练轮数
        device: 使用的设备 ('cuda' 或 'cpu')
        output_dir: 输出目录
    """
    # 创建输出目录
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    tensorboard_dir = os.path.join(output_dir, 'tensorboard')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    # 初始化TensorBoard writer
    writer = SummaryWriter(tensorboard_dir)
    
    # 将模型移至指定设备
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)# 记录最佳模型
    best_test_loss = float('inf')
    
    # 获取一批固定的测试数据用于可视化
    fixed_test_data, _ = next(iter(test_loader))
    fixed_test_data = fixed_test_data.to(device)
    
    # 添加模型图到TensorBoard
    writer.add_graph(model, fixed_test_data)
    
    # 训练开始时间
    start_time = time.time()
    global_step = 0
    
    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}] Training')
        
        for batch_idx, (data, _) in enumerate(train_pbar):
            data = data.to(device)
            
            # 前向传播
            output = model(data)
            loss = criterion(output, data)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录损失
            train_loss += loss.item()
            writer.add_scalar('Loss/train_step', loss.item(), global_step)
            
            # 更新进度条
            train_pbar.set_postfix({'loss': loss.item()})
            global_step += 1
        
        # 计算平均训练损失
        train_loss /= len(train_loader)
        
        # 测试阶段
        model.eval()
        test_loss = 0
        test_pbar = tqdm(test_loader, desc=f'Epoch [{epoch+1}/{num_epochs}] Testing')
        
        with torch.no_grad():
            for data, _ in test_pbar:
                data = data.to(device)
                output = model(data)
                loss = criterion(output, data)
                test_loss += loss.item()
                test_pbar.set_postfix({'loss': loss.item()})
        
        test_loss /= len(test_loader)
        
        # 记录每个epoch的损失
        writer.add_scalars('Loss/epoch', {
            'train': train_loss,
            'test': test_loss
        }, epoch)
        
        # 可视化重建结果
        with torch.no_grad():
            reconstructed = model(fixed_test_data)
            # 创建原始图像和重建图像的对比网格
            comparison = torch.cat([fixed_test_data[:8], reconstructed[:8]])
            grid = make_grid(comparison, nrow=8, normalize=True)
            writer.add_image('Reconstruction', grid, epoch)
        
        # 记录模型参数分布
        for name, param in model.named_parameters():
            writer.add_histogram(f'Parameters/{name}', param, epoch)
            if param.grad is not None:
                writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
        
        # 打印进度
        elapsed_time = time.time() - start_time
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.6f}, '
              f'Test Loss: {test_loss:.6f}, '
              f'Time: {elapsed_time:.2f}s')
        
        # 保存最佳模型
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), 
                      os.path.join(checkpoint_dir, 'best_model.pth'))
        
        # 每10个epoch保存一次检查点
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
            }
            torch.save(checkpoint, 
                      os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # 保存最终模型
    torch.save(model.state_dict(), 
              os.path.join(checkpoint_dir, 'final_model.pth'))
    
    # 记录总训练时间
    total_time = time.time() - start_time
    print(f'Training completed in {total_time:.2f} seconds')
    
    # 关闭TensorBoard writer
    writer.close()


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
    
    # 创建模型
    model = Autoencoder()# 训练模型
    train_losses, test_losses = train_autoencoder(
        model=model,
        lr=1e-3,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=100,
        device=device
    )