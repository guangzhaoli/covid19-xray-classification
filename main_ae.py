import os
import torch
from torch.utils.data import DataLoader
import argparse
from data.dataset import LungXrayDataset
from models.autoencoder import Autoencoder
from utils import load_config
from train_autoencoder import train_autoencoder
def parse_args():
    parser = argparse.ArgumentParser(description='COVID-19 X-ray Denoising Project')
    
    # 基础参数
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Path to output directory')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs (override config file)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (override config file)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (override config file)')
    
    # 设备选项
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], 
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    return parser.parse_args()
def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # 加载配置
    config = load_config(args.config)
    
    # 命令行参数覆盖配置文件
    if args.epochs is not None:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    
    # 设置设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('Warning: CUDA is not available, using CPU instead')
        device = 'cpu'
    else:
        device = args.device
    device = torch.device(device)
    print(f'Using device: {device}')
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)
    
    # 创建数据加载器
    train_dataset = LungXrayDataset(
        root_dir=args.data_dir,
        is_train=True
    )
    
    test_dataset = LungXrayDataset(
        root_dir=args.data_dir,
        is_train=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    # 创建模型
    model = Autoencoder()
    
    # 如果指定了恢复训练的检查点
    if args.resume:
        print(f'Loading checkpoint from {args.resume}')
        model.load_state_dict(torch.load(args.resume, map_location=device))
    
    # 训练模型
    train_losses, test_losses = train_autoencoder(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=config['training']['num_epochs'],
        device=device,
        output_dir=args.output_dir
    )
if __name__ == "__main__":
    main()