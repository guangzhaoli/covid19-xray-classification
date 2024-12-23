import os
import torch
from torch.utils.data import DataLoader
import argparse
from data.dataset import LungXrayDataset
from models.autoencoder import Autoencoder
from models.simplecnn import SimpleCNN
from utils import load_config
from train_autoencoder import train_autoencoder
from train_cnn import train_cnn

def parse_args():
    parser = argparse.ArgumentParser(description='COVID-19 X-ray Classification Project')
    
    # 基础参数
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to data directory')
    
    # 训练阶段选择
    parser.add_argument('--train_autoencoder', action='store_true',
                        help='Train autoencoder model')
    parser.add_argument('--train_cnn', action='store_true',
                        help='Train CNN model')
    
    # 输出目录
    parser.add_argument('--autoencoder_dir', type=str, default='results/autoencoder',
                        help='Output directory for autoencoder')
    parser.add_argument('--cnn_dir', type=str, default='results/cnn',
                        help='Output directory for CNN')
    
    # 自编码器训练参数
    parser.add_argument('--ae_epochs', type=int, default=None,
                        help='Number of epochs for autoencoder')
    parser.add_argument('--ae_batch_size', type=int, default=None,
                        help='Batch size for autoencoder')
    parser.add_argument('--ae_lr', type=float, default=None,
                        help='Learning rate for autoencoder')
    
    # CNN训练参数
    parser.add_argument('--cnn_epochs', type=int, default=None,
                        help='Number of epochs for CNN')
    parser.add_argument('--cnn_batch_size', type=int, default=None,
                        help='Batch size for CNN')
    parser.add_argument('--cnn_lr', type=float, default=None,
                        help='Learning rate for CNN')
    parser.add_argument('--noise_factor', type=float, default=0.3,
                        help='Noise factor for data augmentation')
    
    # 设备选项
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], 
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # 模型加载
    parser.add_argument('--resume_autoencoder', type=str, default=None,
                        help='Path to autoencoder checkpoint to resume from')
    parser.add_argument('--resume_cnn', type=str, default=None,
                        help='Path to CNN checkpoint to resume from')
    
    return parser.parse_args()


def train_phase_autoencoder(args, config, device, train_loader, test_loader):
    """自编码器训练阶段"""
    print("=== Starting Autoencoder Training ===")
    
    # 创建自编码器输出目录
    os.makedirs(args.autoencoder_dir, exist_ok=True)
    os.makedirs(os.path.join(args.autoencoder_dir, 'checkpoints'), exist_ok=True)
    
    # 创建模型
    autoencoder = Autoencoder()
    
    # 如果指定了恢复训练的检查点
    if args.resume_autoencoder:
        print(f'Loading autoencoder checkpoint from {args.resume_autoencoder}')
        autoencoder.load_state_dict(torch.load(args.resume_autoencoder, map_location=device))
    
    # 训练自编码器
    autoencoder_history = train_autoencoder(
        model=autoencoder,
        lr=config['training']['learning_rate'],
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=config['training']['num_epochs'],
        device=device,
        output_dir=args.autoencoder_dir
    )
    
    return autoencoder

def train_phase_cnn(args, config, device, train_loader, test_loader, autoencoder):
    """CNN训练阶段"""
    print("=== Starting CNN Training ===")
    
    # 创建CNN输出目录
    os.makedirs(args.cnn_dir, exist_ok=True)
    os.makedirs(os.path.join(args.cnn_dir, 'checkpoints'), exist_ok=True)
    
    # 创建CNN模型
    cnn_model = SimpleCNN()
    
    # 如果指定了恢复训练的检查点
    if args.resume_cnn:
        print(f'Loading CNN checkpoint from {args.resume_cnn}')
        cnn_model.load_state_dict(torch.load(args.resume_cnn, map_location=device))
    
    # 训练CNN
    cnn_history = train_cnn(
        cnn_model=cnn_model,
        autoencoder=autoencoder,
        lr=config['training']['learning_rate'],
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=config['training']['num_epochs'],
        device=device,
        output_dir=args.cnn_dir,
        noise_factor=args.noise_factor
    )
    
    return cnn_history

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建训练配置副本
    ae_config = config.copy()
    cnn_config = config.copy()
    
    # 命令行参数覆盖配置文件 - 自编码器
    if args.ae_epochs is not None:
        ae_config['training']['num_epochs'] = args.ae_epochs
    if args.ae_batch_size is not None:
        ae_config['training']['batch_size'] = args.ae_batch_size
    if args.ae_lr is not None:
        ae_config['training']['learning_rate'] = args.ae_lr
        
    # 命令行参数覆盖配置文件 - CNN
    if args.cnn_epochs is not None:
        cnn_config['training']['num_epochs'] = args.cnn_epochs
    if args.cnn_batch_size is not None:
        cnn_config['training']['batch_size'] = args.cnn_batch_size
    if args.cnn_lr is not None:
        cnn_config['training']['learning_rate'] = args.cnn_lr
    
    # 设置设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('Warning: CUDA is not available, using CPU instead')
        device = 'cpu'
    else:
        device = args.device
    device = torch.device(device)
    print(f'Using device: {device}')
    
    # 创建数据加载器 - 自编码器
    if args.train_autoencoder:
        train_dataset_ae = LungXrayDataset(
            root_dir=args.data_dir,
            is_train=True
        )
        test_dataset_ae = LungXrayDataset(
            root_dir=args.data_dir,
            is_train=False
        )
        
        train_loader_ae = DataLoader(
            train_dataset_ae,
            batch_size=ae_config['training']['batch_size'],
            shuffle=True
        )
        test_loader_ae = DataLoader(
            test_dataset_ae,
            batch_size=ae_config['training']['batch_size'],
            shuffle=False
        )
    
    # 创建数据加载器 - CNN
    if args.train_cnn:
        train_dataset_cnn = LungXrayDataset(
            root_dir=args.data_dir,
            is_train=True
        )
        test_dataset_cnn = LungXrayDataset(
            root_dir=args.data_dir,
            is_train=False
        )
        
        train_loader_cnn = DataLoader(
            train_dataset_cnn,
            batch_size=cnn_config['training']['batch_size'],
            shuffle=True
        )
        test_loader_cnn = DataLoader(
            test_dataset_cnn,
            batch_size=cnn_config['training']['batch_size'],
            shuffle=False
        )
    
    # 训练自编码器
    if args.train_autoencoder:
        print("\n=== Autoencoder Training Configuration ===")
        print(f"Epochs: {ae_config['training']['num_epochs']}")
        print(f"Batch Size: {ae_config['training']['batch_size']}")
        print(f"Learning Rate: {ae_config['training']['learning_rate']}\n")
        
        autoencoder = train_phase_autoencoder(args, ae_config, device, 
                                            train_loader_ae, test_loader_ae)
    else:
        # 如果不训练自编码器，则加载预训练的模型
        autoencoder = Autoencoder()
        autoencoder_path = args.autoencoder_dir
        if os.path.exists(autoencoder_path):
            print(f'Loading pretrained autoencoder from {autoencoder_path}')
            autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=device))
        else:
            raise FileNotFoundError(f"No pretrained autoencoder found at {autoencoder_path}")
    
    # 训练CNN
    if args.train_cnn:
        print("\n=== CNN Training Configuration ===")
        print(f"Epochs: {cnn_config['training']['num_epochs']}")
        print(f"Batch Size: {cnn_config['training']['batch_size']}")
        print(f"Learning Rate: {cnn_config['training']['learning_rate']}")
        print(f"Noise Factor: {args.noise_factor}\n")
        
        autoencoder.eval()  # 设置自编码器为评估模式
        train_phase_cnn(args, cnn_config, device, train_loader_cnn, 
                       test_loader_cnn, autoencoder)

if __name__ == "__main__":
    main()