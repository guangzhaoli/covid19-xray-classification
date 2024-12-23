import yaml
import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import os

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_reconstructed_images(original, reconstructed, path, nrow=8):
    """保存重建图像对比"""
    comparison = torch.cat([original[:nrow], reconstructed[:nrow]])
    save_image(comparison.cpu(), path, nrow=nrow)

def plot_losses(train_losses, test_losses, save_path):
    """绘制损失曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()