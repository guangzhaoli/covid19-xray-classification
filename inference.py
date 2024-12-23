import torch
from torch.utils.data import DataLoader
import argparse
import os
from data.dataset import LungXrayDataset
from models.autoencoder import Autoencoder
from models.simplecnn import SimpleCNN
from utils import load_config
from torchvision import transforms
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description='COVID-19 X-ray Classification Inference')

    # 模型路径
    parser.add_argument('--autoencoder_path', type=str, required=True,
                        help='Path to the trained autoencoder model')
    parser.add_argument('--cnn_path', type=str, required=True,
                        help='Path to the trained CNN model')

    # 数据路径
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the input X-ray image')

    # 配置路径
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')

    # 设备选项
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for inference (cuda or cpu)')

    return parser.parse_args()

def load_models(args, config, device):
    """加载自编码器和 CNN 模型"""
    autoencoder = Autoencoder().to(device)
    cnn_model = SimpleCNN().to(device) 

    autoencoder.load_state_dict(torch.load(args.autoencoder_path, map_location=device))
    cnn_model.load_state_dict(torch.load(args.cnn_path, map_location=device))

    autoencoder.eval()
    cnn_model.eval()

    return autoencoder, cnn_model

def preprocess_image(image_path, config):
    """预处理输入图像"""
    img = Image.open(image_path).convert('L')  # 转换为灰度图像

    preprocess_config = config['data']['preprocess']
    transform_list = [
        transforms.Resize(preprocess_config['resize_size']),
        transforms.ToTensor(),
    ]
    if preprocess_config.get('normalize', False):
        transform_list.append(transforms.Normalize(
            mean=preprocess_config['mean'],
            std=preprocess_config['std']
        ))
    transform = transforms.Compose(transform_list)
    
    img_tensor = transform(img).unsqueeze(0) # 添加 batch 维度
    return img_tensor

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    config = load_config(args.config)

    # 加载模型
    autoencoder, cnn_model = load_models(args, config, device)

    # 预处理图像
    input_tensor = preprocess_image(args.image_path, config).to(device)

    with torch.no_grad():
        # 通过自编码器去噪
        denoised_image = autoencoder(input_tensor)

        # 通过 CNN 进行分类
        output = cnn_model(denoised_image)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    # 定义类别标签 (需要与你的训练数据集一致)
    class_names = ['Covid', 'Normal', 'Viral Pneumonia']  # 示例类别
    
    print(f"Probabilities: {probabilities.cpu().numpy()[0]}")
    print(f"Predicted class: {class_names[predicted_class]}")


if __name__ == "__main__":
    main()