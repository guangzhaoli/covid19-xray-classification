import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image

class LungXrayDataset(Dataset):
    def __init__(self, root_dir, is_train=True, transform=None):
        """
        参数:
            root_dir (str): 数据集根目录
            is_train (bool): 是否为训练集
            transform (callable, optional): 可选的图像预处理
        """
        self.root_dir = root_dir
        self.is_train = is_train
        self.transform = transform
        self.classes = ['Covid', 'Normal', 'Viral Pneumonia']
        
        # 设置基础图像变换
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),     # 调整图像大小
                transforms.Grayscale(num_output_channels=1),  # 转换为灰度图
                transforms.ToTensor(),             # 转换为tensor
                transforms.Normalize(mean=[0.5], std=[0.5])  # 针对灰度图的标准化
            ])
        
        # 收集数据路径和标签
        self.data_info = []
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, 'train' if is_train else 'noisy_test',class_name)
            for img_name in os.listdir(class_path):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.data_info.append({
                        'path': os.path.join(class_path, img_name),
                        'label': class_idx
                    })
                    
    def add_gaussian_noise(self, image):
        """添加高斯噪声"""
        noise = torch.randn_like(image) * 0.1  # 0.1是噪声强度，可以调整
        noisy_image = image + noise
        return torch.clamp(noisy_image, 0, 1)
    
    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, idx):
        img_path = self.data_info[idx]['path']
        label = self.data_info[idx]['label']
        
        # 加载图像
        image = Image.open(img_path).convert('L')  # 直接以灰度图方式加载
        image = self.transform(image)
        
        # 如果是训练集，添加高斯噪声
        if self.is_train:
            image = self.add_gaussian_noise(image)
        return image, label

        
if __name__ == "__main__":
    train_dataset = LungXrayDataset(root_dir = "/home/lgz/Code/class/ML/e1/covid19", is_train = False)
    
    print(train_dataset[0][0].shape)