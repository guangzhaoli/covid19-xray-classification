import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from models.autoencoder import Autoencoder
from models.simplecnn import SimpleCNN

class CovidClassifier:
    def __init__(self, autoencoder_path, cnn_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        self.autoencoder = self._load_model(Autoencoder(), autoencoder_path)
        self.cnn = self._load_model(SimpleCNN(), cnn_path)
        
        # 定义图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
        
        # 类别标签
        self.classes = ['Covid-19', 'Normal', 'Viral Pneumonia']
        
    def _load_model(self, model, path):
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def add_noise(self, image, noise_factor=0.3):
        noisy = image + noise_factor * torch.randn_like(image)
        return torch.clamp(noisy, 0., 1.)

    def predict(self, image, noise_factor=0.3):
        if isinstance(image, str):
            image = Image.open(image).convert('L')
        image = self.transform(image).unsqueeze(0)
        image = image.to(self.device)
        
        noisy_image = self.add_noise(image, noise_factor)
        
        with torch.no_grad():
            denoised = self.autoencoder(noisy_image)
            outputs = self.cnn(denoised)
            probs = F.softmax(outputs, dim=1)
        
        probs = probs.cpu().numpy()[0]
        
        return {
            'Original': self._prepare_image_for_display(image),
            'Noisy': self._prepare_image_for_display(noisy_image),
            'Denoised': self._prepare_image_for_display(denoised),
            'Prediction': {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}
        }
    
    def _prepare_image_for_display(self, tensor):
        return tensor.cpu().squeeze().numpy()

def create_interface(autoencoder_path, cnn_path):
    classifier = CovidClassifier(autoencoder_path, cnn_path)
    
    def process_image(image, noise_factor):
        if image is None:
            return None, None, None, "Please upload an image."
        
        try:
            results = classifier.predict(image, noise_factor)
            
            # 创建预测结果字符串，包含概率条形图
            pred_html = "<div style='text-align: left; padding: 10px;'>"
            for cls, prob in results['Prediction'].items():
                bar_length = int(prob * 100)
                pred_html += f"<div style='margin: 5px 0;'>"
                pred_html += f"<div style='font-weight: bold;'>{cls}: {prob:.2%}</div>"
                pred_html += f"<div style='background: #ddd; height: 20px; border-radius: 5px;'>"
                pred_html += f"<div style='background: {'#4CAF50' if prob == max(results['Prediction'].values()) else '#2196F3'}; "
                pred_html += f"width: {bar_length}%; height: 100%; border-radius: 5px;'></div></div></div>"
            pred_html += "</div>"
            
            return (
                results['Original'],
                results['Noisy'],
                results['Denoised'],
                pred_html
            )
        except Exception as e:
            return None, None, None, f"Error processing image: {str(e)}"

    # 创建Gradio界面
    with gr.Blocks(theme=gr.themes.Soft()) as iface:
        gr.Markdown(
            """
            # COVID-19 X-ray Classification System
            
            This system uses a deep learning pipeline to classify chest X-ray images:
            1. First, it processes the image through a denoising autoencoder
            2. Then, it uses a CNN to classify the image into one of three categories
            
            Upload your chest X-ray image to get started!
            """
        )
        
        with gr.Row():
            with gr.Column():
                # 修改这里，移除 tool 参数
                input_image = gr.Image(
                    type="pil",
                    label="Upload X-ray Image",
                    sources=["upload", "clipboard"]  # 使用 sources 替代 tool
                )
                noise_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.3,
                    step=0.1,
                    label="Noise Factor"
                )
                submit_btn = gr.Button("Analyze", variant="primary")
            
            with gr.Column():
                output_original = gr.Image(label="Original Image")
                output_noisy = gr.Image(label="Noisy Image")
                output_denoised = gr.Image(label="Denoised Image")
                output_prediction = gr.HTML(label="Prediction Results")

        # 设置示例
        gr.Examples(
            examples=[
                ["examples/covid.jpeg", 0.3],
                ["examples/normal.jpeg", 0.3],
                ["examples/pneumonia.jpeg", 0.3]
            ],
            inputs=[input_image, noise_slider],
            outputs=[output_original, output_noisy, output_denoised, output_prediction],
            fn=process_image,
            cache_examples=True
        )
        
        submit_btn.click(
            fn=process_image,
            inputs=[input_image, noise_slider],
            outputs=[output_original, output_noisy, output_denoised, output_prediction]
        )
        
    return iface

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='COVID-19 X-ray Classification Demo')
    parser.add_argument('--autoencoder_path', type=str, required=True,
                        help='Path to trained autoencoder model')
    parser.add_argument('--cnn_path', type=str, required=True,
                        help='Path to trained CNN model')
    parser.add_argument('--port', type=int, default=7860,
                        help='Port to run the interface on')
    parser.add_argument('--examples_dir', type=str, default='examples',
                        help='Directory containing example images')
    
    args = parser.parse_args()
    
    # 确保示例目录存在
    os.makedirs(args.examples_dir, exist_ok=True)
    
    # 创建并启动界面
    interface = create_interface(args.autoencoder_path, args.cnn_path)
    interface.launch(
        server_port=args.port,
        share=True,
        server_name="0.0.0.0"
    )