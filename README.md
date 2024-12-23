# ⚕️ COVID-19 X 光片智能分类系统


**一个基于深度学习的双阶段 COVID-19 胸部 X 光片分类解决方案，采用去噪自编码器和卷积神经网络技术。**

此项目旨在为医疗专业人员提供一个高效且准确的工具，用于辅助诊断 COVID-19、普通疾病以及病毒性肺炎。通过结合自编码器的图像增强能力和 CNN 的强大分类性能，该系统力求在医学影像分析领域提供卓越的性能。

## 📂 项目结构

```
.
├── config/
│   └── config.yaml         # ⚙️ 配置文件
├── data/
│   └── dataset.py         # 💾 数据集加载类
├── models/
│   ├── autoencoder.py     # 🧠 自编码器模型
│   └── simplecnn.py       # 🔬 CNN分类模型
├── results/
│   ├── autoencoder/       # 📊 自编码器训练结果
│   └── cnn/              # 📈 CNN训练结果
├── examples/             # 🖼️ 示例图片
├── utils.py             # 🛠️ 工具函数
├── main.py              # 🚀 训练主脚本
├── train_autoencoder.py # ⚙️ 自编码器训练代码
├── train_cnn.py        # ⚙️ CNN训练代码
├── inference.py        # 🧪 模型推理脚本
├── gradio_app.py       # 🌐 Web界面应用
└── requirements.txt    # 📦 项目依赖
```

## ✨ 功能特点

1. **🚀 双阶段智能处理流程**：
   - **图像去噪增强**：利用自编码器有效去除 X 光片中的噪声，提升图像质量。
   - **精准分类**：采用卷积神经网络对去噪后的图像进行分类，区分 Covid-19、Normal 和 Viral Pneumonia 三种状态。

2. **⚙️ 灵活的训练配置**：
   - **模块化训练**：支持独立训练自编码器或 CNN 模型，满足不同需求。
   - **串行训练**：支持先训练自编码器，再将其应用于 CNN 的训练流程。
   - **参数可定制**：为每个模型提供独立的训练参数配置，精细化控制训练过程。

3. **📊 可视化与监控**：
   - **TensorBoard 集成**：实时跟踪训练过程中的各项指标，如损失和准确率。
   - **性能可视化**：清晰展示模型在训练和验证集上的性能表现。
   - **混淆矩阵分析**：直观展示模型分类结果，便于误差分析和模型优化。

4. **🌐 用户友好的 Web 界面**：
   - **交互式操作**：通过简单的 Web 界面上传 X 光片进行预测。
   - **实时处理反馈**：展示图像处理和预测的中间过程。
   - **清晰结果呈现**：直观显示预测结果及其置信度。

## 🛠️ 安装指南

1. **克隆仓库**：

   ```bash
   git clone https://github.com/yourusername/covid19-xray-classification.git
   cd covid19-xray-classification
   ```

2. **创建并激活虚拟环境**：

   ```bash
   conda create -n covid python=3.10
   conda activate covid
   ```

3. **安装依赖**：

   ```bash
   pip install -r requirements.txt
   ```

## 🎬 使用方法

## 🗂️ 数据集要求

为了保证模型的有效训练，请确保数据集目录结构如下：

```
dataset/
├── train/
│   ├── Covid/
│   ├── Normal/
│   └── Viral Pneumonia/
└── noisy_test/
    ├── Covid/
    ├── Normal/
    └── Viral Pneumonia/
```

每个子文件夹中包含对应类别的胸部 X 光片图像。训练完毕的模型默认存于**result/{autoencoder, cnn}/checkpoints/best_model.pth**中

### 1. 🏋️‍♀️ 训练模型

**单独训练自编码器**：

```bash
python main.py \
    --data_dir /path/to/your/dataset \
    --train_autoencoder \
    --ae_epochs 100 \
    --ae_batch_size 32 \
    --ae_lr 0.001
```

**单独训练 CNN**：

```bash
python main.py \
    --data_dir /path/to/your/dataset \
    --train_cnn \
    --autoencoder_dir /path/to/your/ae_model.pth \
    --cnn_epochs 100 \
    --cnn_batch_size 32 \
    --cnn_lr 0.001 \
    --noise_factor 0.3 
```

**连续训练从头两个模型**：

```bash
python main.py \
    --train_autoencoder \
    --train_cnn \
    --ae_epochs 100 \
    --ae_batch_size 32 \
    --ae_lr 0.001 \
    --cnn_epochs 50 \
    --cnn_batch_size 64 \
    --cnn_lr 0.0001 \
    --noise_factor 0.3
```

### 2. 🌐 启动 Web 界面

```bash
python gradio_app.py \
    --autoencoder_path /path/to/your/ae_pretrained.pth \
    --cnn_path /path/to/your/cnn_pretrained.pth \
    --port 7860
```

访问 `http://localhost:7860` 在浏览器中使用 Web 界面。

**或使用命令行参数进行推理**

```bash
python inference.py \
    --autoencoder_path /path/to/your/ae_pretrained.pth \
    --cnn_path /path/to/your/cnn_pretrained.pth \
    --image_path test_image.jpeg \
    --config config/config.yaml \
    --device cuda
```

### 3. 📊 查看训练过程

```bash
tensorboard --logdir results/autoencoder/tensorboard  # 查看自编码器

tensorboard --logdir results/cnn/tensorboard  # 查看CNN

tensorboard --logdir_spec autoencoder:results/autoencoder/tensorboard,cnn:results/cnn/tensorboard # 同时查看两个训练过程
```

## ⚙️ 配置文件

可以在 `config/config.yaml` 文件中自定义训练参数，例如：

```yaml
training:
  autoencoder:
    num_epochs: 100
    batch_size: 32
    learning_rate: 0.001
  cnn:
    num_epochs: 50
    batch_size: 64
    learning_rate: 0.0001
```

一个可能的示例: 将数据集放入与main.py同级下:
```bash
project/
├── covid19  # 数据集文件夹
    ├── train/
        ├── Covid/
        ├── Normal/
        └── Viral Pneumonia/
    └── noisy_test/
        ├── Covid/
        ├── Normal/
        └── Viral Pneumonia/
├── main.py
├── inference.py
├── ...

# 只训练AutoEncoder
python main.py \
    --data_dir covid19 \
    --train_autoencoder \
    --ae_epochs 100 \
    --ae_batch_size 32 \
    --ae_lr 0.001

# 只训练CNN（需要先有训练好的自编码器）：
python main.py \
    --data_dir covid19 \
    --train_cnn \
    --autoencoder_dir results/autoencoder/checkpoints/best_model.pth \
    --cnn_epochs 100 \
    --cnn_batch_size 32 \
    --cnn_lr 0.001 \
    --noise_factor 0.3 

# 连续训练两个模型：
python main.py \
    --train_autoencoder \
    --train_cnn \
    --ae_epochs 100 \
    --ae_batch_size 32 \
    --ae_lr 0.001 \
    --cnn_epochs 50 \
    --cnn_batch_size 64 \
    --cnn_lr 0.0001 \
    --noise_factor 0.3

# 运行Gradio前端
python gradio_app.py \
    --autoencoder_path results/autoencoder/checkpoints/best_model.pth \
    --cnn_path results/cnn/checkpoints/best_model.pth \
    --port 7860
```

## 🧠 模型架构

1. **去噪自编码器**：
   - **编码器**：利用卷积层提取输入图像的深层特征。
   - **解码器**：通过反卷积层重建去噪后的图像，学习有效的特征表示。
   - **目标**：减少图像噪声，提升后续 CNN 模型的分类性能。

2. **卷积神经网络分类器**：
   - **多层卷积层**：提取图像中的局部和全局特征。
   - **全连接层**：将提取的特征映射到类别概率。
   - **输出层**：输出 Covid-19、Normal 和 Viral Pneumonia 三个类别的预测概率。

## 📈 结果展示

我们将在 `results/` 目录下存储训练结果，包括：

1. **分类准确率**：评估模型在测试集上的整体分类性能。
2. **混淆矩阵**：可视化模型在不同类别上的分类情况，分析错误类型。
3. **ROC 曲线**：评估模型在不同阈值下的真阳性率和假阳性率。
4. **示例预测结果**：展示模型在一些典型样本上的预测效果。

## ⚠️ 注意事项

1. **硬件要求**：建议使用配备 GPU 的环境进行训练，以加速计算过程。
2. **数据预处理**：确保输入模型的图像已经过适当的标准化处理。
3. **超参数调优**：使用验证集进行超参数调整，以获得最佳模型性能。
4. **模型保存**：定期保存模型检查点，以便在训练中断后恢复或用于后续部署。

## 🤝 贡献指南

我们欢迎社区的贡献！如果您想为此项目做出贡献，请遵循以下步骤：

1. **Fork 仓库**：在 GitHub 上 Fork 该仓库。
2. **创建新分支**：`git checkout -b feature/your-feature`
3. **提交更改**：`git commit -m 'Add your feature'`
4. **推送分支**：`git push origin feature/your-feature`
5. **发起 Pull Request**：提交您的 Pull Request 以供审核。

## 📄 许可证

本项目采用 [MIT License](LICENSE)。

## 📧 联系方式

- **作者**：[Guangzhao Li]
- **邮箱**：[gzhao.cs@gmail.com]
- **GitHub**：[@Your GitHub Profile](https://github.com/guangzhaoli)

---

**让我们一起利用 AI 技术，为抗击疫情贡献力量！**