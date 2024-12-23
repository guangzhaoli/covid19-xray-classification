只训练自编码器：
python main.py \
    --data_dir covid19 \
    --train_autoencoder \
    --ae_epochs 100 \
    --ae_batch_size 32 \
    --ae_lr 0.001

只训练CNN（需要先有训练好的自编码器）：
python main.py \
    --data_dir covid19 \
    --train_cnn \
    --autoencoder_dir results/autoencoder/checkpoints/best_model.pth \
    --cnn_epochs 100 \
    --cnn_batch_size 32 \
    --cnn_lr 0.001 \
    --noise_factor 0.3 

连续训练两个模型：
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

# 查看单个模型的训练过程
tensorboard --logdir results/autoencoder/tensorboard  # 查看自编码器
# 或
tensorboard --logdir results/cnn/tensorboard  # 查看CNN

# 同时查看两个模型的训练过程
tensorboard --logdir_spec autoencoder:results/autoencoder/tensorboard,cnn:results/cnn/tensorboard


# gradio前端
python gradio_app.py \
    --autoencoder_path results/autoencoder/checkpoints/best_model.pth \
    --cnn_path results/cnn/checkpoints/best_model.pth \
    --port 7860


# inference
python inference.py \
    --autoencoder_path results/autoencoder/checkpoints/best_model.pth \
    --cnn_path results/cnn/checkpoints/best_model.pth \
    --image_path test_image.jpeg \
    --config config/config.yaml \
    --device cuda