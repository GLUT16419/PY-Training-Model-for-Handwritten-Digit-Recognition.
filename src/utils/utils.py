import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def get_device():
    """
    获取可用的设备
    Returns:
        device: 设备实例
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_model(model, path):
    """
    保存模型
    Args:
        model: 模型实例
        path: 保存路径
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f'Model saved to: {path}')

def load_model(model, path):
    """
    加载模型
    Args:
        model: 模型实例
        path: 模型路径
    Returns:
        model: 加载了权重的模型实例
    """
    model.load_state_dict(torch.load(path, map_location=get_device()))
    model.eval()
    print(f'Model loaded from: {path}')
    return model

def plot_training_history(history, save_path=None):
    """
    绘制训练历史
    Args:
        history: 训练历史字典
        save_path: 保存路径
    """
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f'Training history plot saved to: {save_path}')
    
    plt.close()

def visualize_mnist_sample(data_loader, num_samples=5):
    """
    可视化MNIST样本
    Args:
        data_loader: 数据加载器
        num_samples: 样本数量
    """
    # 获取一个批次的数据
    for batch in data_loader:
        images, labels = batch
        break
    
    plt.figure(figsize=(10, 2))
    
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(images[i].squeeze().numpy(), cmap='gray')
        plt.title(f'Label: {labels[i].item()}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()