import os
import sys
import matplotlib.pyplot as plt
import torch

# 添加项目根目录到Python搜索路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import MNISTDataProcessor

def visualize_data():
    """可视化MNIST数据集"""
    print("Visualizing MNIST data...")
    
    # 创建数据处理器
    processor = MNISTDataProcessor()
    
    # 加载数据
    train_loader, val_loader, test_loader = processor.load_data()
    
    # 获取一个批次的数据
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels: {labels[:10]}")
        
        # 可视化前5个图像
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        for i in range(5):
            img = images[i].squeeze().numpy()
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"Label: {labels[i].item()}")
            axes[i].axis('off')
        
        # 保存图像
        os.makedirs('reports', exist_ok=True)
        plt.savefig('reports/mnist_sample.png')
        print("Saved sample images to reports/mnist_sample.png")
        
        # 只显示第一个批次
        break
    
    print("\nData visualization completed!")

if __name__ == "__main__":
    visualize_data()