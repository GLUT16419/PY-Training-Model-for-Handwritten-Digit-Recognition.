import os
import torchvision.datasets as datasets

def download_official_mnist():
    """下载官方MNIST数据集"""
    print("Downloading official MNIST dataset...")
    
    # 目标目录
    target_dir = 'data/MNIST'
    
    # 下载训练集
    print("Downloading train dataset...")
    train_dataset = datasets.MNIST(
        root=target_dir,
        train=True,
        download=True
    )
    print(f"Train dataset size: {len(train_dataset)}")
    
    # 下载测试集
    print("Downloading test dataset...")
    test_dataset = datasets.MNIST(
        root=target_dir,
        train=False,
        download=True
    )
    print(f"Test dataset size: {len(test_dataset)}")
    
    print("\nMNIST dataset downloaded successfully!")
    print(f"Dataset saved to: {os.path.abspath(target_dir)}")

if __name__ == "__main__":
    download_official_mnist()