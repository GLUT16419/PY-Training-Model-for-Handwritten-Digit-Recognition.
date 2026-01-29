import os
import shutil
import zipfile
import rarfile

def extract_mnist():
    """解压MNIST数据集"""
    print("Extracting MNIST dataset...")
    
    # 源文件路径
    source_dir = '2be96-main/MNIST/MNIST'
    official_rar = os.path.join(source_dir, 'mnist_official.rar')
    dataset_zip = os.path.join(source_dir, 'mnist_dataset.zip')
    
    # 目标目录
    target_dir = 'data/MNIST'
    
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    
    # 尝试解压官方数据集
    if os.path.exists(official_rar):
        print(f"Extracting {official_rar}...")
        try:
            with rarfile.RarFile(official_rar, 'r') as rf:
                rf.extractall(target_dir)
            print("Official MNIST dataset extracted successfully!")
            return
        except Exception as e:
            print(f"Error extracting RAR file: {e}")
    
    # 尝试解压zip数据集
    if os.path.exists(dataset_zip):
        print(f"Extracting {dataset_zip}...")
        try:
            with zipfile.ZipFile(dataset_zip, 'r') as zf:
                zf.extractall(target_dir)
            print("MNIST dataset extracted successfully!")
            return
        except Exception as e:
            print(f"Error extracting ZIP file: {e}")
    
    print("No MNIST dataset files found!")

if __name__ == "__main__":
    extract_mnist()