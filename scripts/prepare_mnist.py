import os
import shutil
import gzip

def prepare_mnist():
    """准备MNIST数据集，确保文件在正确的位置"""
    print("Preparing MNIST dataset...")
    
    # 源目录
    source_dir = 'data/MNIST/mnist_official'
    # 目标目录
    target_dir = 'data/MNIST/MNIST/raw'
    
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    
    # 需要的文件
    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]
    
    # 复制并解压文件
    for file in files:
        source_path = os.path.join(source_dir, file)
        target_path = os.path.join(target_dir, file)
        
        if os.path.exists(source_path):
            print(f"Processing {file}...")
            
            # 复制文件
            shutil.copy2(source_path, target_path)
            
            # 解压文件
            with gzip.open(target_path, 'rb') as f_in:
                with open(target_path[:-3], 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # 删除压缩文件
            os.remove(target_path)
            print(f"Extracted {file} to {target_path[:-3]}")
        else:
            print(f"Warning: {file} not found in {source_dir}")
    
    print("MNIST dataset prepared successfully!")

if __name__ == "__main__":
    prepare_mnist()