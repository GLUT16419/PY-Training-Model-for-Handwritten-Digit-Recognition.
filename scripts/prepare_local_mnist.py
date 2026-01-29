import os
import zipfile
import numpy as np
import struct
from PIL import Image

def prepare_local_mnist():
    """使用本地mnist_dataset目录中的文件准备MNIST数据集"""
    # 本地数据集目录
    local_dataset_dir = 'data/MNIST/mnist_dataset'
    # 目标目录
    target_dir = 'data/MNIST/MNIST/raw'
    os.makedirs(target_dir, exist_ok=True)
    
    print('Preparing MNIST dataset from local files...')
    
    # 检查本地文件是否存在
    if not os.path.exists(os.path.join(local_dataset_dir, 'train.zip')):
        print('Error: train.zip not found in mnist_dataset directory!')
        return
    
    if not os.path.exists(os.path.join(local_dataset_dir, 'train_labs.txt')):
        print('Error: train_labs.txt not found in mnist_dataset directory!')
        return
    
    if not os.path.exists(os.path.join(local_dataset_dir, 'test.zip')):
        print('Error: test.zip not found in mnist_dataset directory!')
        return
    
    if not os.path.exists(os.path.join(local_dataset_dir, 'test_labs.txt')):
        print('Error: test_labs.txt not found in mnist_dataset directory!')
        return
    
    # 处理训练数据
    print('Processing training data...')
    process_dataset(
        os.path.join(local_dataset_dir, 'train.zip'),
        os.path.join(local_dataset_dir, 'train_labs.txt'),
        os.path.join(target_dir, 'train-images-idx3-ubyte'),
        os.path.join(target_dir, 'train-labels-idx1-ubyte')
    )
    
    # 处理测试数据
    print('Processing test data...')
    process_dataset(
        os.path.join(local_dataset_dir, 'test.zip'),
        os.path.join(local_dataset_dir, 'test_labs.txt'),
        os.path.join(target_dir, 't10k-images-idx3-ubyte'),
        os.path.join(target_dir, 't10k-labels-idx1-ubyte')
    )
    
    print('MNIST dataset prepared successfully from local files!')

def process_dataset(images_zip, labels_txt, output_images, output_labels):
    """处理单个数据集"""
    # 读取标签
    with open(labels_txt, 'r') as f:
        labels = []
        for line in f:
            # 处理 '0    7' 这种格式，取最后一个数字
            parts = line.strip().split()
            if parts:
                labels.append(int(parts[-1]))
    
    # 读取图像
    images = []
    with zipfile.ZipFile(images_zip, 'r') as zip_ref:
        # 获取所有图像文件
        image_files = [name for name in zip_ref.namelist() if name.endswith('.png') or name.endswith('.jpg')]
        image_files.sort()  # 确保顺序正确
        
        for image_file in image_files:
            with zip_ref.open(image_file) as img_file:
                img = Image.open(img_file).convert('L')  # 转换为灰度图
                img_array = np.array(img, dtype=np.uint8)
                images.append(img_array)
    
    # 确保图像和标签数量匹配
    if len(images) != len(labels):
        print(f'Error: Number of images ({len(images)}) does not match number of labels ({len(labels)})!')
        return
    
    # 写入图像文件（IDX格式）
    with open(output_images, 'wb') as f:
        # 写入魔术数字
        f.write(struct.pack('>I', 2051))
        # 写入图像数量
        f.write(struct.pack('>I', len(images)))
        # 写入高度
        f.write(struct.pack('>I', 28))
        # 写入宽度
        f.write(struct.pack('>I', 28))
        # 写入图像数据
        for img in images:
            f.write(img.tobytes())
    
    # 写入标签文件（IDX格式）
    with open(output_labels, 'wb') as f:
        # 写入魔术数字
        f.write(struct.pack('>I', 2049))
        # 写入标签数量
        f.write(struct.pack('>I', len(labels)))
        # 写入标签数据
        for label in labels:
            f.write(struct.pack('>B', label))

if __name__ == '__main__':
    prepare_local_mnist()