import os
import urllib.request
import gzip
import shutil

def download_mnist():
    """下载并准备MNIST数据集"""
    # 数据集URL
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]
    
    # 目标目录
    data_dir = 'data/MNIST/MNIST/raw'
    os.makedirs(data_dir, exist_ok=True)
    
    print('Downloading MNIST dataset...')
    
    for file in files:
        file_url = base_url + file
        file_path = os.path.join(data_dir, file)
        extracted_path = os.path.join(data_dir, file.replace('.gz', ''))
        
        # 下载文件
        print(f'Downloading {file}...')
        urllib.request.urlretrieve(file_url, file_path)
        
        # 解压文件
        print(f'Extracting {file}...')
        with gzip.open(file_path, 'rb') as f_in:
            with open(extracted_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # 删除压缩文件
        os.remove(file_path)
    
    print('MNIST dataset downloaded and prepared successfully!')

if __name__ == '__main__':
    download_mnist()