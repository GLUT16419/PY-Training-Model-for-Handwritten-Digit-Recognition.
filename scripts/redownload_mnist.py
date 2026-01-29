import os
import shutil
import torchvision.datasets as datasets

# 删除现有的MNIST数据目录
mnist_dir = 'data/MNIST'
if os.path.exists(mnist_dir):
    print(f"Deleting existing MNIST directory: {mnist_dir}")
    shutil.rmtree(mnist_dir)

# 重新下载MNIST数据集
print("Downloading MNIST dataset...")
train_dataset = datasets.MNIST(root='data/MNIST', train=True, download=True)
test_dataset = datasets.MNIST(root='data/MNIST', train=False, download=True)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")
print("MNIST dataset downloaded successfully!")