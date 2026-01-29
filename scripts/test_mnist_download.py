import os
import sys

# 添加项目根目录到Python搜索路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import MNISTDataProcessor

def test_mnist_download():
    """测试MNIST数据集自动下载功能"""
    print("Testing MNIST dataset auto-download...")
    
    # 创建数据处理器
    processor = MNISTDataProcessor()
    
    print("Data processor created successfully")
    print(f"Data directory: {processor.data_dir}")
    
    # 检查数据目录是否存在
    if not os.path.exists(processor.data_dir):
        print(f"Data directory {processor.data_dir} does not exist, will be created")
    else:
        print(f"Data directory {processor.data_dir} exists")
        # 检查是否已有MNIST数据
        mnist_train_dir = os.path.join(processor.data_dir, "MNIST", "raw")
        if os.path.exists(mnist_train_dir):
            files = os.listdir(mnist_train_dir)
            print(f"Existing files in MNIST/raw: {files}")
        else:
            print("MNIST/raw directory does not exist, will be created")
    
    # 加载数据（这会触发自动下载）
    print("\nLoading MNIST data...")
    try:
        train_loader, val_loader, test_loader = processor.load_data()
        print("Data loaded successfully!")
        print(f"Train batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # 检查数据是否真的下载成功
        mnist_train_dir = os.path.join(processor.data_dir, "MNIST", "raw")
        if os.path.exists(mnist_train_dir):
            files = os.listdir(mnist_train_dir)
            print(f"\nFiles after download: {files}")
            # 检查是否包含必要的文件
            required_files = [
                "train-images-idx3-ubyte",
                "train-labels-idx1-ubyte",
                "t10k-images-idx3-ubyte",
                "t10k-labels-idx1-ubyte"
            ]
            missing_files = []
            for file in required_files:
                if file not in files:
                    missing_files.append(file)
            
            if not missing_files:
                print("\n✅ All required MNIST files downloaded successfully!")
            else:
                print(f"\n❌ Missing files: {missing_files}")
        else:
            print("\n❌ MNIST/raw directory still does not exist")
            
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mnist_download()
