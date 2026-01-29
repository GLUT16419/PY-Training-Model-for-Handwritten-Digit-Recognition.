import os
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import datasets, transforms
from PIL import Image

# 自定义噪声变换
class AddGaussianNoise(object):
    """添加高斯噪声"""
    def __init__(self, mean=0, std=0.05):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise


class AddSaltPepperNoise(object):
    """添加椒盐噪声"""
    def __init__(self, salt_prob=0.01, pepper_prob=0.01):
        self.salt_prob = salt_prob
        self.pepper_prob = pepper_prob
        
    def __call__(self, tensor):
        noise = torch.rand(tensor.size())
        tensor = torch.where(noise < self.salt_prob, 1.0, tensor)
        tensor = torch.where(noise > (1 - self.pepper_prob), 0.0, tensor)
        return tensor

class AddRandomErasing(object):
    """添加随机擦除"""
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        
    def __call__(self, tensor):
        if torch.rand(1) < self.p:
            return tensor
        else:
            return transforms.RandomErasing(p=1.0, scale=self.scale, ratio=self.ratio)(tensor)

class RandomSharpness(object):
    """随机锐化"""
    def __init__(self, sharpness_factor=2.0, p=0.5):
        self.sharpness_factor = sharpness_factor
        self.p = p
        
    def __call__(self, img):
        if torch.rand(1) < self.p:
            return transforms.functional.adjust_sharpness(img, self.sharpness_factor)
        return img

class RandomPosterize(object):
    """随机降低色彩深度"""
    def __init__(self, bits=2, p=0.3):
        self.bits = bits
        self.p = p
        
    def __call__(self, img):
        if torch.rand(1) < self.p:
            return transforms.functional.posterize(img, self.bits)
        return img

class RandomInvert(object):
    """随机反转颜色"""
    def __init__(self, p=0.2):
        self.p = p
        
    def __call__(self, img):
        if torch.rand(1) < self.p:
            return transforms.functional.invert(img)
        return img


# 自定义数据集类
class CustomMNISTDataset(torch.utils.data.Dataset):
    """自定义MNIST数据集"""
    
    def __init__(self, root_dir, transform=None):
        """
        初始化自定义数据集
        Args:
            root_dir: 数据集根目录
            transform: 数据转换
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.digit_groups = {}  # 按数字分组存储图像路径
        
        try:
            # 收集所有图像和标签
            for digit in range(10):
                digit_dir = os.path.join(root_dir, str(digit))
                if os.path.exists(digit_dir):
                    digit_images = []
                    for filename in os.listdir(digit_dir):
                        if filename.endswith('.png'):
                            img_path = os.path.join(digit_dir, filename)
                            self.images.append(img_path)
                            self.labels.append(digit)
                            digit_images.append(img_path)
                    self.digit_groups[digit] = digit_images
        except Exception as e:
            print(f"Error loading custom dataset: {e}")
            # 确保即使出错也能返回有效的数据集
            self.images = []
            self.labels = []
            self.digit_groups = {i: [] for i in range(10)}
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.images)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        try:
            img_path = self.images[idx]
            label = self.labels[idx]
            
            # 加载和预处理图像
            img = Image.open(img_path).convert('L')
            
            # 添加自适应阈值处理，提高自定义数据的质量
            img = self._preprocess_image(img)
            
            if self.transform:
                img = self.transform(img)
            else:
                # 默认转换
                transform = transforms.Compose([
                    transforms.Resize((28, 28)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
                img = transform(img)
            
            return img, label
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # 返回一个默认的零图像和标签0
            default_img = Image.new('L', (28, 28), color=0)
            if self.transform:
                default_img = self.transform(default_img)
            else:
                transform = transforms.Compose([
                    transforms.Resize((28, 28)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
                default_img = transform(default_img)
            return default_img, 0
    
    def _preprocess_image(self, img):
        """预处理图像，确保与MNIST数据特征对齐"""
        try:
            # 调整大小
            img = img.resize((28, 28), Image.LANCZOS)
            
            # 转换为numpy数组
            import numpy as np
            img_array = np.array(img)
            
            # 自适应阈值处理
            img_array = self._adaptive_threshold(img_array)
            
            # 反转颜色（如果需要），确保黑色背景白色数字
            if np.mean(img_array) > 128:
                img_array = 255 - img_array
            
            # 归一化到0-255范围
            if np.max(img_array) > 0:
                img_array = (img_array / np.max(img_array)) * 255
            
            # 转换回PIL图像
            img = Image.fromarray(img_array.astype(np.uint8))
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            # 返回默认的零图像
            img = Image.new('L', (28, 28), color=0)
        
        return img
    
    def _adaptive_threshold(self, img_array, block_size=11, C=10):
        """自适应阈值处理"""
        try:
            import numpy as np
            from scipy.ndimage import uniform_filter
            
            # 计算局部均值
            local_mean = uniform_filter(img_array, size=block_size)
            
            # 应用自适应阈值
            thresholded = np.where(img_array > local_mean - C, 255, 0)
        except Exception as e:
            print(f"Error applying adaptive threshold: {e}")
            # 返回原始图像
            thresholded = img_array
        
        return thresholded
    
    def get_digit_group(self, digit):
        """获取指定数字的所有图像路径"""
        return self.digit_groups.get(digit, [])

class SubsetDataset(torch.utils.data.Dataset):
    """子集数据集"""
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            label = int(os.path.basename(os.path.dirname(img_path)))
            img = Image.open(img_path).convert('L')
            
            # 添加与CustomMNISTDataset相同的预处理逻辑
            img = self._preprocess_image(img)
            
            if self.transform:
                img = self.transform(img)
            else:
                transform = transforms.Compose([
                    transforms.Resize((28, 28)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
                img = transform(img)
            return img, label
        except Exception as e:
            print(f"Error loading subset sample {idx}: {e}")
            # 返回一个默认的零图像和标签0
            default_img = Image.new('L', (28, 28), color=0)
            if self.transform:
                default_img = self.transform(default_img)
            else:
                transform = transforms.Compose([
                    transforms.Resize((28, 28)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
                default_img = transform(default_img)
            return default_img, 0
    
    def _preprocess_image(self, img):
        """预处理图像，确保与MNIST数据特征对齐"""
        try:
            # 调整大小
            img = img.resize((28, 28), Image.LANCZOS)
            
            # 转换为numpy数组
            import numpy as np
            img_array = np.array(img)
            
            # 自适应阈值处理
            img_array = self._adaptive_threshold(img_array)
            
            # 反转颜色（如果需要），确保黑色背景白色数字
            if np.mean(img_array) > 128:
                img_array = 255 - img_array
            
            # 归一化到0-255范围
            if np.max(img_array) > 0:
                img_array = (img_array / np.max(img_array)) * 255
            
            # 转换回PIL图像
            img = Image.fromarray(img_array.astype(np.uint8))
        except Exception as e:
            print(f"Error preprocessing subset image: {e}")
            # 返回默认的零图像
            img = Image.new('L', (28, 28), color=0)
        
        return img
    
    def _adaptive_threshold(self, img_array, block_size=11, C=10):
        """自适应阈值处理"""
        try:
            import numpy as np
            from scipy.ndimage import uniform_filter
            
            # 计算局部均值
            local_mean = uniform_filter(img_array, size=block_size)
            
            # 应用自适应阈值
            thresholded = np.where(img_array > local_mean - C, 255, 0)
        except Exception as e:
            print(f"Error applying adaptive threshold: {e}")
            # 返回原始图像
            thresholded = img_array
        
        return thresholded

class MNISTDataProcessor:
    """MNIST数据集处理器"""
    
    def __init__(self, data_dir='data/MNIST', batch_size=64, val_split=0.1):
        """
        初始化数据处理器
        Args:
            data_dir: 数据集目录
            batch_size: 批处理大小
            val_split: 验证集比例
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 定义数据转换，增强自定义数据的预处理
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.1,
                hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # 验证数据转换，使用相同的预处理
        self.test_transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.1,
                hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
    def load_data(self):
        """
        加载MNIST数据集
        Returns:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
        """
        # 加载训练集和测试集
        # 使用download=True，PyTorch会自动检查本地是否已有数据集
        # 如果本地已有完整数据集，就不会重新下载
        train_dataset = datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=False,  # 不下载，使用本地数据
            transform=self.transform
        )
        
        test_dataset = datasets.MNIST(
            root=self.data_dir,
            train=False,
            download=False,  # 不下载，使用本地数据
            transform=self.test_transform
        )
        
        # 划分训练集和验证集
        val_size = int(len(train_dataset) * self.val_split)
        train_size = len(train_dataset) - val_size
        
        train_subset, val_subset = random_split(
            train_dataset,
            [train_size, val_size]
        )
        
        # 创建数据加载器
        # 根据系统CPU核心数设置worker数量
        import multiprocessing
        num_workers = min(multiprocessing.cpu_count(), 12)  # 最多使用12个worker，提高数据加载速度
        
        # 优化DataLoader配置
        train_loader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,  # 使用CUDA时启用，加速数据传输到GPU
            pin_memory_device=self.device.type if self.device.type == 'cuda' else '',
            prefetch_factor=8,  # 提高预取因子，增加数据预加载
            persistent_workers=True  # 保持worker进程活跃，减少启动开销
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            pin_memory_device=self.device.type if self.device.type == 'cuda' else '',
            prefetch_factor=8,
            persistent_workers=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            pin_memory_device=self.device.type if self.device.type == 'cuda' else '',
            prefetch_factor=8,
            persistent_workers=True
        )
        
        return train_loader, val_loader, test_loader
    
    def load_custom_data(self, custom_data_dir):
        """
        加载自定义数据集
        Args:
            custom_data_dir: 自定义数据集目录
        Returns:
            custom_loader: 自定义数据加载器
        """
        # 加载自定义数据集
        custom_dataset = CustomMNISTDataset(
            root_dir=custom_data_dir,
            transform=self.transform
        )
        
        # 创建数据加载器
        import multiprocessing
        num_workers = min(multiprocessing.cpu_count(), 12)  # 最多使用12个worker，提高数据加载速度
        
        custom_loader = DataLoader(
            custom_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            pin_memory_device=self.device.type if self.device.type == 'cuda' else '',
            prefetch_factor=8,  # 提高预取因子，增加数据预加载
            persistent_workers=True
        )
        
        return custom_loader
    
    def load_mixed_data(self, custom_data_dir, custom_ratio=None):
        """
        加载混合数据集（MNIST + 自定义数据）
        Args:
            custom_data_dir: 自定义数据集目录
            custom_ratio: 自定义数据在混合数据集中的比例（None表示自动计算）
        Returns:
            train_loader: 混合训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
            custom_val_loader: 自定义数据验证加载器
        """
        # 加载MNIST训练集和测试集
        mnist_train = datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=False,
            transform=self.transform
        )
        
        mnist_test = datasets.MNIST(
            root=self.data_dir,
            train=False,
            download=False,
            transform=self.test_transform
        )
        
        # 加载自定义数据集
        custom_dataset = CustomMNISTDataset(
            root_dir=custom_data_dir,
            transform=self.transform
        )
        
        # 分割自定义数据集：每个数字最多60张用于训练，剩余用于验证
        train_custom_images = []
        val_custom_images = []
        
        for digit in range(10):
            digit_images = custom_dataset.get_digit_group(digit)
            # 打乱顺序
            random.shuffle(digit_images)
            # 每个数字最多取60张用于训练
            train_count = min(60, len(digit_images))
            train_custom_images.extend(digit_images[:train_count])
            # 剩余的用于验证
            val_custom_images.extend(digit_images[train_count:])
        
        print(f"自定义数据集分配:")
        print(f"  训练集: {len(train_custom_images)} 张图片")
        print(f"  验证集: {len(val_custom_images)} 张图片")
        
        # 创建训练和验证的自定义数据集
        train_custom_dataset = SubsetDataset(train_custom_images, transform=self.transform)
        val_custom_dataset = SubsetDataset(val_custom_images, transform=self.test_transform)
        
        # 计算混合比例
        if custom_ratio is None:
            # 自动计算比例，基于自定义数据集大小
            custom_size = len(train_custom_dataset)
            mnist_size = len(mnist_train)
            
            # 计算比例，使自定义数据占合理比例
            # 自定义数据最多占50%，最少占10%
            if custom_size > 0:
                custom_ratio = min(0.5, max(0.1, custom_size / (mnist_size + custom_size) * 2.5))
            else:
                custom_ratio = 0.0
        
        # 计算数据集大小
        total_size = len(mnist_train) + len(train_custom_dataset)
        custom_size = int(total_size * custom_ratio)
        mnist_size = total_size - custom_size
        
        # 确保大小合理
        if custom_size > len(train_custom_dataset):
            custom_size = len(train_custom_dataset)
            mnist_size = total_size - custom_size
        
        if mnist_size <= 0:
            mnist_size = 1
        
        # 随机选择MNIST数据
        mnist_indices = random.sample(range(len(mnist_train)), mnist_size)
        selected_mnist = torch.utils.data.Subset(mnist_train, mnist_indices)
        
        # 合并数据集
        mixed_dataset = ConcatDataset([selected_mnist, train_custom_dataset])
        
        # 划分训练集和验证集
        val_size = int(len(mixed_dataset) * self.val_split)
        train_size = len(mixed_dataset) - val_size
        
        train_subset, val_subset = random_split(
            mixed_dataset,
            [train_size, val_size]
        )
        
        # 创建数据加载器
        import multiprocessing
        num_workers = min(multiprocessing.cpu_count(), 12)  # 最多使用12个worker，提高数据加载速度
        
        train_loader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            pin_memory_device=self.device.type if self.device.type == 'cuda' else '',
            prefetch_factor=8,  # 提高预取因子，增加数据预加载
            persistent_workers=True
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            pin_memory_device=self.device.type if self.device.type == 'cuda' else '',
            prefetch_factor=8,
            persistent_workers=True
        )
        
        test_loader = DataLoader(
            mnist_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            pin_memory_device=self.device.type if self.device.type == 'cuda' else '',
            prefetch_factor=8,
            persistent_workers=True
        )
        
        # 创建自定义数据验证加载器
        if len(val_custom_dataset) > 0:
            custom_val_loader = DataLoader(
                val_custom_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                pin_memory_device=self.device.type if self.device.type == 'cuda' else '',
                prefetch_factor=8,
                persistent_workers=True
            )
        else:
            custom_val_loader = None
        
        return train_loader, val_loader, test_loader, custom_val_loader
    
    def get_data_shape(self):
        """
        获取数据形状
        Returns:
            input_shape: 输入数据形状
            num_classes: 类别数量
        """
        # 直接返回固定的MNIST数据形状，避免重复加载数据
        input_shape = (1, 28, 28)  # MNIST图像形状：(通道, 高度, 宽度)
        num_classes = 10  # MNIST有10个类别
        
        return input_shape, num_classes

if __name__ == "__main__":
    # 测试数据处理器
    processor = MNISTDataProcessor()
    train_loader, val_loader, test_loader = processor.load_data()
    input_shape, num_classes = processor.get_data_shape()
    
    print(f"Data loaded successfully!")
    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # 测试自定义数据加载
    custom_data_dir = 'data/custom/sample'
    if os.path.exists(custom_data_dir):
        custom_loader = processor.load_custom_data(custom_data_dir)
        print(f"Custom data batches: {len(custom_loader)}")
        
        # 测试混合数据加载
        mixed_train, mixed_val, mixed_test = processor.load_mixed_data(custom_data_dir)
        print(f"Mixed train batches: {len(mixed_train)}")
        print(f"Mixed validation batches: {len(mixed_val)}")
        print(f"Mixed test batches: {len(mixed_test)}")