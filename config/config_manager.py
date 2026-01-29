import os
from dotenv import load_dotenv

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, env_file='.env'):
        """
        初始化配置管理器
        Args:
            env_file: 环境配置文件路径
        """
        # 加载环境变量
        load_dotenv(env_file)
        
        # 数据配置
        self.DATA_DIR = os.getenv('DATA_DIR', 'data/MNIST')
        self.BATCH_SIZE = int(os.getenv('BATCH_SIZE', '64'))
        self.VAL_SPLIT = float(os.getenv('VAL_SPLIT', '0.1'))
        
        # 模型配置
        self.MODEL_ARCHITECTURE = os.getenv('MODEL_ARCHITECTURE', 'lightweight')
        self.EPOCHS = int(os.getenv('EPOCHS', '10'))
        self.LEARNING_RATE = float(os.getenv('LEARNING_RATE', '0.001'))
        
        # 保存配置
        self.MODEL_SAVE_DIR = os.getenv('MODEL_SAVE_DIR', 'models/trained')
        self.EXPORT_DIR = os.getenv('EXPORT_DIR', 'models/exported')
        self.REPORTS_DIR = os.getenv('REPORTS_DIR', 'reports')
        
        # 设备配置
        self.DEVICE = os.getenv('DEVICE', 'cuda')
    
    def get_config(self):
        """
        获取所有配置
        Returns:
            config: 配置字典
        """
        return {
            'DATA_DIR': self.DATA_DIR,
            'BATCH_SIZE': self.BATCH_SIZE,
            'VAL_SPLIT': self.VAL_SPLIT,
            'MODEL_ARCHITECTURE': self.MODEL_ARCHITECTURE,
            'EPOCHS': self.EPOCHS,
            'LEARNING_RATE': self.LEARNING_RATE,
            'MODEL_SAVE_DIR': self.MODEL_SAVE_DIR,
            'EXPORT_DIR': self.EXPORT_DIR,
            'REPORTS_DIR': self.REPORTS_DIR,
            'DEVICE': self.DEVICE
        }
    
    def print_config(self):
        """打印配置信息"""
        print('Project Configuration:')
        print('=' * 50)
        for key, value in self.get_config().items():
            print(f'{key}: {value}')
        print('=' * 50)

# 创建全局配置实例
config = ConfigManager()