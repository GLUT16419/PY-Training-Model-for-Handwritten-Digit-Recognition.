import os
import sys

# 添加项目根目录到Python搜索路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.data_preprocessing import MNISTDataProcessor
from src.training import get_model, ModelTrainer

def test_training():
    """测试模型训练是否正常"""
    print("Testing model training...")
    
    # 创建数据处理器
    processor = MNISTDataProcessor(batch_size=64, val_split=0.1)
    
    # 加载数据
    train_loader, val_loader, test_loader = processor.load_data()
    
    # 获取数据形状
    input_shape, num_classes = processor.get_data_shape()
    
    # 测试简单模型
    print("\nTesting SimpleMNISTModel...")
    simple_model = get_model('simple', input_shape, num_classes)
    print(f"Model created: {simple_model.__class__.__name__}")
    
    # 创建训练器
    trainer = ModelTrainer(simple_model)
    
    # 测试训练
    print("\nStarting training...")
    trained_model = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,  # 只训练5个epoch进行测试
        model_save_path='models/trained/test'
    )
    
    # 评估模型
    print("\nEvaluating model...")
    test_loss, test_accuracy = trainer.evaluate(test_loader)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.2f}%")
    
    print("\nTraining test completed!")

if __name__ == "__main__":
    test_training()