import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

# 添加项目根目录到Python搜索路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import MNISTDataProcessor

class DebugModel(nn.Module):
    """用于调试的简单模型"""
    def __init__(self):
        super(DebugModel, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def debug_training():
    """调试模型训练过程"""
    print("Debugging model training...")
    
    # 创建数据处理器
    processor = MNISTDataProcessor(batch_size=64, val_split=0.1)
    
    # 加载数据
    train_loader, val_loader, test_loader = processor.load_data()
    
    # 创建简单模型
    model = DebugModel()
    print(f"Model created: {model.__class__.__name__}")
    
    # 检查模型参数
    print("\nModel parameters:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练几轮
    print("\nStarting training...")
    for epoch in range(3):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, targets) in enumerate(train_loader):
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 计算损失和准确率
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 每100批次打印一次
            if i % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {i}, Loss: {running_loss/(i+1):.4f}, Acc: {100.*correct/total:.2f}%')
            
            # 只训练前200批次
            if i >= 200:
                break
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        print(f'Epoch {epoch+1} Summary:')
        print(f'Train Loss: {running_loss/201:.4f}, Train Acc: {100.*correct/total:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {100.*val_correct/val_total:.2f}%')
    
    print("\nDebug training completed!")

if __name__ == "__main__":
    debug_training()