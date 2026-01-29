import torch
import torch.nn as nn
from torchvision import datasets, transforms

class SimpleFC(nn.Module):
    def __init__(self):
        super(SimpleFC, self).__init__()
        self.fc1 = nn.Linear(28*28, 10)
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        return x

def check_model_output():
    """检查模型输出"""
    print("Checking model output...")
    
    # 数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # 加载数据
    test_dataset = datasets.MNIST('data/MNIST', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)
    
    # 创建模型
    model = SimpleFC()
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # 训练一个批次
    train_dataset = datasets.MNIST('data/MNIST', train=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        print(f"First batch loss: {loss.item()}")
        print(f"Output shape: {outputs.shape}")
        print(f"First 5 outputs: {outputs[:5]}")
        print(f"Output max values: {outputs.max(dim=1)[0][:5]}")
        print(f"Output argmax: {outputs.argmax(dim=1)[:20]}")
        print(f"Targets: {targets[:20]}")
        break
    
    # 测试模型
    model.eval()
    correct = 0
    total = 0
    predictions = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            predictions.extend(predicted.tolist())
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    # 统计预测分布
    import collections
    pred_counts = collections.Counter(predictions)
    print("\nPrediction distribution:")
    for pred in sorted(pred_counts.keys()):
        print(f"Prediction {pred}: {pred_counts[pred]} ({pred_counts[pred]/total*100:.2f}%)")
    
    print(f"\nTest accuracy: {100.*correct/total:.2f}%")

if __name__ == "__main__":
    check_model_output()