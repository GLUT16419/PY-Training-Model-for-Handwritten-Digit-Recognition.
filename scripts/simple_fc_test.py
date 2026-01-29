import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

class SimpleFC(nn.Module):
    def __init__(self):
        super(SimpleFC, self).__init__()
        self.fc1 = nn.Linear(28*28, 10)
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        return x

def main():
    # 设置设备
    device = torch.device("cpu")  # 使用CPU测试
    print(f"Using device: {device}")
    
    # 数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # 加载数据
    train_dataset = datasets.MNIST('data/MNIST', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data/MNIST', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # 创建模型
    model = SimpleFC().to(device)
    print(f"Model: {model.__class__.__name__}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # 训练模型
    epochs = 5
    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        print(f'Epoch {epoch} Train: Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
        
        # 测试
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_loss = test_loss / len(test_loader)
        test_acc = 100. * correct / total
        print(f'Test: Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%')
        print()

if __name__ == "__main__":
    main()