import torch
from torchvision import datasets, transforms

def check_data():
    """检查数据加载和标签分布"""
    print("Checking data loading and label distribution...")
    
    # 数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # 加载数据
    train_dataset = datasets.MNIST('data/MNIST', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data/MNIST', train=False, transform=transform)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # 检查标签分布
    train_labels = []
    for i in range(len(train_dataset)):
        _, label = train_dataset[i]
        train_labels.append(label)
    
    test_labels = []
    for i in range(len(test_dataset)):
        _, label = test_dataset[i]
        test_labels.append(label)
    
    # 统计标签分布
    import collections
    train_label_counts = collections.Counter(train_labels)
    test_label_counts = collections.Counter(test_labels)
    
    print("\nTrain label distribution:")
    for label in sorted(train_label_counts.keys()):
        print(f"Label {label}: {train_label_counts[label]} ({train_label_counts[label]/len(train_dataset)*100:.2f}%)")
    
    print("\nTest label distribution:")
    for label in sorted(test_label_counts.keys()):
        print(f"Label {label}: {test_label_counts[label]} ({test_label_counts[label]/len(test_dataset)*100:.2f}%)")
    
    # 检查一个批次的数据
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    for inputs, targets in train_loader:
        print(f"\nBatch shape: {inputs.shape}")
        print(f"Targets shape: {targets.shape}")
        print(f"Targets: {targets[:20]}")
        print(f"Unique targets: {torch.unique(targets)}")
        break

if __name__ == "__main__":
    check_data()