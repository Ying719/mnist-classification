"""
数据加载：把 MNIST 数据集的下载、预处理、分批三件事收在一起。

原版代码把这段直接写在主脚本里，升级后抽成模块，
这样 train.py 和 predict.py 可以共用同一套预处理逻辑
（这一点很重要：训练和推理的预处理必须一致，否则准确率会莫名下降）。
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# MNIST 官方统计出的全局均值和标准差，几乎所有教程都用这两个数
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def get_device():
    """有 GPU 就用 GPU，没有就用 CPU"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transform():
    """构建预处理流水线：转张量 + 标准化"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])


def get_dataloaders(data_dir="./data", batch_size=64):
    """返回 (train_loader, test_loader)"""
    transform = build_transform()

    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
