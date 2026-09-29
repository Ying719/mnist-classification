"""
模型定义：ANN（全连接网络） 与 CNN（卷积神经网络）

两者输入完全相同（1x28x28 的灰度图，共 784 个像素），
区别在于「怎么处理这些像素」：

- ANN ：把 28x28 直接拉平成长度 784 的一维向量，每个像素独立地连到下一层。
        它不知道「左上角的像素」和「它右边的像素」是邻居，空间结构被丢掉了。

- CNN ：保留 28x28 的二维结构，用卷积核在图上滑动，因此能捕捉
        「笔画边缘」「局部形状」这类空间特征。这正是图像任务里 CNN 更强的原因。
"""
import torch.nn as nn


class MNIST_ANN(nn.Module):
    """全连接网络（原版模型，保留作为对比基线）"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)          # 关键一步：把 28x28 拉平成 784 维向量
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MNIST_CNN(nn.Module):
    """卷积神经网络（升级版）

    结构：2 组 [卷积 -> ReLU -> 池化] + 2 层全连接
    - Conv2d   : 用卷积核提取局部特征（边缘、转角、笔画）
    - MaxPool2d: 2x2 最大池化，把特征图尺寸减半，同时保留最强的响应
    - Dropout  : 随机丢弃一部分神经元，抑制过拟合
    """

    def __init__(self):
        super().__init__()
        # ---- 特征提取部分 ----
        self.features = nn.Sequential(
            # 输入 1x28x28 -> 输出 32x28x28（padding=1 保证尺寸不变）
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            # 32x28x28 -> 32x14x14
            nn.MaxPool2d(kernel_size=2),

            # 32x14x14 -> 64x14x14
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            # 64x14x14 -> 64x7x7
            nn.MaxPool2d(kernel_size=2),
        )

        # ---- 分类部分 ----
        self.classifier = nn.Sequential(
            nn.Flatten(),                 # 64x7x7 -> 3136 维
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),            # 训练时随机丢弃 50% 的神经元
            nn.Linear(128, 10),           # 输出 10 个类别（数字 0-9）
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(name="cnn"):
    """按名字构建模型，方便在 train.py 里切换对比"""
    name = name.lower()
    if name == "cnn":
        return MNIST_CNN()
    elif name == "ann":
        return MNIST_ANN()
    else:
        raise ValueError(f"未知模型名称: {name}，可选 'cnn' 或 'ann'")
