"""
推理脚本：用训练好的模型做预测

用法：
    # 在测试集里随机抽几张图预测（不需要准备图片）
    python src/predict.py --checkpoint mnist_cnn.pth --model cnn --demo

    # 预测你自己的手写图片（白底黑字，推荐）
    python src/predict.py --checkpoint mnist_cnn.pth --model cnn --image 你的图片.png

说明：
    真实的手写照片往往不是 MNIST 那种「黑底白字、居中、28x28」的格式，
    所以 --image 模式会自动做三件事：
      1. 转灰度
      2. 自动判断并反色（保证是黑底白字）
      3. 裁剪到内容边界、等比缩放到 20x20、再贴到 28x28 画布中央
    这套处理是让「自己拍的照片」也能被识别出来的关键，
    也是这个项目从「跑通教程」变成「能用的工具」的分界线。
"""
import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps

from data_loader import MNIST_MEAN, MNIST_STD, get_device
from model import build_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def preprocess_image(path):
    """把任意手写图片处理成 MNIST 风格的张量"""
    img = Image.open(path).convert("L")

    # 判断是黑底白字还是白底黑字，统一成黑底白字
    arr = np.array(img)
    if arr.mean() > 127:            # 平均偏亮 => 白底黑字，需要反色
        img = ImageOps.invert(img)

    # 按内容裁剪，去掉周围大片空白
    bbox = ImageOps.invert(img).getbbox()
    if bbox:
        img = img.crop(bbox)

    # 等比缩放到 20x20（MNIST 标准做法：数字主体约 20x20）
    img.thumbnail((20, 20), Image.LANCZOS)

    # 贴到 28x28 画布正中央
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(img, ((28 - img.width) // 2, (28 - img.height) // 2))

    tensor = torch.tensor(np.array(canvas), dtype=torch.float32) / 255.0
    tensor = (tensor - MNIST_MEAN[0]) / MNIST_STD[0]
    return canvas, tensor.unsqueeze(0).unsqueeze(0)     # (1, 1, 28, 28)


@torch.no_grad()
def predict_tensor(model, tensor, device):
    """返回 (预测类别, 各类别置信度)"""
    model.eval()
    logits = model(tensor.to(device))
    probs = torch.softmax(logits, dim=1)[0]
    return int(probs.argmax()), probs.cpu().numpy()


def demo_random(model, device, data_dir, model_name, n=6):
    """从测试集随机抽 n 张图做预测，画成一张网格图"""
    from torchvision import datasets

    from data_loader import build_transform
    test_set = datasets.MNIST(root=data_dir, train=False, download=True,
                              transform=build_transform())

    indices = random.sample(range(len(test_set)), n)
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 3))

    for ax, idx in zip(axes, indices):
        img_tensor, label = test_set[idx]
        pred, probs = predict_tensor(model, img_tensor.unsqueeze(0), device)
        conf = probs[pred] * 100

        ax.imshow(img_tensor.squeeze().numpy(), cmap="gray")
        ok = pred == label
        ax.set_title(f"真实 {label} / 预测 {pred}\n置信度 {conf:.1f}%",
                     fontsize=10, color="#2f7d32" if ok else "#c0392b")
        ax.axis("off")

    fig.suptitle(f"{model_name.upper()} 随机样本预测", fontsize=13)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, f"prediction_samples_{model_name}.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"预测结果图已保存: {path}")


def main():
    parser = argparse.ArgumentParser(description="MNIST 推理脚本")
    parser.add_argument("--model", choices=["ann", "cnn"], default="cnn")
    parser.add_argument("--checkpoint", default=None, help="权重文件路径")
    parser.add_argument("--image", default=None, help="要预测的图片路径")
    parser.add_argument("--demo", action="store_true", help="随机抽测试集样本预测")
    parser.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    args = parser.parse_args()

    device = get_device()
    ckpt = args.checkpoint or os.path.join(ROOT, f"mnist_{args.model}.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"找不到权重文件 {ckpt}\n请先运行: python src/train.py --model {args.model}"
        )

    model = build_model(args.model).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    print(f"已加载权重: {ckpt}  (模型: {args.model.upper()})")

    if args.image:
        canvas, tensor = preprocess_image(args.image)
        pred, probs = predict_tensor(model, tensor, device)

        print(f"\n预测结果: {pred}   置信度: {probs[pred] * 100:.2f}%")
        print("各类别概率:")
        for d, p in enumerate(probs):
            bar = "#" * int(p * 40)
            print(f"  {d}: {p * 100:6.2f}%  {bar}")

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(Image.open(args.image).convert("L"), cmap="gray")
        axes[0].set_title("原始图片")
        axes[0].axis("off")
        axes[1].imshow(np.array(canvas), cmap="gray")
        axes[1].set_title(f"模型看到的输入 24-> 预测 {pred}")
        axes[1].axis("off")
        fig.tight_layout()
        path = os.path.join(FIG_DIR, "prediction_custom.png")
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"\n可视化已保存: {path}")

    elif args.demo:
        demo_random(model, device, args.data_dir, args.model)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
