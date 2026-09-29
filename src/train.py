"""
训练与评估脚本

用法：
    分别训练并对比 ANN / CNN
        python src/train.py --compare

    只训练 CNN
        python src/train.py --model cnn --epochs 5

    只训练 ANN
        python src/train.py --model ann --epochs 5

运行后会在 figures/ 下生成：
    - training_curve_<模型名>.png   训练过程的损失与准确率曲线
    - benchmark.png                 ANN 与 CNN 的测试准确率对比柱状图
    权重保存到 mnist_<模型名>.pth
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")          # 用非交互后端，方便脚本化保存图片（不弹窗）
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import get_dataloaders, get_device
from model import build_model

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """跑完一个 epoch，返回平均训练损失"""
    model.train()
    total_loss = 0.0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


@torch.no_grad()
def evaluate(model, test_loader, device):
    """在测试集上算准确率"""
    model.eval()
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        pred = model(data).argmax(dim=1)
        correct += pred.eq(target).sum().item()
    return 100.0 * correct / len(test_loader.dataset)


def train(model_name, epochs=5, lr=1e-3, batch_size=64, data_dir="./data"):
    """完整训练一个模型，返回结果字典"""
    device = get_device()
    print(f"\n{'=' * 58}")
    print(f"  开始训练 [{model_name.upper()}]  设备: {device}  epochs: {epochs}")
    print(f"{'=' * 58}")

    train_loader, test_loader = get_dataloaders(data_dir, batch_size)
    model = build_model(model_name).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型可训练参数量: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"loss": [], "acc": []}
    best_acc = 0.0
    start = time.time()

    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        acc = evaluate(model, test_loader, device)

        history["loss"].append(avg_loss)
        history["acc"].append(acc)
        best_acc = max(best_acc, acc)

        print(f"  Epoch {epoch}/{epochs} | 训练损失 {avg_loss:.4f} | 测试准确率 {acc:.2f}%")

        if acc >= best_acc:
            torch.save(model.state_dict(), os.path.join(ROOT, f"mnist_{model_name}.pth"))

    elapsed = time.time() - start
    print(f"  最佳准确率: {best_acc:.2f}% | 耗时: {elapsed:.1f}s")

    result = {
        "model": model_name,
        "epochs": epochs,
        "params": n_params,
        "best_acc": round(best_acc, 2),
        "final_acc": round(history["acc"][-1], 2),
        "elapsed_sec": round(elapsed, 1),
        "history": history,
    }

    plot_curve(history, model_name)
    return result


def plot_curve(history, model_name):
    """画训练过程曲线"""
    epochs = range(1, len(history["loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, history["loss"], marker="o", color="#2f5fa8")
    ax1.set_title(f"{model_name.upper()} 训练损失")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["acc"], marker="s", color="#c0392b")
    ax2.set_title(f"{model_name.upper()} 测试准确率")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, f"training_curve_{model_name}.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  曲线已保存: {path}")


def plot_benchmark(results):
    """画 ANN vs CNN 对比柱状图"""
    names = [r["model"].upper() for r in results]
    accs = [r["best_acc"] for r in results]

    fig, ax = plt.subplots(figsize=(6, 4.2))
    bars = ax.bar(names, accs, color=["#8fa8c8", "#2f5fa8"], width=0.5)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.15,
                f"{acc:.2f}%", ha="center", va="bottom", fontsize=12)

    ax.set_ylim(90, 100)
    ax.set_ylabel("测试准确率 (%)")
    ax.set_title("MNIST 手写数字分类：ANN vs CNN")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "benchmark.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n对比图已保存: {path}")


def main():
    parser = argparse.ArgumentParser(description="MNIST 训练脚本")
    parser.add_argument("--model", choices=["ann", "cnn"], default="cnn")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    parser.add_argument("--compare", action="store_true",
                        help="依次训练 ANN 和 CNN 并生成对比图")
    args = parser.parse_args()

    if args.compare:
        results = []
        for name in ["ann", "cnn"]:
            results.append(train(name, args.epochs, args.lr,
                                 args.batch_size, args.data_dir))
        plot_benchmark(results)

        # 结果落盘成 JSON，方便写进 README、也方便以后对比不同参数
        out = os.path.join(ROOT, "benchmark_result.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n{'=' * 58}")
        print("  对比结果汇总")
        print(f"{'=' * 58}")
        for r in results:
            print(f"  {r['model'].upper():<4} 参数量 {r['params']:>9,} | "
                  f"准确率 {r['best_acc']:.2f}% | 耗时 {r['elapsed_sec']}s")
        print(f"\n结果已保存: {out}")
    else:
        train(args.model, args.epochs, args.lr, args.batch_size, args.data_dir)


if __name__ == "__main__":
    main()
