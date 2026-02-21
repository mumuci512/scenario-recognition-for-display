import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import argparse
import os
import sys

# --- 模型结构定义 (必须与训练时完全一致) ---

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))

class CBAM(nn.Module):
    def __init__(self, in_planes, learnable=True, init_weight=0.2):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()
        # 兼容两种加载模式
        self.center_bias = nn.Parameter(torch.tensor([init_weight]))

    def forward(self, x):
        x = x * self.ca(x)
        sa_map = self.sa(x)
        return x * sa_map, sa_map

class GuidedModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super(GuidedModel, self).__init__()
        self.model_name = model_name
        if model_name == 'resnet18':
            base = models.resnet18(weights=None)
            self.features = nn.Sequential(*list(base.children())[:-2])
            self.cbam = CBAM(512)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(512, num_classes)
        elif model_name == 'mobilenet_v2':
            base = models.mobilenet_v2(weights=None)
            self.features = base.features
            self.cbam = CBAM(1280)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.features(x)
        x, _ = self.cbam(x)
        out = self.pool(x)
        out = torch.flatten(out, 1)
        if self.model_name == 'resnet18':
            return self.fc(out)
        else:
            return self.classifier(out)

# --- 工具函数 ---

def load_model(arch, path, mtype, num_classes, device):
    """
    arch: 'resnet18' or 'mobilenet_v2'
    mtype: 'std' (无引导) or 'guided' (Hard/Soft引导)
    """
    if mtype == 'std':
        if arch == 'resnet18':
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            model = models.mobilenet_v2(weights=None)
            model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    else:
        model = GuidedModel(arch, num_classes)
    
    # weights_only=False 用于加载包含自定义类的 checkpoint
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    # 兼容处理 model_state_dict 键值
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # 尝试加载，strict=False 以防全连接层名称微小差异
    model.load_state_dict(state_dict, strict=False)
    return model.to(device).eval()

def run_eval(model, loader, device, model_name):
    preds, labels = [], []
    total = len(loader)
    print(f"  --> Progress:", end=" ", flush=True)
    
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            inputs = x.to(device)
            outputs = model(inputs)
            
            # 兼容 GuidedModel 可能返回 tuple 的情况
            if isinstance(outputs, tuple):
                outputs = outputs[0]
                
            preds.extend(torch.argmax(outputs, 1).cpu().numpy())
            labels.extend(y.numpy())
            
            if i % max(1, total // 5) == 0:
                print(f"{int((i+1)/total*100)}%...", end=" ", flush=True)
                
    print("Done.")
    return accuracy_score(labels, preds), confusion_matrix(labels, preds)

def plot_results(results, class_names):
    if not results: return

    # 1. 准确率柱状图
    names = [r[0] for r in results]
    accs = [r[1] for r in results]
    
    plt.figure(figsize=(12, 6))
    colors = sns.color_palette("husl", len(results))
    bars = plt.bar(names, accs, color=colors)
    plt.ylim(0, 1.1)
    plt.title('Accuracy Comparison across Guidance Strategies', fontsize=14)
    plt.ylabel('Average Accuracy')
    plt.xticks(rotation=15)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, f'{height:.2%}', ha='center')
    
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png')
    print("\n[Output] Saved: accuracy_comparison.png")

    # 2. 混淆矩阵热力图
    n = len(results)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes = axes.flatten()
    
    for i, (name, acc, cm) in enumerate(results):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=class_names, yticklabels=class_names)
        axes[i].set_title(f'{name} (Acc: {acc:.2%})')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')
    
    # 隐藏多余的子图
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    print("[Output] Saved: confusion_matrices.png")

# --- 主程序 ---

def main():
    parser = argparse.ArgumentParser(description="Multi-Model Accuracy Evaluation Script")
    parser.add_argument('--test_dir', required=True, help='Path to test/val dataset')
    parser.add_argument('--batch_size', type=int, default=32)
    # 路径参数
    parser.add_argument('--res_none', help='ResNet18 Standard path')
    parser.add_argument('--res_hard', help='ResNet18 Hard Mask path')
    parser.add_argument('--res_soft', help='ResNet18 Soft Mask path')
    parser.add_argument('--mb_none', help='MobileNetV2 Standard path')
    parser.add_argument('--mb_hard', help='MobileNetV2 Hard Mask path')
    parser.add_argument('--mb_soft', help='MobileNetV2 Soft Mask path')
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        test_dataset = datasets.ImageFolder(args.test_dir, transform=transform)
        # Windows 必须设置 num_workers=0 以防死锁
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                                shuffle=False, num_workers=0)
        class_names = test_dataset.classes
        print(f"Dataset loaded: {len(test_dataset)} images, {len(class_names)} classes.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 待评估配置列表
    configs = [
        ('ResNet-None', 'resnet18', args.res_none, 'std'),
        ('ResNet-Hard', 'resnet18', args.res_hard, 'guided'),
        ('ResNet-Soft', 'resnet18', args.res_soft, 'guided'),
        ('MobileNet-None', 'mobilenet_v2', args.mb_none, 'std'),
        ('MobileNet-Hard', 'mobilenet_v2', args.mb_hard, 'guided'),
        ('MobileNet-Soft', 'mobilenet_v2', args.mb_soft, 'guided'),
    ]

    results = []
    
    for name, arch, path, mtype in configs:
        if path and os.path.exists(path):
            print(f"\nEvaluating {name}...")
            try:
                model = load_model(arch, path, mtype, len(class_names), device)
                acc, cm = run_eval(model, test_loader, device, name)
                results.append((name, acc, cm))
                print(f"  Result: Accuracy = {acc:.2%}")
                
                # 显式清理内存，防止多个大模型挤爆显存
                del model
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"  Failed to evaluate {name}: {e}")
        else:
            if path: print(f"\n[Warning] Skip {name}: File not found at {path}")

    # 可视化
    if results:
        plot_results(results, class_names)
        print("\nEvaluation complete.")
    else:
        print("\nNo models were successfully evaluated. Please check your file paths.")

if __name__ == '__main__':
    main()