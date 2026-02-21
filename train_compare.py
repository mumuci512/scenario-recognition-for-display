import torch
import argparse
import time
import os
import numpy as np
from torchvision import models
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        mx = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg + mx)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=3 if kernel_size == 7 else 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        nn.init.constant_(self.conv.weight, 0)
        nn.init.constant_(self.conv.bias, 0)
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg, mx], dim=1)
        return self.sigmoid(self.conv(x))

class CBAM(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()
    def forward(self, x):
        x = x * self.ca(x)
        sa_map = self.sa(x)
        return x * sa_map.detach(), sa_map

class GuidedModel(nn.Module):
    def __init__(self, arch, num_classes):
        super().__init__()
        self.arch = arch
        if arch == 'resnet18':
            base = models.resnet18(weights=None)
            self.features = nn.Sequential(*list(base.children())[:-2])
            self.cbam = CBAM(512)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(512, num_classes)
        elif arch == 'mobilenet_v2':
            base = models.mobilenet_v2(weights=None)
            self.features = base.features
            self.cbam = CBAM(1280)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Linear(1280, num_classes)
    def forward(self, x):
        x = self.features(x)
        x, att = self.cbam(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        res = self.fc(x) if self.arch == 'resnet18' else self.classifier(x)
        return res, att


def get_standard_model(arch, num_classes):
    """构建与 train_compare.py 一致的标准模型结构"""
    if arch == 'resnet18':
        model = models.resnet18(weights=None)
        # train_compare.py 中修改的是 model.fc
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=None)
        # train_compare.py 中修改的是 model.classifier[1]
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def benchmark(args):
    # ---设备选择---
    # 优先使用用户指定的设备
    if args.device == 'gpu':
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            print(f"Using GPU: {gpu_name}")
        else:
            print("Warning: GPU is not available, fallback to CPU")
            device = torch.device("cpu")
    elif args.device == 'cpu':
        device = torch.device("cpu")
        print(f"Using CPU (User specified)")
    else:
        # 理论上不会走到这里，因为 argparse 有 choices 限制
        device = torch.device("cpu")
        print(f"Invalid device choice, using CPU")

    # --- 加载权重并自动检测模型类型 ---
    print(f"Loading checkpoint: {args.model}")
    try:
        checkpoint = torch.load(args.model, map_location=device)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # 兼容处理：获取 state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # 检查 state_dict 的 keys 是否包含 'cbam'。
    # GuidedModel 必然包含 'cbam.ca...' 或 'cbam.sa...'，标准模型没有。
    is_guided = any('cbam' in key for key in state_dict.keys())
    
    # 尝试自动推断类别数 (防止 hardcode 22 导致 mismatch)
    # 我们检查最后一层的权重形状
    num_classes = 22 # 默认备份
    last_keys = ['fc.weight', 'classifier.weight', 'classifier.1.weight']
    for k in last_keys:
        if k in state_dict:
            num_classes = state_dict[k].shape[0]
            break
            
    print(f"Detected Configuration:")
    print(f"   - Model Type: {'[Guided / CBAM Attention]' if is_guided else '[Standard / Torchvision]'}")
    print(f"   - Num Classes: {num_classes}")
    print(f"   - Architecture: {args.arch}")

    if is_guided:
        model = GuidedModel(args.arch, num_classes).to(device)
    else:
        model = get_standard_model(args.arch, num_classes).to(device)

    # strict=False 是为了兼容 Guided 模型的 'center_bias' (如果存在)
    # 或者标准模型中可能存在的一些冗余键
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    # 简单的完整性检查
    if is_guided and 'cbam.ca.fc1.weight' in missing:
         print("Warning: Detected Guided mode but CBAM weights are missing!")
    
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    print(f"Warming up...")
    for _ in range(50):
        with torch.no_grad():
            _ = model(dummy_input)

    print(f"Starting benchmark (500 iterations)...")
    
    # 根据设备选择计时方式
    if device.type == 'cuda':
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    timings = []

    with torch.no_grad():
        for i in range(500):
            if device.type == 'cuda':
                starter.record()
                _ = model(dummy_input)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender) # 毫秒
                timings.append(curr_time)
            else:
                start_t = time.time()
                _ = model(dummy_input)
                end_t = time.time()
                timings.append((end_t - start_t) * 1000) # 毫秒

    timings = np.array(timings)
    avg_ms = np.mean(timings)
    std_ms = np.std(timings)
    fps = 1000.0 / avg_ms

    print(f"\n--- Results for {args.arch} ---")
    print(f"Type: {'Guided' if is_guided else 'Standard'}")
    print(f"Device: {device.type.upper()}")  # 新增：打印使用的设备
    print(f"Average Latency: {avg_ms:.4f} ms")
    print(f"Std Deviation: {std_ms:.4f} ms")
    print(f"Theoretical FPS: {fps:.2f}")

    # 保存结果 (新增设备信息)
    filename = f"benchmark_{args.arch}_{'guided' if is_guided else 'standard'}_{device.type}.txt"
    with open(filename, "w") as f:
        f.write(f"Model File: {args.model}\n")
        f.write(f"Device: {device.type.upper()}\n")
        f.write(f"Type: {'Guided' if is_guided else 'Standard'}\n")
        f.write(f"Average Latency: {avg_ms:.4f} ms\n")
        f.write(f"Theoretical FPS: {fps:.2f}\n")
    print(f"Result saved to {filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to .pth file')
    parser.add_argument('--arch', type=str, default='resnet18', choices=['resnet18', 'mobilenet_v2'])
    parser.add_argument('--device', type=str, default='gpu', choices=['cpu', 'gpu'], 
                        help='Inference device (default: gpu)')
    args = parser.parse_args()
    benchmark(args)