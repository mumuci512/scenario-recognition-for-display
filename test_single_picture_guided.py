import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
import cv2
import numpy as np
import os
import argparse
import sys
import torch.nn.functional as F

# 模型定义 (必须与训练脚本完全一致)
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
    def __init__(self, in_planes, learnable=False, init_weight=0.2):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()
        
        # 为了兼容加载权重，这里必须复现训练时的逻辑
        # 即使推理时不需要反向传播，变量名也必须存在
        if learnable:
            val = min(max(init_weight, 0.0), 0.5)
            self.center_bias = nn.Parameter(torch.tensor([val]))
        else:
            self.register_buffer('center_bias', torch.tensor([init_weight]))

    def forward(self, x):
        x = x * self.ca(x)
        sa_map = self.sa(x)
        return x * sa_map, sa_map # 注意：这里不要 detach，因为GradCAM可能需要梯度

class GuidedModel(nn.Module):
    def __init__(self, model_name, num_classes, learnable=False, init_weight=0.2):
        super(GuidedModel, self).__init__()
        self.model_name = model_name
        
        if model_name == 'resnet18':
            base = models.resnet18(weights=None) # 推理时不需下载预训练权重
            self.features = nn.Sequential(*list(base.children())[:-2])
            self.cbam = CBAM(512, learnable, init_weight)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(512, num_classes)
        elif model_name == 'mobilenet_v2':
            base = models.mobilenet_v2(weights=None)
            self.features = base.features
            self.cbam = CBAM(1280, learnable, init_weight)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Linear(1280, num_classes)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def forward(self, x):
        x = self.features(x)
        x, att = self.cbam(x) # 接收 feature 和 attention map
        out = self.pool(x)
        out = torch.flatten(out, 1)

        if self.model_name == 'resnet18':
            out = self.fc(out)
        else:
            out = self.classifier(out)
        return out, att

# Grad-CAM 逻辑 (适配 Tuple 输出)
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 注册钩子
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output[0] 是梯度张量
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        # 前向传播
        # 注意：GuidedModel 返回 (logits, attention_map)
        output, attn_map = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)

        # 反向传播
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        
        # 我们只对分类 logits 求导，忽略 attention_map
        output.backward(gradient=one_hot, retain_graph=True)

        # 生成 CAM
        gradients = self.gradients.cpu().detach().numpy()
        activations = self.activations.cpu().detach().numpy()
        
        # Global Average Pooling over gradients
        weights = np.mean(gradients, axis=(2, 3)) # [batch, channels]
        
        # 此时 weights shape: (1, 512) or (1, 1280)
        # activations shape: (1, 512, 7, 7)
        
        cam = np.zeros(activations.shape[2:], dtype=np.float32) # (7, 7)

        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i, :, :]

        cam = np.maximum(cam, 0) # ReLU
        cam = cv2.resize(cam, (224, 224))
        cam -= np.min(cam)
        cam /= (np.max(cam) + 1e-7)

        # 同时返回实际的 Spatial Attention Map 用于对比
        real_attn = attn_map.detach().cpu().numpy()[0, 0] # [1, 1, 7, 7] -> [7, 7]
        real_attn = cv2.resize(real_attn, (224, 224))
        # 归一化以便显示
        real_attn -= np.min(real_attn)
        real_attn /= (np.max(real_attn) + 1e-7)

        return cam, real_attn, class_idx.item(), output

# 工具函数

def get_classes(data_dir):
    dataset = datasets.ImageFolder(data_dir)
    return dataset.classes

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")
    img = cv2.resize(img, (224, 224))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor = transform(img_rgb).unsqueeze(0)
    return img, tensor

def overlay_heatmap(img, mask, title, save_path):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    img_norm = np.float32(img) / 255

    cam_img = 0.6 * img_norm + 0.4 * heatmap
    cam_img = np.uint8(255 * cam_img)

    cv2.putText(cam_img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (255, 255, 255), 2)
    
    cv2.imwrite(save_path, cam_img)
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', help='Test image path')
    parser.add_argument('--model_path', required=True, help='Path to best.pth')
    parser.add_argument('--data_dir', default='dataset', help='Dataset dir for class names')
    parser.add_argument('--model_type', default='resnet18', choices=['resnet18', 'mobilenet_v2'])
    parser.add_argument('--output_prefix', default='result', help='Prefix for output images')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 获取类别
    try:
        class_names = get_classes(args.data_dir)
    except:
        print(f"Warning: Cannot find dataset at {args.data_dir}. Using dummy classes.")
        class_names = [str(i) for i in range(100)] # Fallback

    # 2. 初始化模型
    # 注意：这里我们默认 learnable=True 初始化，这样模型会创建 nn.Parameter
    # 即使训练时是 learnable=False (buffer)，load_state_dict 只要形状匹配通常能兼容，
    # 或者如果不匹配，我们可以尝试 strict=False
    model = GuidedModel(args.model_type, len(class_names), learnable=True).to(device)

    # 3. 加载权重
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        state_dict = checkpoint['model_state_dict']
        
        # 尝试加载
        keys = model.load_state_dict(state_dict, strict=False)
        print(f"Model loaded. Missing keys: {keys.missing_keys}, Unexpected: {keys.unexpected_keys}")
        model.eval()
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

    # 4. 设置 Grad-CAM 目标层 (卷积特征的最后一层)
    if args.model_type == 'resnet18':
        # ResNet features 是 Sequential，最后一层是 block
        target_layer = model.features[-1]
    else:
        # MobileNet features 也是 Sequential
        target_layer = model.features[-1]

    grad_cam = GradCAM(model, target_layer)

    # 5. 推理
    orig_img, tensor = preprocess_image(args.image_path)
    tensor = tensor.to(device)

    # 获取 Grad-CAM 和 真实的 Spatial Attention Map
    cam_mask, att_mask, pred_idx, output = grad_cam(tensor)

    probs = torch.softmax(output, dim=1)
    conf = probs[0][pred_idx].item()
    pred_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)

    print(f"Prediction: {pred_name} ({conf:.2%})")

    # 6. 保存结果
    # 结果 1: Grad-CAM (模型依靠哪些特征做出的分类决定？)
    overlay_heatmap(orig_img, cam_mask, 
                   f"GradCAM: {pred_name} {conf:.1%}", 
                   f"{args.output_prefix}_gradcam.jpg")

    # 结果 2: Spatial Attention (模型实际被引导关注哪里？)
    # 这张图应该显示出边缘高亮（如果你用了 hard/smooth mask 训练）
    overlay_heatmap(orig_img, att_mask, 
                   f"Attention Map (Guidance)", 
                   f"{args.output_prefix}_attention.jpg")

if __name__ == '__main__':
    main()