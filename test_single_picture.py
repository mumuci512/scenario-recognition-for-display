import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from PIL import Image
import argparse
import os
import sys
import cv2
import numpy as np

# 类别获取逻辑
def get_classes(data_dir):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}. 无法自动获取类别。")
    # 使用 ImageFolder 自动识别子文件夹名称作为类别
    dataset = datasets.ImageFolder(data_dir)
    return dataset.classes

# Grad-CAM Logic
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        # Forward pass
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)

        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Generate CAM
        gradients = self.gradients[0].cpu().detach().numpy()
        activations = self.activations[0].cpu().detach().numpy()
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam -= np.min(cam)
        cam /= (np.max(cam) + 1e-7)

        return cam

# --- 修改点 1: 增加 title 参数并绘制文字 ---
def save_heatmap(img_path, mask, save_path, title=None):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read image for heatmap overlay: {img_path}")
        return
        
    img = cv2.resize(img, (224, 224))
    
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    img_norm = np.float32(img) / 255

    cam_img = 0.6 * img_norm + 0.4 * heatmap
    cam_img = np.uint8(255 * cam_img)
    
    #在此处添加文字
    if title:
        cv2.putText(cam_img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 255, 255), 2)

    cv2.imwrite(save_path, cam_img)
    print(f"Heatmap saved to: {save_path}")

# --- Functions ---

def get_args():
    parser = argparse.ArgumentParser(description='ResNet18 Inference Script with Auto-Classes')
    parser.add_argument('--image_path', required=True, help='Path to the image file')
    parser.add_argument('--model', default='resnet18_best.pth', help='Path to the model weights')
    parser.add_argument('--data_dir', default='dataset', help='Path to dataset for class names')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'], help='Device to use')
    parser.add_argument('--output', default=None, help='Path to save the heatmap output')
    return parser.parse_args()

def load_model(model_path, num_classes, device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at: {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    # 兼容性处理：如果 checkpoint 包含 model_state_dict 键
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def predict(model, image_path, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, 1)

    return pred_idx.item(), conf.item(), probs[0], img_tensor

def main():
    args = get_args()

    # 设备配置
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    try:
        # 1. 自动获取类别
        classes = get_classes(args.data_dir)
        num_classes = len(classes)
        print(f"Classes found from '{args.data_dir}': {num_classes}")

        # 2. 加载模型
        model = load_model(args.model, num_classes, device)
        
        # 3. 推理
        pred_idx, conf, all_probs, input_tensor = predict(model, args.image_path, device)

        # 输出结果
        print(f"Image: {args.image_path}")
        result_str = f"{classes[pred_idx]} ({conf:.2%})" # 准备文字字符串
        print(f"Result: {result_str}")
        print("-" * 30)
        for i, prob in enumerate(all_probs):
            print(f"{classes[i]:<12}: {prob:.4f}")

        # --- Generate Heatmap ---
        # 启用梯度计算用于 Grad-CAM
        model.zero_grad()
        
        # 自动定位 ResNet18 的最后一层卷积层
        target_layer = model.layer4[-1]
        grad_cam = GradCAM(model, target_layer)

        # 运行 Grad-CAM (传入推理时已处理好的 tensor)
        mask = grad_cam(input_tensor, class_idx=pred_idx)

        # 确定保存路径
        if args.output:
            save_name = args.output
        else:
            filename = os.path.basename(args.image_path)
            save_name = f"heatmap_{filename}"
            
        # --- 修改点 2: 传入文字 ---
        save_heatmap(args.image_path, mask, save_name, title=result_str)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()