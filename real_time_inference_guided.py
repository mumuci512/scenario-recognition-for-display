import sys
import os
import argparse
import ctypes
import numpy as np
import cv2
import mss
import torch
import torch.nn as nn
import time
import signal
from torchvision import models
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QShortcut
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QKeySequence

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
        self.register_buffer('center_bias', torch.tensor([0.2]))

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

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(lambda m, i, o: setattr(self, 'activations', o))
        self.target_layer.register_full_backward_hook(lambda m, gi, go: setattr(self, 'gradients', go[0]))

    def __call__(self, x):
        output, _ = self.model(x)
        idx = torch.argmax(output, dim=1)
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        grads = self.gradients[0].cpu().detach().numpy()
        acts = self.activations[0].cpu().detach().numpy()
        weights = np.mean(grads, axis=(1, 2))
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]
        
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam -= cam.min()
        cam /= (cam.max() + 1e-7)
        return cam, idx.item(), output

class InferenceThread(QThread):
    update_signal = pyqtSignal(dict)

    def __init__(self, args, class_names):
        super().__init__()
        self.args = args
        self.class_names = class_names

        if args.device.lower() == 'cpu':
            self.device = torch.device("cpu")
        else:
            # 检查 CUDA 是否可用，若不可用则回退到 CPU 并警告
            if torch.cuda.is_available():
                self.device = torch.device(args.device)
            else:
                print(f"[WARN] CUDA 不可用，尽管指定了 {args.device}，仍将回退到 CPU。")
                self.device = torch.device("cpu")
        
        # 获取设备详细名称用于显示
        if self.device.type == 'cuda':
            self.device_name = f"GPU ({torch.cuda.get_device_name(self.device)})"
        else:
            self.device_name = "CPU"

        print(f"[INFO] 模型推理/反向传播使用设备: {self.device_name} (torch device: {self.device})")

        self.model = GuidedModel(args.arch, len(class_names)).to(self.device)
        
        checkpoint = torch.load(args.model, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        
        target = self.model.features[-1]
        self.grad_cam = GradCAM(self.model, target)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
        self.latencies = []

    def run(self):
        with mss.mss() as sct:
            m_idx = self.args.monitor if self.args.monitor < len(sct.monitors) else 1
            rect = sct.monitors[m_idx]
            
            fps_avg = 0.0
            alpha = 0.1

            while not self.isInterruptionRequested():
                try:
                    start_time = time.perf_counter()

                    img = np.array(sct.grab(rect))[:, :, :3]
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_in = cv2.resize(img_rgb, (224, 224))
                    
                    tensor = torch.from_numpy(img_in.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(self.device)
                    tensor = (tensor - self.mean) / self.std

                    with torch.enable_grad():
                        mask, idx, output = self.grad_cam(tensor)
                    
                    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
                    blended = cv2.addWeighted(cv2.resize(img_in, (240, 240)), 0.6, cv2.resize(heatmap, (240, 240)), 0.4, 0)
                    
                    conf = torch.softmax(output, dim=1)[0][idx].item()

                    end_time = time.perf_counter()
                    elapsed = end_time - start_time
                    self.latencies.append(elapsed)
                    
                    current_fps = 1.0 / elapsed if elapsed > 0 else 0
                    fps_avg = (alpha * current_fps) + (1.0 - alpha) * fps_avg

                    self.update_signal.emit({
                        'pred': self.class_names[idx],
                        'conf': conf,
                        'heatmap': blended,
                        'fps': fps_avg,
                        'ms': elapsed * 1000
                    })
                except Exception:
                    break
        
        self.save_summary()

    def save_summary(self):
        if not self.latencies:
            return
        
        lat_array = np.array(self.latencies)
        avg_ms = np.mean(lat_array) * 1000
        min_ms = np.min(lat_array) * 1000
        max_ms = np.max(lat_array) * 1000
        avg_fps = 1.0 / np.mean(lat_array)
        
        summary = (
            f"Device: {self.device_name} (torch device: {self.device})\n"
            f"Architecture: {self.args.arch}\n"
            f"Model: {os.path.basename(self.args.model)}\n"
            f"Total Frames: {len(self.latencies)}\n"
            f"Average Speed: {avg_fps:.2f} FPS\n"
            f"Average Latency: {avg_ms:.2f} ms\n"
            f"Min Latency: {min_ms:.2f} ms\n"
            f"Max Latency: {max_ms:.2f} ms\n"
        )
        
        # 使用 a 追加模式，防止多次实验数据丢失，或使用 w 覆盖模式
        with open("speed_summary.txt", "a") as f:
            f.write(summary)
        print(f"\n[INFO] Speed summary saved for {self.args.arch}.")

class OverlayWindow(QWidget):
    def __init__(self, pos="left", hide_from_capture=True):
        super().__init__()
        self.hide_from_capture = hide_from_capture
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        screen = QApplication.primaryScreen().geometry()
        w, h = 260, 360
        x = (screen.width() - w - 50) if pos == "right" else 50
        self.setGeometry(x, 50, w, h)

        layout = QVBoxLayout()
        self.lbl_info = QLabel("Initializing...")
        self.lbl_info.setStyleSheet("color: #00FF00; font: bold 14px 'Consolas'; background: rgba(0,0,0,150); padding: 8px;")
        
        self.lbl_speed = QLabel("FPS: 0.0 (0.0ms)")
        self.lbl_speed.setStyleSheet("color: #FFFFFF; font: 12px 'Consolas'; background: rgba(0,0,0,150); padding: 4px 8px;")
        
        self.lbl_img = QLabel()
        layout.addWidget(self.lbl_info)
        layout.addWidget(self.lbl_speed)
        layout.addWidget(self.lbl_img)
        self.setLayout(layout)

        self.apply_window_rules()
        self.shortcut = QShortcut(QKeySequence("F12"), self)
        self.shortcut.activated.connect(self.toggle_capture_policy)

    def apply_window_rules(self):
        hwnd = int(self.winId())
        style = ctypes.windll.user32.GetWindowLongW(hwnd, -20)
        ctypes.windll.user32.SetWindowLongW(hwnd, -20, style | 0x80000 | 0x20)
        affinity = 0x11 if self.hide_from_capture else 0x00
        ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, affinity)

    def toggle_capture_policy(self):
        self.hide_from_capture = not self.hide_from_capture
        ctypes.windll.user32.SetWindowDisplayAffinity(int(self.winId()), 0x11 if self.hide_from_capture else 0x00)

    def update_view(self, data):
        self.lbl_info.setText(f"CLASS: {data['pred']}\nCONF: {data['conf']:.2%}")
        self.lbl_speed.setText(f"FPS: {data['fps']:.1f} ({data['ms']:.1f}ms)")
        img = data['heatmap']
        q_img = QImage(img.data, 240, 240, 240 * 3, QImage.Format_RGB888).rgbSwapped()
        self.lbl_img.setPixmap(QPixmap.fromImage(q_img))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='dataset')
    parser.add_argument('--monitor', type=int, default=1)
    parser.add_argument('--pos', type=str, default='left')
    parser.add_argument('--hide_window', type=str2bool, default=True)
    parser.add_argument('--device', type=str, default='cuda', help="Device to use: 'cuda', 'cpu', or 'cuda:0'")
    args = parser.parse_args()

    classes = sorted([d for d in os.listdir(args.dataset) if os.path.isdir(os.path.join(args.dataset, d))])
    
    app = QApplication(sys.argv)
    overlay = OverlayWindow(args.pos, args.hide_window)
    overlay.show()
    
    thread = InferenceThread(args, classes)
    thread.update_signal.connect(overlay.update_view)
    
    # 定义退出函数
    def handle_exit(signum, frame):
        print("\n[INFO] Termination signal received. Saving results...")
        app.quit()

    # 捕获 Ctrl+C (SIGINT) 和 终止信号 (SIGTERM)
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    
    # 确保正常退出时也能保存
    def on_quit():
        thread.requestInterruption()
        thread.wait()

    app.aboutToQuit.connect(on_quit)
    
    thread.start()
    sys.exit(app.exec_())