import os
import cv2
import mss
import time
import argparse
import numpy as np
import onnxruntime as ort
import signal
import sys
from collections import deque, Counter

def get_args():
    parser = argparse.ArgumentParser(description="Real-time ONNX Inference Monitor")
    parser.add_argument('--model', type=str, default='resnet18.onnx', help='Path to ONNX model')
    parser.add_argument('--dataset', type=str, default='dataset', help='Path to dataset directory')
    parser.add_argument('--monitor', type=int, default=1, help='Monitor index (default: 1)')
    parser.add_argument('--window_size', type=int, default=5, help='Smoothing window size')
    parser.add_argument('--top_most', action='store_true', default=True, help='Keep window always on top')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use (cuda or cpu)')
    return parser.parse_args()

class InferenceEngine:
    def __init__(self, model_path, device='cuda'):
        print(f"Loading model: {model_path}")
        try:
            if device.lower() == 'cpu':
                providers = ['CPUExecutionProvider']
            else:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            self.actual_provider = self.session.get_providers()[0]
            print(f"[INFO] 实际使用的设备 Provider: {self.actual_provider}")
            
            if device.lower() == 'cuda' and self.actual_provider == 'CPUExecutionProvider':
                print("[WARN] 已指定使用 CUDA，但 ONNX Runtime 未检测到可用 GPU 环境，已回退到 CPU。")

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def preprocess(self, img_bgr):
        img = cv2.resize(img_bgr, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)
        return np.expand_dims(img, axis=0)

    def predict(self, img_bgr):
        input_tensor = self.preprocess(img_bgr)
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        scores = outputs[0][0]
        probs = np.exp(scores) / np.sum(np.exp(scores)) 
        return probs

class ResultSmoother:
    def __init__(self, window_size=5):
        self.queue = deque(maxlen=window_size)
    
    def update(self, idx):
        self.queue.append(idx)
        return Counter(self.queue).most_common(1)[0][0]

def save_report(latencies, provider, model_name):
    """保存统计结果到 txt 文件的函数"""
    if not latencies:
        print("\n[INFO] 无统计数据，未保存。")
        return
        
    lat_array = np.array(latencies)
    avg_ms = np.mean(lat_array) * 1000
    min_ms = np.min(lat_array) * 1000
    max_ms = np.max(lat_array) * 1000
    avg_fps = 1.0 / np.mean(lat_array)
    
    summary = (
        f"--- ONNX Inference Summary ---\n"
        f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Device Provider: {provider}\n"
        f"Model: {model_name}\n"
        f"Total Frames: {len(latencies)}\n"
        f"Average Speed: {avg_fps:.2f} FPS\n"
        f"Average Latency: {avg_ms:.2f} ms\n"
        f"Min Latency: {min_ms:.2f} ms\n"
        f"Max Latency: {max_ms:.2f} ms\n"
        f"------------------------------\n\n"
    )
    
    with open("speed_summary.txt", "a") as f:
        f.write(summary)
    print(f"\n[INFO] 统计结果已保存至 speed_summary.txt")

def main():
    args = get_args()
    
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset directory '{args.dataset}' not found.")
        return
        
    classes = sorted([
        d for d in os.listdir(args.dataset) 
        if os.path.isdir(os.path.join(args.dataset, d))
    ])
    
    if not classes:
        print(f"Error: No subdirectories found in '{args.dataset}'.")
        return
        
    print(f"Loaded {len(classes)} classes: {classes}")

    engine = InferenceEngine(args.model, device=args.device)
    smoother = ResultSmoother(args.window_size)
    
    latencies = []
    
    # 状态标志，用于平滑退出
    state = {'running': True}

    def signal_handler(sig, frame):
        print("\n[INFO] 捕获到终止信号 (Ctrl+C)，正在保存结果并退出...")
        state['running'] = False

    # 注册信号
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    with mss.mss() as sct:
        monitor_rect = sct.monitors[args.monitor]
        window_name = "AI Monitor (ONNX)"
        
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 400, 120)
        
        if args.top_most:
            try:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
            except:
                pass

        print("运行中... 按下 'q' 或 'Ctrl+C' 退出并保存结果。")
        
        fps_avg = 0.0
        alpha = 0.1
        
        while state['running']:
            t0 = time.perf_counter()
            
            try:
                screenshot = sct.grab(monitor_rect)
                img_bgr = np.array(screenshot)[:, :, :3]
                
                probs = engine.predict(img_bgr)
                raw_idx = np.argmax(probs)
                confidence = probs[raw_idx]
                
                final_idx = smoother.update(raw_idx)
                label = classes[final_idx] if final_idx < len(classes) else "Unknown"
                
                t_end = time.perf_counter()
                elapsed = t_end - t0
                latencies.append(elapsed)
                
                current_fps = 1.0 / elapsed if elapsed > 0 else 0
                fps_avg = (alpha * current_fps) + (1.0 - alpha) * fps_avg
                
                panel = np.zeros((120, 400, 3), dtype=np.uint8)
                color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
                
                cv2.putText(panel, f"PRED: {label}", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
                cv2.putText(panel, f"Conf: {confidence:.2f} | FPS: {fps_avg:.1f}", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                cv2.imshow(window_name, panel)
                
                # 检查 OpenCV 窗口的退出按键
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n[INFO] 按下 'q' 键，退出程序...")
                    break
            except Exception as e:
                print(f"[ERROR] 循环中发生异常: {e}")
                break

    # 统一执行保存
    save_report(latencies, engine.actual_provider, os.path.basename(args.model))
    cv2.destroyAllWindows()
    sys.exit(0)

if __name__ == "__main__":
    main()