import torch
import torch.nn as nn
from torchvision import models
import onnx
import onnxruntime
import numpy as np
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX")
    parser.add_argument('--input', type=str, required=True, help='Path to .pth model file')
    parser.add_argument('--output', type=str, default='model.onnx', help='Output .onnx file path')
    parser.add_argument('--arch', type=str, default='resnet18', choices=['resnet18', 'mobilenet_v2'], help='Model architecture')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes')
    parser.add_argument('--opset', type=int, default=12, help='ONNX opset version')
    return parser.parse_args()

def load_pytorch_model(arch, num_classes, weights_path):
    print(f"Loading {arch} with {num_classes} classes...")
    
    if arch == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    try:
        # map_location ensures we can load CUDA weights on CPU
        state_dict = torch.load(weights_path, map_location='cpu')
        
        # 兼容性处理：如果保存的是整个checkpoint字典，提取state_dict
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
            
        model.load_state_dict(state_dict)
    except Exception as e:
        raise RuntimeError(f"Failed to load weights: {e}")

    model.eval()
    return model

def verify_onnx(torch_model, onnx_path, dummy_input):
    print("Verifying ONNX export consistency...")
    
    # 1. PyTorch inference
    with torch.no_grad():
        torch_out = torch_model(dummy_input).numpy()

    # 2. ONNX Runtime inference
    ort_session = onnxruntime.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_out = ort_session.run(None, ort_inputs)[0]

    # 3. Compare
    try:
        np.testing.assert_allclose(torch_out, ort_out, rtol=1e-03, atol=1e-05)
        print("Success: ONNX output matches PyTorch output.")
        print(f"Sample Output (First 3): {torch_out[0][:3]}")
    except AssertionError as e:
        print("Failure: Outputs do not match.")
        print(f"Max difference: {np.max(np.abs(torch_out - ort_out))}")
        raise e

def main():
    args = get_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return

    # 1. Load Model
    model = load_pytorch_model(args.arch, args.num_classes, args.input)

    # 2. Create Dummy Input
    # [Batch, Channel, Height, Width]
    dummy_input = torch.randn(1, 3, 224, 224)

    # 3. Export
    print(f"Exporting to {args.output}...")
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        # dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} # Uncomment for dynamic batch
    )
    print("Export complete.")

    # 4. Verify
    verify_onnx(model, args.output, dummy_input)

if __name__ == "__main__":
    main()