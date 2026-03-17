import argparse
import yaml
import torch
import time
import numpy as np
import importlib
import os
import sys

# Thêm đường dẫn hiện tại vào path để load được modules
sys.path.append(os.getcwd())

def import_class(name):
    try:
        module_name, class_name = name.rsplit('.', 1)
        mod = importlib.import_module(module_name)
        klass = getattr(mod, class_name)
        return klass
    except Exception as e:
        raise ImportError(f"Cannot load class {name}: {e}")

def count_parameters(model):
    """Đếm tổng số tham số và tham số train được"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def measure_latency(model, input_tensor, device, iterations=100, warmup=20):
    """Đo độ trễ và FPS"""
    model.eval()
    
    # 1. Warmup (Làm nóng GPU/CPU để cache hoạt động ổn định)
    print(f"[INFO] Warming up ({warmup} iterations)...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize()

    # 2. Benchmark Loop
    print(f"[INFO] Benchmarking ({iterations} iterations)...")
    times = []
    with torch.no_grad():
        for _ in range(iterations):
            # Bắt đầu bấm giờ
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()
            
            # Inference
            _ = model(input_tensor)
            
            # Kết thúc bấm giờ
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            
            times.append(end_time - start_time)

    avg_time = np.mean(times)
    std_time = np.std(times)
    fps = 1.0 / avg_time
    
    return avg_time, std_time, fps

def main():
    parser = argparse.ArgumentParser(description='Benchmark CTR-GCN Performance')
    parser.add_argument('--config', default='./configs/transfer_joint.yaml', help='Đường dẫn file config')
    parser.add_argument('--device', default=0, type=int, help='GPU ID (để -1 nếu dùng CPU)')
    args = parser.parse_args()

    # 1. Load Config
    if not os.path.exists(args.config):
        print(f"[ERROR] Không tìm thấy config: {args.config}")
        return

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Setup Device
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f"cuda:{args.device}")
        print(f"[INFO] Using Device: {torch.cuda.get_device_name(args.device)}")
    else:
        device = torch.device("cpu")
        print("[INFO] Using Device: CPU")

    # 3. Build Model
    print("[INFO] Building Model...")
    Model = import_class(config['model'])
    model = Model(**config['model_args']).to(device)

    # 4. Measure Parameters
    total_params, trainable_params = count_parameters(model)
    
    # 5. Prepare Dummy Input
    # Cấu trúc Input CTR-GCN: (Batch, Channel, Frames, Joints, Person)
    # Lấy thông số từ Config hoặc mặc định
    in_channels = config['model_args'].get('in_channels', 3)
    num_point = config['model_args'].get('num_point', 46) # VSL là 46
    num_person = config['model_args'].get('num_person', 1)
    
    # Tìm window size (độ dài video) trong config feeder
    window_size = 64 # Mặc định
    if 'train_feeder_args' in config and 'window_size' in config['train_feeder_args']:
        window_size = config['train_feeder_args']['window_size']
    elif 'val_feeder_args' in config and 'window_size' in config['val_feeder_args']:
        window_size = config['val_feeder_args']['window_size']
        
    batch_size = 1 # Benchmark độ trễ thực tế thì batch luôn là 1
    
    dummy_input = torch.randn(batch_size, in_channels, window_size, num_point, num_person).to(device)
    
    print("-" * 50)
    print(f" INPUT SHAPE: {dummy_input.shape}")
    print(f" (Batch: {batch_size}, C: {in_channels}, T: {window_size}, V: {num_point}, M: {num_person})")
    print("-" * 50)

    # 6. Thử tính FLOPs (Nếu có thư viện thop)
    try:
        from thop import profile
        print("[INFO] Calculating FLOPs using 'thop'...")
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        flops_str = f"{flops / 1e9:.2f} G"
    except ImportError:
        flops_str = "N/A (Cài 'pip install thop' để xem)"
    except Exception as e:
        flops_str = f"Error: {e}"

    # 7. Run Benchmark
    avg_latency, std_latency, fps = measure_latency(model, dummy_input, device)

    # 8. Print Results Table
    print("\n" + "="*40)
    print(f"  MODEL PERFORMANCE REPORT")
    print("="*40)
    print(f"  Config file   : {args.config}")
    print(f"  Device        : {device}")
    print("-" * 40)
    print(f"  Parameters : {total_params / 1e6:.2f} M")
    print(f"  FLOPs      : {flops_str}")
    print("-" * 40)
    print(f"  Latency    : {avg_latency * 1000:.2f} ms ± {std_latency*1000:.2f} ms")
    print(f"  FPS        : {fps:.2f} frames/sec")
    print("="*40 + "\n")

    # Giải thích
    print("[NOTE] Giải thích:")
    print(" - Parameters (M): Triệu tham số. Càng ít model càng nhẹ.")
    print(" - Latency (ms): Thời gian xử lý 1 video. Càng thấp càng tốt (Realtime cần < 33ms).")
    print(" - FPS: Số lượng video xử lý được trong 1 giây.")

if __name__ == '__main__':
    main()