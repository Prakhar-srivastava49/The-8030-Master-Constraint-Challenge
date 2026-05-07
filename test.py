"""
Performance Benchmarking and Validation Module.
Benchmarks latency with time.perf_counter() and evaluates complete metrics
including Mean IoU and Confusion Matrix across PyTorch and ONNX engines.
"""
import sys
import io
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import os
import time
import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from dataset import SemanticSegmentationDataset, get_transforms

# Set up Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class ConfusionMatrix:
    """Manages computation of the multi-class Confusion Matrix and Mean IoU."""
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        
    def update(self, preds: np.ndarray, targets: np.ndarray):
        assert preds.shape == targets.shape, "Prediction and ground truth shapes must match."
        mask = (targets >= 0) & (targets < self.num_classes)
        hist = np.bincount(
            self.num_classes * targets[mask].astype(int) + preds[mask].astype(int),
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        self.matrix += hist
        
    def get_metrics(self):
        diag = np.diag(self.matrix)
        row_sum = np.sum(self.matrix, axis=1)
        col_sum = np.sum(self.matrix, axis=0)
        
        union = row_sum + col_sum - diag
        class_iou = np.zeros(self.num_classes)
        valid_indices = union > 0
        class_iou[valid_indices] = diag[valid_indices] / union[valid_indices]
        
        pixel_acc = diag.sum() / self.matrix.sum() if self.matrix.sum() > 0 else 0.0
        mean_iou = np.mean(class_iou)
        
        return mean_iou, class_iou, pixel_acc

def benchmark_pytorch(loader, device):
    """Benchmarks baseline PyTorch model speed and IoU."""
    from model import MobileNetV3UNet
    model = MobileNetV3UNet(num_classes=Config.NUM_CLASSES, pretrained=False).to(device)
    if os.path.exists(Config.CHECKPOINT_PATH):
        model.load_state_dict(torch.load(Config.CHECKPOINT_PATH, map_location=device))
        logger.info("Loaded PyTorch Baseline Checkpoint.")
    else:
        logger.warning("Baseline PyTorch checkpoint not found. Running with random weights.")
        
    model.eval()
    cf = ConfusionMatrix(Config.NUM_CLASSES)
    
    # Warmup
    dummy = torch.randn(1, 3, Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)
            
    latencies = []
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            
            # Record exact latency
            start_time = time.perf_counter()
            outputs = model(images)
            if device == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            latencies.append((end_time - start_time) / images.size(0))
            
            preds = outputs.argmax(dim=1).cpu().numpy()
            cf.update(preds, masks.numpy())
            
    avg_latency = np.mean(latencies) * 1000  # ms
    mean_iou, class_iou, pixel_acc = cf.get_metrics()
    
    return avg_latency, mean_iou, class_iou, pixel_acc, cf.matrix

def benchmark_onnx(loader):
    """Benchmarks exported ONNX Runtime model speed and IoU with TensorRT/CUDA."""
    import onnxruntime as ort
    if not os.path.exists(Config.ONNX_MODEL_PATH):
        raise FileNotFoundError(f"ONNX model file not found at {Config.ONNX_MODEL_PATH}. Run training first.")
        
    providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(Config.ONNX_MODEL_PATH, providers=providers)
    logger.info(f"Loaded ONNX session with providers: {session.get_providers()}")
    
    input_name = session.get_inputs()[0].name
    cf = ConfusionMatrix(Config.NUM_CLASSES)
    
    # Warmup
    dummy = np.random.randn(1, 3, Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH).astype(np.float32)
    for _ in range(10):
        _ = session.run(None, {input_name: dummy})
        
    latencies = []
    for images, masks in loader:
        img_np = images.numpy()
        
        start_time = time.perf_counter()
        outputs = session.run(None, {input_name: img_np})
        end_time = time.perf_counter()
        
        latencies.append((end_time - start_time) / images.size(0))
        
        preds = np.argmax(outputs[0], axis=1)
        cf.update(preds, masks.numpy())
        
    avg_latency = np.mean(latencies) * 1000  # ms
    mean_iou, class_iou, pixel_acc = cf.get_metrics()
    
    return avg_latency, mean_iou, class_iou, pixel_acc, cf.matrix

def print_results(engine_name, latency, m_iou, class_iou, pixel_acc, matrix):
    print("\n" + "="*50)
    print(f" RESULTS FOR {engine_name.upper()} ENGINE ")
    print("="*50)
    print(f"Average Inference Latency: {latency:.2f} ms per image  " + ("(PASSED < 30ms Limit)" if latency < 30 else "(FAILED Limit)"))
    print(f"Overall Mean IoU (mIoU)  : {m_iou * 100:.2f}%  " + ("(PASSED >= 80% Threshold)" if m_iou >= 0.80 else "(FAILED Threshold)"))
    print(f"Overall Pixel Accuracy   : {pixel_acc * 100:.2f}%")
    print("-" * 50)
    print("Class-wise IoUs:")
    for i, score in enumerate(class_iou):
        print(f"  Class {i:02d} ({Config.CLASS_NAMES[i]:<15}): {score * 100:.2f}%")
    print("-" * 50)
    print("Confusion Matrix:")
    print(matrix)
    print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser(description="HorizonSeg-TRT Inference Benchmarking Suite")
    parser.add_argument("--engine", choices=["pytorch", "onnx", "both"], default="both", help="Inference engine backend")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic dataset generation")
    args = parser.parse_args()
    
    _, val_transform = get_transforms(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH)
    synthetic_flag = 64 if args.synthetic or not os.path.exists(Config.VAL_IMAGES_DIR) else 0
    
    val_dataset = SemanticSegmentationDataset(
        Config.VAL_IMAGES_DIR, Config.VAL_MASKS_DIR, Config.NUM_CLASSES, 
        transform=val_transform, synthetic_count=synthetic_flag
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    if args.engine in ["pytorch", "both"]:
        latency, m_iou, class_iou, pixel_acc, matrix = benchmark_pytorch(val_loader, Config.DEVICE)
        print_results("PyTorch (Standard CPU/GPU)", latency, m_iou, class_iou, pixel_acc, matrix)
        
    if args.engine in ["onnx", "both"]:
        try:
            latency, m_iou, class_iou, pixel_acc, matrix = benchmark_onnx(val_loader)
            print_results("ONNX Runtime (with TensorRT/CUDA)", latency, m_iou, class_iou, pixel_acc, matrix)
        except Exception as e:
            logger.error(f"Failed to benchmark ONNX engine: {e}. Please ensure you have run train.py to generate the ONNX file.")

if __name__ == "__main__":
    main()
