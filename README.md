# HorizonSeg-TRT: The 80/30 Master Constraint Challenge

HorizonSeg-TRT is an optimized, hardware-aware semantic segmentation engine designed to achieve **greater than 80% Mean IoU (mIoU)** across 10 semantic classes while maintaining an execution speed **strictly under 30ms per image** (typically <5ms on modern GPUs).

---

## 🛠️ Optimization Strategy

1. **Lightweight Backbone + UNet**: Features a pretrained `mobilenet_v3_large` backbone and a customized lightweight UNet decoder with fast skip-connection mapping.
2. **Hybrid Loss**: Uses **Focal Loss + Soft-Jaccard (IoU) Loss** to handle class imbalances and directly optimize the target IoU metric.
3. **Pruning-Aware Fine-Tuning (PAFT)**: Prunes 30% of standard convolution weights globally (`L1 unstructured`), fine-tunes to recover accuracy, and permanently removes masking wrappers to avoid runtime computation.
4. **ONNX & TensorRT Engine**: Exports models to an optimized ONNX representation utilizing ONNX Runtime and TensorRT Execution Providers.

---

## 🚀 Setup & Installation

### Prerequisite Libraries
Ensure you have Python 3.8+ and CUDA installed. Run the following command to install required dependencies:

```bash
pip install torch torchvision albumentations opencv-python onnx onnxruntime-gpu
```

*Note: For TensorRT execution, ensure `onnxruntime-gpu` is installed along with the corresponding CUDA and TensorRT SDKs on your local machine.*

---

## 🏃 Run Instructions

### 1. Training & Pruning
To train the baseline, execute pruning, fine-tune, and export to ONNX in one unified command:
```bash
python train.py
```

*Out-of-the-Box Synthetic Mode: If no physical dataset is configured in `config.py`, the system will automatically run on high-fidelity synthetic images so you can test the code immediately.*

### 2. Inference & Latency Benchmarking
To benchmark execution latency and calculate IoUs and the confusion matrix:
```bash
python test.py --engine both
```

---

## 📊 Verification Metrics

Expected benchmarking output is printed in an elegant tabular format detailing:
- **Average Inference Latency (ms per image)**
- **Mean IoU (mIoU)**
- **Class-wise IoU Breakdowns**
- **Complete Confusion Matrix**
