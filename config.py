"""
Centralized Configuration Module for HorizonSeg-TRT.
All hyper-parameters, hardware toggles, and dataset mappings are defined here.
"""
import os
import torch

class Config:
    # Project Details
    PROJECT_NAME = "HorizonSeg-TRT"
    SEED = 42

    # Dataset & Classes
    NUM_CLASSES = 10
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256
    CLASS_NAMES = [
        "Background", "Road", "Sidewalk", "Building", "Wall", 
        "Fence", "Pole", "Traffic Light", "Traffic Sign", "Vegetation"
    ]

    # Paths
    DATA_DIR = "./data"
    TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, "train/images")
    TRAIN_MASKS_DIR = os.path.join(DATA_DIR, "train/masks")
    VAL_IMAGES_DIR = os.path.join(DATA_DIR, "val/images")
    VAL_MASKS_DIR = os.path.join(DATA_DIR, "val/masks")

    # Training Configuration
    BATCH_SIZE = 16
    NUM_EPOCHS = 15
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Optimization & Pruning Configuration
    PRUNING_RATIO = 0.30  # Prune 30% of Conv weights
    FINE_TUNE_EPOCHS = 5  # Fine-tune epochs after pruning

    # Export & Checkpoints
    CHECKPOINT_PATH = "best_model.pth"
    PRUNED_CHECKPOINT_PATH = "best_model_pruned.pth"
    ONNX_MODEL_PATH = "best_model_optimized.onnx"

    # Hardware Toggles
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_AMP = True  # Automatic Mixed Precision for FP16 execution speedups
    NUM_WORKERS = 4
