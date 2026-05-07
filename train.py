"""
Training Pipeline for HorizonSeg-TRT.
Features Hybrid Focal/Jaccard Loss, AMP, Cosine Annealing, 
L1-Unstructured Pruning, and ONNX Export with Dynamic Opset.
"""
import sys
import io
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import os
import random
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader

from config import Config
from dataset import SemanticSegmentationDataset, get_transforms
from model import MobileNetV3UNet

# Set up Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """Ensures deterministic behavior across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class JaccardLoss(nn.Module):
    def __init__(self, num_classes: int, smooth: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        intersection = torch.sum(probs * targets_one_hot, dim=(2, 3))
        total = torch.sum(probs + targets_one_hot, dim=(2, 3))
        union = total - intersection
        
        jaccard = (intersection + self.smooth) / (union + self.smooth)
        return 1.0 - jaccard.mean()

class HybridLoss(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.focal = FocalLoss()
        self.jaccard = JaccardLoss(num_classes)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.focal(inputs, targets) + 0.5 * self.jaccard(inputs, targets)

def calculate_miou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """Calculates Mean IoU over a batch of predictions and targets."""
    preds_flat = preds.argmax(dim=1).view(-1)
    targets_flat = targets.view(-1)
    
    ious = []
    for cls in range(num_classes):
        pred_mask = (preds_flat == cls)
        target_mask = (targets_flat == cls)
        
        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()
        
        if union > 0:
            ious.append(intersection / union)
            
    return np.mean(ious) if ious else 0.0

def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device, num_classes):
    model.train()
    total_loss, total_miou = 0.0, 0.0
    
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        
        if Config.USE_AMP and device == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss = loss_fn(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            
        total_loss += loss.item()
        total_miou += calculate_miou(outputs, masks, num_classes)
        
    return total_loss / len(loader), total_miou / len(loader)

def validate(model, loader, loss_fn, device, num_classes):
    model.eval()
    total_loss, total_miou = 0.0, 0.0
    
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            total_loss += loss.item()
            total_miou += calculate_miou(outputs, masks, num_classes)
            
    return total_loss / len(loader), total_miou / len(loader)

def apply_pruning(model: nn.Module, amount: float):
    """Applies global L1 unstructured pruning to all convolutional weights."""
    logger.info(f"Applying L1 Unstructured Pruning with factor {amount*100:.1f}% to Conv2d layers.")
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)

def finalize_pruning(model: nn.Module):
    """Bakes pruning sparse weights permanently, removing PyTorch masking overhead."""
    logger.info("Baking sparse weights and removing pruning reparametrization.")
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if hasattr(module, 'weight_orig'):
                prune.remove(module, 'weight')

def export_onnx(model: nn.Module, path: str):
    """Exports model to optimized ONNX format with dynamic batching."""
    logger.info(f"Exporting optimized model to ONNX: {path}")
    model.eval()
    dummy_input = torch.randn(1, 3, Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH).to(Config.DEVICE)
    torch.onnx.export(
        model,
        dummy_input,
        path,
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    logger.info("ONNX Export completed successfully.")

def main():
    parser = argparse.ArgumentParser(description="HorizonSeg-TRT Training & Pruning Pipeline")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic dataset generation")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--fine-tune-epochs", type=int, default=None, help="Override fine-tune epochs")
    args = parser.parse_args()
    
    set_seed(Config.SEED)
    
    # 1. Dataset Setup
    train_transform, val_transform = get_transforms(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH)
    
    synthetic_flag = 128 if args.synthetic or not os.path.exists(Config.TRAIN_IMAGES_DIR) else 0
    
    train_dataset = SemanticSegmentationDataset(
        Config.TRAIN_IMAGES_DIR, Config.TRAIN_MASKS_DIR, Config.NUM_CLASSES, 
        transform=train_transform, synthetic_count=synthetic_flag
    )
    val_dataset = SemanticSegmentationDataset(
        Config.VAL_IMAGES_DIR, Config.VAL_MASKS_DIR, Config.NUM_CLASSES, 
        transform=val_transform, synthetic_count=(64 if synthetic_flag else 0)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True)
    
    # 2. Model, Optimizer, Loss Setup
    model = MobileNetV3UNet(num_classes=Config.NUM_CLASSES, pretrained=True).to(Config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    
    num_epochs = args.epochs if args.epochs is not None else Config.NUM_EPOCHS
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    loss_fn = HybridLoss(Config.NUM_CLASSES)
    scaler = torch.amp.GradScaler(enabled=Config.USE_AMP)
    
    # 3. Baseline Training Phase
    logger.info("--- Phase 1: Baseline Model Training ---")
    best_iou = 0.0
    for epoch in range(1, num_epochs + 1):
        train_loss, train_iou = train_one_epoch(model, train_loader, optimizer, loss_fn, scaler, Config.DEVICE, Config.NUM_CLASSES)
        val_loss, val_iou = validate(model, val_loader, loss_fn, Config.DEVICE, Config.NUM_CLASSES)
        scheduler.step()
        
        logger.info(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f}, mIoU: {train_iou:.4f} | Val Loss: {val_loss:.4f}, mIoU: {val_iou:.4f}")
        
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), Config.CHECKPOINT_PATH)
            logger.info(f"Saved Best Baseline Checkpoint with mIoU: {best_iou:.4f}")
            
    # 4. Pruning-Aware Fine-Tuning Phase (PAFT)
    logger.info("--- Phase 2: Pruning & Fine-Tuning ---")
    
    # Check if we saved a baseline model, otherwise save current
    if not os.path.exists(Config.CHECKPOINT_PATH):
        torch.save(model.state_dict(), Config.CHECKPOINT_PATH)
        
    model.load_state_dict(torch.load(Config.CHECKPOINT_PATH))
    apply_pruning(model, Config.PRUNING_RATIO)
    
    # Re-initialize optimizer for fine-tuning
    ft_optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE * 0.1, weight_decay=Config.WEIGHT_DECAY)
    best_pruned_iou = 0.0
    fine_tune_epochs = args.fine_tune_epochs if args.fine_tune_epochs is not None else Config.FINE_TUNE_EPOCHS
    
    for epoch in range(1, fine_tune_epochs + 1):
        train_loss, train_iou = train_one_epoch(model, train_loader, ft_optimizer, loss_fn, scaler, Config.DEVICE, Config.NUM_CLASSES)
        val_loss, val_iou = validate(model, val_loader, loss_fn, Config.DEVICE, Config.NUM_CLASSES)
        
        logger.info(f"PAFT Epoch {epoch:02d} | Train Loss: {train_loss:.4f}, mIoU: {train_iou:.4f} | Val Loss: {val_loss:.4f}, mIoU: {val_iou:.4f}")
        
        if val_iou > best_pruned_iou:
            best_pruned_iou = val_iou
            torch.save(model.state_dict(), Config.PRUNED_CHECKPOINT_PATH)
            logger.info(f"Saved Best Pruned Checkpoint with mIoU: {best_pruned_iou:.4f}")
            
    # 5. Finalize & Export to ONNX
    if not os.path.exists(Config.PRUNED_CHECKPOINT_PATH):
        torch.save(model.state_dict(), Config.PRUNED_CHECKPOINT_PATH)
        
    model.load_state_dict(torch.load(Config.PRUNED_CHECKPOINT_PATH))
    finalize_pruning(model)
    export_onnx(model, Config.ONNX_MODEL_PATH)
    logger.info("Training pipeline and optimization successfully finalized.")

if __name__ == "__main__":
    main()
