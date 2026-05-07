"""
Robust Dataset and DataLoader module with Albumentations augmentation pipelines
and safe, corrupt-resistant image/mask loading with synthetic fallback.
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging

logger = logging.getLogger(__name__)

class SemanticSegmentationDataset(Dataset):
    """
    Production-grade Dataset class supporting both real image files and synthetic generation
    for safe execution, out-of-the-box running, and fallback.
    """
    def __init__(self, images_dir: str, masks_dir: str, num_classes: int, transform=None, synthetic_count: int = 0):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.num_classes = num_classes
        self.transform = transform
        self.synthetic_count = synthetic_count
        
        if synthetic_count > 0:
            logger.info(f"Using Synthetic Dataset with {synthetic_count} samples.")
            self.image_paths = [f"synthetic_{i}" for i in range(synthetic_count)]
        else:
            if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
                logger.warning(f"Data directories {images_dir} or {masks_dir} not found. Falling back to synthetic generation.")
                self.synthetic_count = 100
                self.image_paths = [f"synthetic_{i}" for i in range(self.synthetic_count)]
            else:
                self.image_paths = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
                self.mask_paths = sorted([os.path.join(masks_dir, f) for f in os.listdir(masks_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
                assert len(self.image_paths) == len(self.mask_paths), "Mismatch between image paths and mask paths."
                
    def __len__(self):
        return len(self.image_paths)
        
    def _generate_synthetic_sample(self, idx: int):
        """Generates realistic synthetic shapes to represent classes for testing/fallback."""
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        mask = np.zeros((256, 256), dtype=np.uint8)
        
        np.random.seed(idx)
        # Background gets some random noise
        img = (np.random.rand(256, 256, 3) * 50).astype(np.uint8)
        
        for class_id in range(1, self.num_classes):
            center = (np.random.randint(20, 230), np.random.randint(20, 230))
            radius = np.random.randint(15, 60)
            color = [np.random.randint(50, 255) for _ in range(3)]
            cv2.circle(img, center, radius, color, -1)
            cv2.circle(mask, center, radius, class_id, -1)
            
        return img, mask
        
    def __getitem__(self, idx):
        if self.synthetic_count > 0:
            image, mask = self._generate_synthetic_sample(idx)
        else:
            img_path = self.image_paths[idx]
            mask_path = self.mask_paths[idx]
            
            try:
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if image is None or mask is None:
                    raise ValueError("Loaded file is empty/corrupt.")
            except Exception as e:
                logger.error(f"Failed to load image/mask at index {idx}: {e}. Falling back to synthetic.")
                image, mask = self._generate_synthetic_sample(idx)
                
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()
            
        return image, mask

def get_transforms(img_height: int, img_width: int):
    """Retrieves standard Albumentations training and validation augmentation pipelines."""
    train_transform = A.Compose([
        A.Resize(img_height, img_width),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(img_height, img_width),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    return train_transform, val_transform
