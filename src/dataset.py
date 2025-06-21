from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report
import albumentations as A


class InteriorDataset(Dataset):
    """Датасет с поддержкой Albumentations аугментаций"""
    
    CLASSES = ["A0", "A1", "B0", "B1", "C0", "C1", "D0", "D1"]
    
    def __init__(self, samples, transform=None, mode='train'):
        self.samples = samples
        self.transform = transform
        self.mode = mode
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, class_idx

    @classmethod
    def find_samples(cls, dataset_dir: Path) -> List[Tuple[Path, int]]:
        """Collect samples with both path and numerical index.
        
        Args:
            dataset_dir: Root directory containing class folders
            
        Returns:
            List of (image_path, class_index) tuples
        """
        samples = []
        for class_dir in dataset_dir.iterdir():
            if class_dir.is_dir() and class_dir.name in cls.CLASS_TO_IDX:
                class_idx = cls.CLASS_TO_IDX[class_dir.name]
                samples.extend([
                    (img_path, class_idx)
                    for img_path in class_dir.glob("*.jpg")
                    if img_path.is_file()
                ])
        return samples


def get_train_val_test_split(data_dir, test_size=0.15, val_size=0.15, random_state=42):
    """Разделение данных на train/val/test с сохранением пропорций классов"""
    samples = []
    for class_idx, class_name in enumerate(InteriorDataset.CLASSES):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            continue
            
        for img_path in class_dir.glob("*.*"):
            if img_path.suffix.lower() in (".jpg", ".png", ".jpeg"):
                samples.append((img_path, class_idx))
    
    # Первичное разделение на train+val и test
    train_val_samples, test_samples = train_test_split(
        samples, test_size=test_size, stratify=[s[1] for s in samples], random_state=random_state
    )
    
    # Разделение train и val
    train_samples, val_samples = train_test_split(
        train_val_samples, 
        test_size=val_size/(1-test_size), 
        stratify=[s[1] for s in train_val_samples],
        random_state=random_state
    )
    
    return train_samples, val_samples, test_samples

def get_transforms(mode='train', img_size=380):
    """Аугментации для разных этапов"""
    if mode == 'train':
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:  # val/test
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

def create_datasets(data_dir):
    """Создание датасетов с аугментациями"""
    train_samples, val_samples, test_samples = get_train_val_test_split(data_dir)
    
    train_ds = InteriorDataset(
        train_samples,
        transform=get_transforms(mode='train'),
        mode='train'
    )
    
    val_ds = InteriorDataset(
        val_samples,
        transform=get_transforms(mode='val'),
        mode='val'
    )
    
    test_ds = InteriorDataset(
        test_samples,
        transform=get_transforms(mode='test'),
        mode='test'
    )
    
    return train_ds, val_ds, test_ds






