import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report
import timm  # Библиотека для современных архитектур


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
