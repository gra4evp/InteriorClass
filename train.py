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


class InteriorClassifier(nn.Module):
    """Модель для классификации интерьеров"""
    
    def __init__(self, num_classes: int = 8, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b3", 
            pretrained=pretrained,
            num_classes=0  # Без финального слоя
        )
        self.head = nn.Sequential(
            nn.Linear(1536, 512),  # 1536 - выходной размер EfficientNet-B3
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


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
            A.VerticalFlip(p=0.1),
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

def train():
    # Конфигурация
    data_dir = Path("data/processed")
    batch_size = 64
    epochs = 25
    lr = 3e-5
    img_size = 380  # Для EfficientNet-B3
    
    # Создание датасетов
    train_ds, val_ds, test_ds = create_datasets(data_dir)
    
    # DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    # Модель и оптимизатор
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InteriorClassifier().to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader)
    )
    
    # Цикл обучения (аналогично предыдущему, но с тестовой оценкой в конце)
    # ... (ваш код обучения)
    
    # Финальная оценка на тестовом наборе
    model.eval()
    test_preds, test_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    print("\nFinal Test Results:")
    print(classification_report(
        test_labels, test_preds,
        target_names=InteriorDataset.CLASSES,
        digits=4
    ))


if __name__ == "__main__":
    data_dir = Path("data")
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    
    model = train_model(
        train_dir=train_dir,
        val_dir=val_dir,
        epochs=15,
        batch_size=64,
        lr=3e-5
    )
    
    # Сохранение модели
    torch.save(model.state_dict(), "interior_classifier_effnet_b3.pth")