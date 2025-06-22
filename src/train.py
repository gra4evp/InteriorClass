import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report



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
    model = InteriorClassifier(num_classes=len(InteriorDataset.CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Цикл обучения
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        scheduler.step()
        
        # Валидация
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Метрики
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        report = classification_report(
            all_labels, all_preds, 
            target_names=InteriorDataset.CLASSES,
            zero_division=0
        )
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(report)
    
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