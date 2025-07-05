# src/train.py
from pathlib import Path
import json

from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import torchvision.transforms as transforms
from PIL import Image

from src.config import RANDOM_SEED, SPLIT_CONFIG, MIN_VAL_TEST_PER_CLASS, CLASS_LABELS
from datasets.utils.collector import DataCollector
from datasets.utils.splitter import DatasetSplitter
from src.datasets.interior_dataset import InteriorDataset, get_transforms
from src.models.interior_classifier_EfficientNet_B3 import InteriorClassifier
from src.trainer import Trainer


if __name__ == "__main__":
    # 1. =========================== Define hyperparameters ===========================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    EPOCHS = 10
    LR = 3e-5
    IMG_SIZE = 448
    EXP_NUMBER = 10


    # 2. ================================ Define paths =================================
    project_root = Path.cwd()
    data_dir = project_root / "data"
    print(f"data_dir: {data_dir}")

    dataset_dir = data_dir / "interior_dataset"

    collector = DataCollector(dataset_dir=dataset_dir, class_labels=CLASS_LABELS)
    samples = collector()
    print(f"Total samples: {len(samples)}")


    # 3. ============================= Create DatasetSplitter ===========================
    splitter = DatasetSplitter(
        split_config=SPLIT_CONFIG,
        random_seed=RANDOM_SEED
    )

    train_samples, val_samples, test_samples = splitter(samples, shuffle=True)
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")


    # 4. =============================== Create Datasets ==================================
    train_dataset = InteriorDataset(
        train_samples,
        transform=get_transforms(img_size=IMG_SIZE, mode='train')
    )

    val_dataset = InteriorDataset(
        val_samples,
        transform=get_transforms(img_size=IMG_SIZE, mode='val')
    )

    test_dataset = InteriorDataset(
        test_samples,
        transform=get_transforms(img_size=IMG_SIZE, mode='test')
    )


    # 5. ============================= Create DataLoaders ==================================
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )


    # 6. =============================== Initializing model ===================================
    model = InteriorClassifier(num_classes=len(CLASS_LABELS)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    experiments_dir = project_root / "experiments"
    exp_dir = experiments_dir / f"exp{EXP_NUMBER:03d}"
    exp_results_dir = exp_dir / "results"
    exp_results_dir.mkdir(parents=True, exist_ok=True)

    def load_latest_file(pattern: str) -> Path | None:
        """Находит самый свежий файл по паттерну"""
        files = list(exp_results_dir.glob(pattern))
        if not files:
            return None
        # Сортируем по дате изменения (новейший первый)
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return files[0]

    # Пытаемся загрузить чекпоинт
    checkpoint_path = load_latest_file("ckpt*")
    if checkpoint_path:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            print(f"Loaded checkpoint from: {checkpoint_path.name}")
            
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            print(f"Successfully loaded checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_path}: {str(e)}")
            checkpoint_path = None

    # Если чекпоинта нет, пробуем загрузить полную модель
    if not checkpoint_path:
        model_path = load_latest_file("model*")
        if model_path:
            try:
                # Для полной модели не нужен load_state_dict
                model = torch.load(model_path, map_location=DEVICE)
                print(f"Successfully loaded full model from: {model_path.name}")
            except Exception as e:
                print(f"Error loading model {model_path}: {str(e)}")
                model_path = None

    # # Если ничего не загрузилось - исключение
    # if not checkpoint_path and not model_path:
    #     available_files = [f.name for f in exp_results_dir.iterdir() if f.is_file()]
    #     raise FileNotFoundError(
    #         f"No valid checkpoint or model found in {exp_results_dir}\n"
    #         f"Available files: {available_files or 'None'}"
    #     )


    # 7. ======================= Creating Trainer and start train =============================
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        sheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=EPOCHS,
        device=DEVICE,
        exp_results_dir=exp_results_dir

    )
    model = trainer.train()
