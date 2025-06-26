from pathlib import Path
import json

from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import torchvision.transforms as transforms
from PIL import Image

from src.config import RANDOM_SEED, SPLIT_RATIO, MIN_VAL_TEST_PER_CLASS, CLASS_LABELS
from src.dataset.splitter import DatasetSplitter
from src.dataset.interior_dataset import InteriorDataset, get_transforms
from src.models.interior_classifier_EfficientNet_B3 import InteriorClassifier
from src.trainer import Trainer


if __name__ == "__main__":

    # 1. Собираем все пути
    current_dir = Path.cwd()
    root_project = current_dir.parent
    data_dir = root_project / "data"
    print(f"data_dir: {data_dir}")

    dataset_dir = data_dir / "interior_dataset"

    samples = InteriorDataset.collect_samples(dataset_dir=dataset_dir)
    print(f"Total samples: {len(samples)}")

    # 2. Создание сплиттера
    splitter = DatasetSplitter(
        class_labels=CLASS_LABELS,
        split_config=SPLIT_RATIO,
        random_seed=RANDOM_SEED
    )

    # Разделение данных
    train_samples, val_samples, test_samples = splitter.split(samples, shuffle=True)
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")

    # Конфигурация
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    EPOCHS = 10
    LR = 3e-5
    IMG_SIZE = 380  # Для EfficientNet-B3
    EXP_NUMBER = 2

    # 3. Datasets
    train_dataset = InteriorDataset(
        train_samples,
        transform=get_transforms(mode='train'),
        mode='train'
    )

    val_dataset = InteriorDataset(
        val_samples,
        transform=get_transforms(mode='val'),
        mode='val'
    )

    test_dataset = InteriorDataset(
        test_samples,
        transform=get_transforms(mode='test'),
        mode='test'
    )


    # 4. DataLoaders
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

    model = InteriorClassifier(num_classes=len(CLASS_LABELS)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    experiments_dir = root_project / "experiments"
    exp_dir = experiments_dir / f"exp{EXP_NUMBER:03d}"
    exp_results_dir = exp_dir / "results"
    exp_results_dir.mkdir(parents=True, exist_ok=True)

    # Запуск обучения
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
