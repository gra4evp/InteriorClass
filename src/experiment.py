from pathlib import Path
import json
from typing import Dict, Any, Optional, Literal
from pydantic import BaseModel
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from src.datasets.interior_dataset import InteriorDataset, get_transforms
from src.models.interior_classifier_EfficientNet import InteriorClassifier
from src.trainer import Trainer, TrainerConfig
from src.datasets.utils.splitter import DatasetSplitter
from src.datasets.utils.collector import SampleCollector
from src.config import RANDOM_SEED, SPLIT_CONFIG, MIN_VAL_TEST_PER_CLASS, CLASS_LABELS


class Experiment:
    def __init__(
        self,
        exp_results_dir: Path,
        class_labels: list[str],
        split_config: dict[str, dict[str, int | float]],
        exp_number: int | None = None,
        batch_size: int = 32,
        epochs: int = 10,
        img_size: int = 448,
        start_lr: float = 3e-5,
        random_seed: int = 42,
        device: Literal['cuda', 'cpu'] | None = None
    ):
        """
        Инициализация эксперимента.
        
        Можно передать либо полный конфиг, либо отдельные параметры.
        """
        exp_results_dir.mkdir(parents=True, exist_ok=True)
        self.exp_results_dir = exp_results_dir

        # Базовые параметры
        self.class_labels = class_labels
        self.split_config = split_config
        self.exp_number = exp_number
        self.batch_size = batch_size
        self.epochs = epochs
        self.img_size = img_size
        self.start_lr = start_lr
        self.random_seed = random_seed
        self.device = device

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # # Инициализация путей
        # self.project_root = Path(self.paths_config.get("project_root", "."))
        # self.exp_dir = self.project_root / "experiments" / f"exp{self.exp_number:03d}"
        # self.exp_results_dir = self.exp_dir / "results"
        # self.exp_results_dir.mkdir(parents=True, exist_ok=True)

        self.init_exp()
    
    def init_exp(self):
        # 2. ================================ Define paths =================================
        project_root = Path.cwd()
        data_dir = project_root / "data"
        print(f"data_dir: {data_dir}")

        dataset_dir = data_dir / "interior_dataset"

        collector = SampleCollector(dataset_dir=dataset_dir, class_labels=self.class_labels)
        samples = collector()
        print(f"Total samples: {len(samples)}")


        # 3. ============================= Create DatasetSplitter ===========================
        splitter = DatasetSplitter(splits=self.split_config, random_seed=self.random_seed)

        train_samples, val_samples, test_samples = splitter(samples, shuffle=True)
        print(f"Train samples: {len(train_samples)}")
        print(f"Val samples: {len(val_samples)}")
        print(f"Test samples: {len(test_samples)}")


        # 4. ============================= Create Datasets =============================
        self.train_dataset = InteriorDataset(
            train_samples,
            transform=get_transforms(img_size=self.img_size, mode='train')
        )

        self.val_dataset = InteriorDataset(
            val_samples,
            transform=get_transforms(img_size=self.img_size, mode='val')
        )

        self.test_dataset = InteriorDataset(
            test_samples,
            transform=get_transforms(img_size=self.img_size, mode='test')
        )


        # 5. ============================= Create DataLoaders =============================
        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )

        self.val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        self.test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )


        # 6. ============================= Initializing model =============================
        self.model = InteriorClassifier(num_classes=len(self.class_labels)).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=self.start_lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)

        experiments_dir = project_root / "experiments"
        exp_dir = experiments_dir / f"exp{self.exp_number:03d}"
        exp_results_dir = exp_dir / "results"
        exp_results_dir.mkdir(parents=True, exist_ok=True)

        # Пытаемся загрузить чекпоинт
        checkpoint_path = self.load_latest_file("ckpt*")
        if checkpoint_path:
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                print(f"Loaded checkpoint from: {checkpoint_path.name}")
                
                model.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                print(f"Successfully loaded checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
            except Exception as e:
                print(f"Error loading checkpoint {checkpoint_path}: {str(e)}")
                checkpoint_path = None

        # Если чекпоинта нет, пробуем загрузить полную модель
        if not checkpoint_path:
            model_path = self.load_latest_file("model*")
            if model_path:
                try:
                    # Для полной модели не нужен load_state_dict
                    model = torch.load(model_path, map_location=self.device)
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


        # 7. ============================= Creating Trainer and start train =============================
        self.trainer = Trainer(
            model=model,
            criterion=self.criterion,
            optimizer=self.optimizer,
            sheduler=self.scheduler,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            test_loader=self.test_loader,
            epochs=self.epochs,
            device=self.device,
            exp_results_dir=exp_results_dir

        )
    
    def run(self) -> torch.nn.Module:
        """Запускает эксперимент"""
        # 1. Инициализация компонентов
        self.init_exp()
        
        # 5. Запускаем обучение
        print(f"Starting experiment {self.exp_number}")
        model = self.trainer.train()
        
        # 6. Сохраняем модель
        final_model_path = self.exp_results_dir / "final_model.pth"
        torch.save(model.state_dict(), final_model_path)
        
        return model

    def load_latest_file(self, pattern: str) -> Path | None:
        """Находит самый свежий файл по паттерну"""
        files = list(self.exp_results_dir.glob(pattern))
        if not files:
            return None
        # Сортируем по дате изменения (новейший первый)
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return files[0]
    
    @classmethod
    def from_config(cls, config: ExperimentConfig) -> 'Experiment':
        """Создает эксперимент из Pydantic конфига"""
        trainer = Trainer.from_config(config.trainer_config)
        return cls(
            trainer=trainer,
            exp_number=config.exp_number,
            random_seed=config.random_seed,
            exp_results_dir=config.exp_results_dir
        )
    
    def to_config(self) -> ExperimentConfig:
        """Возвращает конфиг эксперимента как Pydantic модель"""
        return ExperimentConfig(
            exp_number=self.exp_number,
            random_seed=self.random_seed,
            trainer_config=self.trainer_config,
            paths=self.paths_config
        )
    
    def save_config(self, path: Path | None = None) -> None:
        """Сохраняет конфиг эксперимента в файл"""
        path = path or (self.exp_dir / "config.json")
        with open(path, "w") as f:
            json.dump(self.to_config().dict(), f, indent=4)
    
    @classmethod
    def load_config(cls, path: Path) -> ExperimentConfig:
        """Загружает конфиг из файла"""
        with open(path) as f:
            config_data = json.load(f)
        return ExperimentConfig(**config_data)


if __name__ == "__main__":
    # Example Создание эксперимента напрямую (без конфиг-файла):
    experiment = Experiment(
        dataset_class=InteriorDataset,
        model_class=InteriorClassifier,
        trainer_class=Trainer,
        splitter_class=DatasetSplitter,
        exp_number=5,
        random_seed=42,
        dataset_config={
            "img_size": 380,
            "batch_size": 32,
            "split_ratio": {"train": 0.7, "val": 0.15, "test": 0.15},
            "class_labels": ["class1", "class2", "class3"]
        },
        model_config={
            "name": "InteriorClassifier",
            "params": {"num_classes": 3}
        },
        training_config={
            "epochs": 10,
            "criterion": "CrossEntropyLoss",
            "optimizer": {
                "name": "AdamW",
                "params": {"lr": 3e-5}
            }
        }
    )

    experiment.run()


    # Использование с конфиг файлом

    # Загрузка конфига
    config = Experiment.load_config(Path("configs/experiment_005.json"))

    # Создание эксперимента
    experiment = Experiment.from_config(
        config=config,
        dataset_class=InteriorDataset,
        model_class=InteriorClassifier,
        trainer_class=Trainer,
        splitter_class=DatasetSplitter
    )

    # Запуск
    experiment.run()
