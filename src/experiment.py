from pathlib import Path
import json
from typing import Literal
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from src.datasets.interior_dataset import InteriorDataset, get_transforms
from src.models.interior_classifier_EfficientNet import InteriorClassifier
from src.trainer import Trainer
from src.datasets.utils.splitter import DatasetSplitter
from src.datasets.utils.collector import SampleCollector
from src.schemas.configs import ExperimentConfig


class Experiment:
    def __init__(
        self,
        exp_dir: Path,
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
        exp_dir.mkdir(parents=True, exist_ok=True)
        self.exp_dir = exp_dir

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

        self._init_exp_components()
    
    def _init_exp_components(self):
        # 2. ================================ Define paths =================================
        project_root = Path.cwd()
        data_dir = project_root / "data"
        print(f"data_dir: {data_dir}")

        dataset_dir = data_dir / "interior_dataset"

        self.collector = SampleCollector(dataset_dir=dataset_dir, class_labels=self.class_labels)
        samples = self.collector()
        print(f"Total samples: {len(samples)}")


        # 3. ============================= Create DatasetSplitter ===========================
        self.splitter = DatasetSplitter(splits=self.split_config, random_seed=self.random_seed)

        train_samples, val_samples, test_samples = self.splitter(samples, shuffle=True)
        print(f"Train samples: {len(train_samples)}")
        print(f"Val samples: {len(val_samples)}")
        print(f"Test samples: {len(test_samples)}")


        # 4. ============================= Create Datasets =============================
        train_dataset = InteriorDataset(
            train_samples,
            transform=get_transforms(img_size=self.img_size, mode='train')
        )
        val_dataset = InteriorDataset(
            val_samples,
            transform=get_transforms(img_size=self.img_size, mode='val')
        )
        test_dataset = InteriorDataset(
            test_samples,
            transform=get_transforms(img_size=self.img_size, mode='test')
        )


        # 5. ============================= Create DataLoaders =============================
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )

        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # 6. ============================= Initializing model =============================
        model = InteriorClassifier(num_classes=len(self.class_labels)).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.start_lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)

        # experiments_dir = project_root / "experiments"
        exp_dir = self.exp_dir / f"exp{self.exp_number:03d}"
        exp_results_dir = exp_dir / "results"
        exp_results_dir.mkdir(parents=True, exist_ok=True)

        # 7. ============================= Creating Trainer and start train =============================
        self.trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            sheduler=scheduler,
            epochs=self.epochs,
            device=self.device,
            exp_results_dir=exp_results_dir
        )
    
    def run(self) -> torch.nn.Module:
        """Запускает эксперимент"""

        # 1. Подгатавливаем Trainer
        self.trainer.set_data_loaders(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            test_loader=self.test_loader
        )

        # 2. Запускаем обучение
        print(f"Starting experiment {self.exp_number}")
        model = self.trainer.train()
        
        return model

    def load_latest_file(self, pattern: str) -> Path | None:
        """Находит самый свежий файл по паттерну"""
        files = list(self.exp_results_dir.glob(pattern))
        if not files:
            return None
        # Сортируем по дате изменения (новейший первый)
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return files[0]
    
    def to_config(self) -> ExperimentConfig:
        """Возвращает конфиг эксперимента как Pydantic модель"""
        return ExperimentConfig(
            exp_dir=self.exp_dir,
            exp_number=self.exp_number,
            epochs=self.epochs,
            img_size=self.img_size,
            start_lr=self.start_lr,
            random_seed=self.random_seed,
            collector_config=self.collector.to_config(),
            splitter_config=self.splitter.to_config(),
            train_dataset_config=self.train_loader.dataset.to_config(),
            val_dataset_config=self.val_loader.dataset.to_config(),
            test_dataset_config=self.test_loader.dataset.to_config(),
            trainer_config=self.trainer.to_config()
        )
    
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

    @staticmethod
    def load_model_from_dir(exp_results_dir: Path, device: str = "cpu"):
        """
        Загружает модель или чекпоинт из директории эксперимента.
        Возвращает: (model, checkpoint_path or model_path)
        """
        checkpoint_path = None
        model_path = None
        model = None

        # Пытаемся загрузить чекпоинт
        checkpoint_files = list(exp_results_dir.glob("ckpt*"))
        if checkpoint_files:
            checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            checkpoint_path = checkpoint_files[0]
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                print(f"Loaded checkpoint from: {checkpoint_path.name}")
                model = checkpoint  # Обычно тут нужен класс модели для load_state_dict!
                return model, checkpoint_path
            except Exception as e:
                print(f"Error loading checkpoint {checkpoint_path}: {str(e)}")
                checkpoint_path = None

        # Если чекпоинта нет, пробуем загрузить полную модель
        model_files = list(exp_results_dir.glob("model*"))
        if model_files:
            model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            model_path = model_files[0]
            try:
                model = torch.load(model_path, map_location=device)
                print(f"Successfully loaded full model from: {model_path.name}")
                return model, model_path
            except Exception as e:
                print(f"Error loading model {model_path}: {str(e)}")
                model_path = None

        # Если ничего не загрузилось - исключение
        available_files = [f.name for f in exp_results_dir.iterdir() if f.is_file()]
        raise FileNotFoundError(
            f"No valid checkpoint or model found in {exp_results_dir}\n"
            f"Available files: {available_files or 'None'}"
        )


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
