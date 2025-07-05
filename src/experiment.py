from pathlib import Path
import json
from typing import Dict, Any, Optional
from pydantic import BaseModel

from src.datasets.interior_dataset import InteriorDataset
from src.models.interior_classifier_EfficientNet_B3 import InteriorClassifier
from src.trainer import Trainer
from datasets.utils.splitter import DatasetSplitter


class DatasetConfig(BaseModel):
    img_size: int
    batch_size: int
    split_ratio: Dict[str, float]
    min_val_test_per_class: int
    class_labels: list[str]


class PathsConfig(BaseModel):
    project_root: str = "."
    dataset_dir: str


class ExperimentConfig(BaseModel):
    exp_number: int
    random_seed: int
    trainer_config: TrainerConfig
    paths: PathsConfig


class Experiment:
    def __init__(
        self,
        trainer: Trainer,
        exp_number: int,
        random_seed: int = 42,
        exp_results_dir: Path = Path("experiments")
    ):
        """
        Инициализация эксперимента.
        
        Можно передать либо полный конфиг, либо отдельные параметры.
        """
        # Базовые параметры
        self.trainer = trainer
        self.exp_number = exp_number
        self.random_seed = random_seed
        self.exp_results_dir = exp_results_dir
        
        # Инициализация путей
        self.project_root = Path(self.paths_config.get("project_root", "."))
        self.exp_dir = self.project_root / "experiments" / f"exp{self.exp_number:03d}"
        self.exp_results_dir = self.exp_dir / "results"
        self.exp_results_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_config(cls, config: ExperimentConfig) -> "Experiment":
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
    
    def save_config(self, path: Optional[Path] = None) -> None:
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
    
    def _init_samples(self) -> None:
        """Инициализирует samples"""
        if self._samples is None:
            dataset_dir = self.project_root / self.paths_config.get("dataset_dir", "data/interior_dataset")
            self._samples = self.dataset_class.collect_samples(dataset_dir=dataset_dir)
    
    def _init_splitter(self) -> None:
        """Инициализирует splitter и разделяет данные"""
        if self._train_samples is None:
            self._init_samples()
            splitter = self.splitter_class(
                class_labels=self.dataset_config.get("class_labels", []),
                split_config=self.dataset_config.get("split_ratio", {}),
                random_seed=self.random_seed
            )
            self._train_samples, self._val_samples, self._test_samples = splitter.split(
                self._samples, shuffle=True
            )
    
    def run(self) -> M:
        """Запускает эксперимент"""
        # 1. Инициализация компонентов
        self._init_splitter()
        self._init_model()
        self._init_criterion()
        self._init_optimizer()
        self._init_scheduler()
        
        # 2. Сохраняем конфиг перед началом
        self.save_config()
        
        # 3. Создаем даталоадеры
        img_size = self.dataset_config.get("img_size", 224)
        batch_size = self.dataset_config.get("batch_size", 32)
        
        train_dataset = self.dataset_class(
            self._train_samples,
            transform=get_transforms(img_size=img_size, mode='train'),
            mode='train'
        )
        
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        # ... аналогично для val и test ...
        
        # 4. Инициализируем trainer
        trainer = self.trainer_class(
            model=self._model,
            criterion=self._criterion,
            optimizer=self._optimizer,
            scheduler=self._scheduler,
            train_loader=train_loader,
            # ... остальные параметры ...
            epochs=self.training_config.get("epochs", 10),
            device=self.device,
            exp_results_dir=self.exp_results_dir
        )
        
        # 5. Запускаем обучение
        print(f"Starting experiment {self.exp_number}")
        self._model = trainer.train()
        
        # 6. Сохраняем модель
        final_model_path = self.exp_results_dir / "final_model.pth"
        torch.save(self._model.state_dict(), final_model_path)
        
        return self._model


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
