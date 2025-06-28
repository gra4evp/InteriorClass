from pydantic import BaseModel
from typing import Dict, Any, Optional

class DatasetConfig(BaseModel):
    img_size: int
    batch_size: int
    split_ratio: Dict[str, float]
    min_val_test_per_class: int
    class_labels: list[str]

class OptimizerConfig(BaseModel):
    name: str  # "Adam", "SGD", etc.
    params: Dict[str, Any]  # {"lr": 0.001, "weight_decay": 0.01}

class SchedulerConfig(BaseModel):
    name: str  # "StepLR", "CosineAnnealingLR", etc.
    params: Dict[str, Any]  # {"step_size": 30, "gamma": 0.1}

class ModelConfig(BaseModel):
    name: str  # "InteriorClassifier"
    params: Dict[str, Any]  # {"num_classes": 10, "pretrained": True}

class TrainingConfig(BaseModel):
    epochs: int
    device: str
    criterion: str  # "CrossEntropyLoss", "MSELoss", etc.
    optimizer: OptimizerConfig
    scheduler: Optional[SchedulerConfig] = None

class ExperimentConfig(BaseModel):
    exp_number: int
    random_seed: int
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig











from pathlib import Path
import json
from typing import Dict, Any, Optional, Type, TypeVar, Generic
from pydantic import BaseModel, validator
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Тип для датасета
D = TypeVar('D', bound=Dataset)
# Тип для модели
M = TypeVar('M', bound=nn.Module)
# Тип для сплиттера
S = TypeVar('S')
# Тип для тренера
T = TypeVar('T')

class BaseExperimentConfig(BaseModel):
    """Базовый конфиг эксперимента"""
    exp_number: int
    random_seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class DatasetConfig(BaseModel):
    img_size: int
    batch_size: int
    split_ratio: Dict[str, float]
    min_val_test_per_class: int
    class_labels: list[str]

class ModelConfig(BaseModel):
    name: str
    params: Dict[str, Any]

class TrainingConfig(BaseModel):
    epochs: int
    criterion: str
    optimizer: Dict[str, Any]
    scheduler: Optional[Dict[str, Any]] = None

class PathsConfig(BaseModel):
    project_root: str = "."
    dataset_dir: str

class FullExperimentConfig(BaseExperimentConfig):
    """Полный конфиг эксперимента со всеми компонентами"""
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    paths: PathsConfig

class Experiment(Generic[D, M, S, T]):
    def __init__(
        self,
        *,
        dataset_class: Type[D],
        model_class: Type[M],
        trainer_class: Type[T],
        splitter_class: Type[S],
        exp_number: int,
        random_seed: int = 42,
        device: Optional[str] = None,
        dataset_config: Optional[Dict[str, Any]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        paths_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Инициализация эксперимента.
        
        Можно передать либо полный конфиг, либо отдельные параметры.
        """
        # Базовые параметры
        self.exp_number = exp_number
        self.random_seed = random_seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Классы компонентов
        self.dataset_class = dataset_class
        self.model_class = model_class
        self.trainer_class = trainer_class
        self.splitter_class = splitter_class
        
        # Конфигурации
        self.dataset_config = dataset_config or {}
        self.model_config = model_config or {}
        self.training_config = training_config or {}
        self.paths_config = paths_config or {}
        
        # Инициализация путей
        self.project_root = Path(self.paths_config.get("project_root", "."))
        self.exp_dir = self.project_root / "experiments" / f"exp{self.exp_number:03d}"
        self.exp_results_dir = self.exp_dir / "results"
        self.exp_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Компоненты эксперимента (инициализируются лениво)
        self._samples: Optional[list] = None
        self._train_samples: Optional[list] = None
        self._val_samples: Optional[list] = None
        self._test_samples: Optional[list] = None
        self._model: Optional[M] = None
        self._criterion: Optional[nn.Module] = None
        self._optimizer: Optional[optim.Optimizer] = None
        self._scheduler: Optional[optim.lr_scheduler._LRScheduler] = None

    @property
    def config(self) -> FullExperimentConfig:
        """Возвращает полный конфиг эксперимента как Pydantic модель"""
        return FullExperimentConfig(
            exp_number=self.exp_number,
            random_seed=self.random_seed,
            device=str(self.device),
            dataset=DatasetConfig(**self.dataset_config),
            model=ModelConfig(**self.model_config),
            training=TrainingConfig(**self.training_config),
            paths=PathsConfig(**self.paths_config)
        )
    
    @classmethod
    def from_config(
        cls,
        config: FullExperimentConfig,
        *,
        dataset_class: Type[D],
        model_class: Type[M],
        trainer_class: Type[T],
        splitter_class: Type[S],
    ) -> "Experiment[D, M, S, T]":
        """Создает эксперимент из Pydantic конфига"""
        return cls(
            dataset_class=dataset_class,
            model_class=model_class,
            trainer_class=trainer_class,
            splitter_class=splitter_class,
            exp_number=config.exp_number,
            random_seed=config.random_seed,
            device=config.device,
            dataset_config=config.dataset.dict(),
            model_config=config.model.dict(),
            training_config=config.training.dict(),
            paths_config=config.paths.dict()
        )
    
    def to_config(self) -> FullExperimentConfig:
        """Возвращает конфиг эксперимента как Pydantic модель"""
        return self.config
    
    def save_config(self, path: Optional[Path] = None) -> None:
        """Сохраняет конфиг эксперимента в файл"""
        path = path or (self.exp_dir / "config.json")
        with open(path, "w") as f:
            json.dump(self.config.dict(), f, indent=4)
    
    @classmethod
    def load_config(cls, path: Path) -> FullExperimentConfig:
        """Загружает конфиг из файла"""
        with open(path) as f:
            config_data = json.load(f)
        return FullExperimentConfig(**config_data)
    
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
    
    def _init_model(self) -> None:
        """Инициализирует модель"""
        if self._model is None:
            self._model = self.model_class(**self.model_config.get("params", {})).to(self.device)
    
    def _init_criterion(self) -> None:
        """Инициализирует функцию потерь"""
        if self._criterion is None:
            criterion_name = self.training_config.get("criterion", "CrossEntropyLoss")
            self._criterion = getattr(nn, criterion_name)()
    
    def _init_optimizer(self) -> None:
        """Инициализирует оптимизатор"""
        if self._optimizer is None and self._model is not None:
            optimizer_config = self.training_config.get("optimizer", {})
            optimizer_class = getattr(optim, optimizer_config.get("name", "AdamW"))
            self._optimizer = optimizer_class(
                self._model.parameters(),
                **optimizer_config.get("params", {})
            )
    
    def _init_scheduler(self) -> None:
        """Инициализирует scheduler (если есть)"""
        if self._scheduler is None and self._optimizer is not None:
            scheduler_config = self.training_config.get("scheduler")
            if scheduler_config:
                scheduler_class = getattr(
                    optim.lr_scheduler, 
                    scheduler_config.get("name", "CosineAnnealingLR")
                )
                self._scheduler = scheduler_class(
                    self._optimizer,
                    **scheduler_config.get("params", {})
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
