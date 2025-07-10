# src/experiment.py
from pathlib import Path
import json
from typing import Literal, Any
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.datasets.interior_dataset import InteriorDataset, get_transforms
from src.models.interior_classifier_EfficientNet import InteriorClassifier
from src.trainer import Trainer
from src.datasets.utils.splitter import DatasetSplitter
from src.datasets.utils.collector import SampleCollector
from src.schemas.configs import ExperimentConfig


class Experiment:
    def __init__(
        self,
        dataset_dir: Path,
        exp_dir: Path,
        class_labels: list[str],
        splits: dict[str, dict[str, int | float]],
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
        self.dataset_dir = dataset_dir

        exp_dir.mkdir(parents=True, exist_ok=True)
        self.exp_dir = exp_dir

        # Базовые параметры
        self.class_labels = class_labels
        self.splits = splits
        self.exp_number = exp_number
        self.batch_size = batch_size
        self.epochs = epochs
        self.img_size = img_size
        self.start_lr = start_lr
        self.random_seed = random_seed
        self.device = device

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.log_dict: dict[str, Any] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "best_val_loss": float("inf")
        }
        self._best_val_loss = float("inf")

        self._init_exp_components()
    
    def _init_exp_components(self):
        # 1. ============================= Create SampleCollector =============================
        self.collector = SampleCollector(dataset_dir=self.dataset_dir, class_labels=self.class_labels)
        samples = self.collector()
        print(f"Total samples: {len(samples)}")


        # 2. ============================= Create DatasetSplitter =============================
        self.splitter = DatasetSplitter(splits=self.splits, random_seed=self.random_seed)

        train_samples, val_samples, test_samples = self.splitter(samples, shuffle=True)
        print(f"Train samples: {len(train_samples)}")
        print(f"Val samples: {len(val_samples)}")
        print(f"Test samples: {len(test_samples)}")


        # 3. ============================= Create Datasets =============================
        train_dataset = InteriorDataset(
            transforms=get_transforms(img_size=self.img_size, mode='train'),
            transforms_filepath=self.exp_dir / "train_tranforms.json"
        )
        val_dataset = InteriorDataset(
            transforms=get_transforms(img_size=self.img_size, mode='val'),
            transforms_filepath=self.exp_dir / "val_tranforms.json"
        )
        test_dataset = InteriorDataset(
            transforms=get_transforms(img_size=self.img_size, mode='test'),
            transforms_filepath=self.exp_dir / "test_tranforms.json"
        )
        train_dataset.prepare(train_samples)
        val_dataset.prepare(val_samples)
        test_dataset.prepare(test_samples)


        # 4. ============================= Create DataLoaders =============================
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

        # 5. ============================= Initializing model =============================
        model = InteriorClassifier(
            num_classes=len(self.class_labels),
            backbone_name='efficientnet_b3',
            use_head=True
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.start_lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        exp_results_dir = self.exp_dir / "results"
        exp_results_dir.mkdir(parents=True, exist_ok=True)

        # 6. ============================= Creating Trainer and start train =============================
        self.trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
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

        # 2. Сохраняем конфиг
        self.save_config()

        # 2. Запускаем обучение
        print(f"Starting experiment {self.exp_number}")
        for train_event in self.trainer.train():
            self.log_dict["train_loss"].append(train_event.loss_value)


            # ========================== VALIDATION REPORT ==============================
            val_event = self.trainer.validate()

            val_accuracy = round(val_event.metrics['accuracy'], 4)
            self.log_dict["val_loss"].append(val_event.loss_value)
            self.log_dict["val_accuracy"].append(val_accuracy)

            print(f"\nEpoch {train_event.epoch} Summary:")
            print(f"Train Loss: {train_event.loss_value:.4f} | Val Loss: {val_event.loss_value:.4f}")
            print(f"Val Accuracy: {val_accuracy:.4f}")
            print(
                f"Macro Avg: P={val_event.metrics['macro avg']['precision']:.4f} "
                f"R={val_event.metrics['macro avg']['recall']:.4f} "
                f"F1={val_event.metrics['macro avg']['f1-score']:.4f}"
            )
            
            if val_event.loss_value < self.best_val_loss:
                self.best_val_loss = val_event.loss_value
                self.log_dict["best_val_loss"] = self.best_val_loss
                self.save_checkpoint(train_event.epoch, val_event.loss_value, val_accuracy)
                self.save_model()
                self._best_ckeckpoint_path = self.checkpoint_path  # Сохраняем путь к лучшему чекпоинту
                saved_to_text = self.exp_results_dir.relative_to(self.exp_results_dir.parent.parent.parent)
                print(f"Model saved to {saved_to_text} (Val Loss improved to {val_event.loss_value:.4f})")
                print(f"Checkpoint saved to {saved_to_text} (Val Loss improved to {val_event.loss_value:.4f})")
            self.save_log()
        
        # =========================== TEST REPORT ==========================
        test_event = self.trainer.test()

        print("Final Test Results:")
        print(f"Test Accuracy: {test_event.metrics['accuracy']:.4f}")
        print(
            f"Macro Avg: P={test_event.metrics['macro avg']['precision']:.4f} "
            f"R={test_event.metrics['macro avg']['recall']:.4f} "
            f"F1={test_event.metrics['macro avg']['f1-score']:.4f}"
        )

        self.log_dict["test_report"] = test_report
        self.log_dict["confusion_matrix"] = conf_matrix_dict
        self._save_log()
                # Save confusion matrix plot
        self._save_confusion_matrix_plot(conf_matrix=conf_matrix)
        
        return self.trainer.model

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
            dataset_dir=self.dataset_dir,
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
        """
        Создает эксперимент из Pydantic конфига.
        Восстанавливает все компоненты и подгружает модель из директории эксперимента
        """
        # 1. Восстановление SampleCollector и сборка сэмплов
        collector = SampleCollector.from_config(config.collector_config)
        samples = collector()

        # 2. Восстановление DatasetSplitter и разбиение
        splitter = DatasetSplitter.from_config(config.splitter_config)
        train_samples, val_samples, test_samples = splitter(samples, shuffle=True)

        # 3. Восстановление датасетов
        train_dataset = InteriorDataset.from_config(config.train_dataset_config)
        train_dataset.prepare(train_samples)
        val_dataset = InteriorDataset.from_config(config.val_dataset_config)
        val_dataset.prepare(val_samples)
        test_dataset = InteriorDataset.from_config(config.test_dataset_config)
        test_dataset.prepare(test_samples)

        # 4. DataLoader'ы
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=config.trainer_config.train_loader_config.batch_size,
            shuffle=config.trainer_config.train_loader_config.shuffle,
            num_workers=config.trainer_config.train_loader_config.num_workers,
            pin_memory=config.trainer_config.train_loader_config.pin_memory,
            drop_last=True
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=config.trainer_config.val_loader_config.batch_size,
            shuffle=config.trainer_config.val_loader_config.shuffle,
            num_workers=config.trainer_config.val_loader_config.num_workers,
            pin_memory=config.trainer_config.val_loader_config.pin_memory
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=config.trainer_config.test_loader_config.batch_size,
            shuffle=config.trainer_config.test_loader_config.shuffle,
            num_workers=config.trainer_config.test_loader_config.num_workers,
            pin_memory=config.trainer_config.test_loader_config.pin_memory
        )

        # 5. Восстановление Trainer из конфига
        trainer = Trainer.from_config(config.trainer_config)
        trainer.set_data_loaders(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader
        )

        # 6. Загрузка модели/весов
        loaded, path = cls.load_model_from_dir(config.trainer_config.exp_results_dir, device=config.trainer_config.device)
        if isinstance(loaded, dict) and 'model_state_dict' in loaded:
            trainer.model.load_state_dict(loaded['model_state_dict'])
        elif isinstance(loaded, torch.nn.Module):
            trainer.model = loaded
        # Иначе оставляем trainer.model как есть

        # 7. Собираем Experiment
        exp = cls(
            dataset_dir=config.dataset_dir,
            exp_dir=config.exp_dir,
            class_labels=config.collector_config.class_labels,
            splits=config.splitter_config.splits,
            exp_number=config.exp_number,
            batch_size=config.trainer_config.train_loader_config.batch_size,
            epochs=config.epochs,
            img_size=config.img_size,
            start_lr=config.start_lr,
            random_seed=config.random_seed,
            device=config.trainer_config.device
        )
        # Перезаписываем подготовленные компоненты
        exp.collector = collector
        exp.splitter = splitter
        exp.train_loader = train_loader
        exp.val_loader = val_loader
        exp.test_loader = test_loader
        exp.trainer = trainer
        return exp
    
    def save_config(self, path: Path | None = None) -> None:
        """Сохраняет конфиг эксперимента в файл"""
        path = path or (self.exp_dir / "config.json")
        with open(path, "w") as f:
            json.dump(self.to_config().model_dump(mode='json'), f, indent=4)
    
    @classmethod
    def load_config(cls, path: Path) -> ExperimentConfig:
        """Загружает конфиг из файла"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return ExperimentConfig(**config_dict)

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
    
    def _save_log(self) -> None:
        with open(self.log_path, "w") as f:
            json.dump(self.log_dict, f, indent=4)
    
    def _load_log(self) -> None:
        if self.log_path.exists():
            with open(self.log_path, "r") as f:
                self.log_dict = json.load(f)
            self._best_val_loss = self.log_dict.get("best_val_loss", float("inf"))
    
    def _save_confusion_matrix_plot(self, conf_matrix: np.ndarray) -> None:
        plt.figure(figsize=(10, 8))
        df_cm = pd.DataFrame(
            conf_matrix, 
            index=self.class_labels,
            columns=self.class_labels
        )
        sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Ground Truth')
        
        plot_path = self.exp_results_dir / "confusion_matrix.png"
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        
        # Also save normalized version
        cm_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(10, 8))
        df_cm_norm = pd.DataFrame(
            cm_normalized, 
            index=self.class_labels,
            columns=self.class_labels
        )
        sns.heatmap(df_cm_norm, annot=True, fmt='.2f', cmap='Blues')
        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        plot_norm_path = self.exp_results_dir / "confusion_matrix_normalized.png"
        plt.savefig(plot_norm_path, bbox_inches='tight')
        plt.close()
