# src/trainer.py
from pathlib import Path
from typing import Any
import json
from tqdm import tqdm

import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from src.config import CLASS_LABELS
from src.schemas.configs import TrainerConfig, CriterionConfig, OptimizerConfig, SchedulerConfig, DataLoaderConfig
from src.models.interior_classifier_EfficientNet import InteriorClassifier


class Trainer:
    def __init__(
            self,
            model: InteriorClassifier,
            criterion: torch.nn.CrossEntropyLoss,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.LRScheduler,
            epochs: int,
            device: str,
            exp_results_dir: Path
        ):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.epochs = epochs
        self.device = device

        self.train_loader: torch.utils.data.DataLoader | None = None
        self.val_loader: torch.utils.data.DataLoader | None = None
        self.test_loader: torch.utils.data.DataLoader | None = None

        # Save paths
        self.exp_results_dir = exp_results_dir
        self.log_path = exp_results_dir / "training_report.json"

        self.log_dict: dict[str, Any] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "best_val_loss": float("inf")
        }
        self.best_val_loss = float("inf")
        self._load_log()
        self._current_epoch: int | None = None
        self._best_ckeckpoint_path: Path | None = None

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        train_loss = 0.0
        train_bar = tqdm(
            self.train_loader,
            desc=f'Epoch {epoch}/{self.epochs} [Train]',
            postfix={'loss': '?', 'lr': self.optimizer.param_groups[0]['lr']},
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
        )
        for inputs, _, class_idxs in train_bar:  # get batch [inputs, labels, class_idxs]
            inputs, class_idxs = inputs.to(self.device), class_idxs.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, class_idxs)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })

        self.scheduler.step()
        train_loss = round(train_loss / len(self.train_loader.dataset), 4)
        return train_loss
    
    def train(self) -> torch.nn.Module:
        if not all([self.train_loader, self.val_loader, self.test_loader]):
            raise ValueError("Для обучения должны быть заданы все три DataLoader: train_loader, val_loader, test_loader.")

        for epoch in range(1, self.epochs + 1):
            self._current_epoch = epoch
            train_loss = self.train_epoch(epoch)

            # ========================== VALIDATION REPORT ==============================
            val_loss, val_report = self.validate(epoch)
            val_accuracy = round(val_report['accuracy'], 4)
            self.log_dict["train_loss"].append(train_loss)
            self.log_dict["val_loss"].append(val_loss)
            self.log_dict["val_accuracy"].append(val_accuracy)

            print(f"\nEpoch {epoch} Summary:")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.4f}")
            print(
                f"Macro Avg: P={val_report['macro avg']['precision']:.4f} "
                f"R={val_report['macro avg']['recall']:.4f} "
                f"F1={val_report['macro avg']['f1-score']:.4f}"
            )
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.log_dict["best_val_loss"] = self.best_val_loss
                self.save_checkpoint(epoch, val_loss, val_accuracy)
                self.save_model()
                self._best_ckeckpoint_path = self.checkpoint_path  # Сохраняем путь к лучшему чекпоинту
                saved_to_text = self.exp_results_dir.relative_to(self.exp_results_dir.parent.parent.parent)
                print(f"Model saved to {saved_to_text} (Val Loss improved to {val_loss:.4f})")
                print(f"Checkpoint saved to {saved_to_text} (Val Loss improved to {val_loss:.4f})")
            self.save_log()
        
        # =========================== TEST REPORT ==========================
        # Перед тестированием подгружаем лучший чекпоинт, если он есть
        if self._best_ckeckpoint_path is not None and self._best_ckeckpoint_path.exists():
            checkpoint = torch.load(self._best_ckeckpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded best checkpoint from {self._best_ckeckpoint_path}")
        else:
            print("Warning: Best checkpoint path is not set or file does not exist. Testing current model state.")
        test_report, conf_matrix_dict = self.test()

        print("Final Test Results:")
        print(f"Test Accuracy: {test_report['accuracy']:.4f}")
        print(
            f"Macro Avg: P={test_report['macro avg']['precision']:.4f} "
            f"R={test_report['macro avg']['recall']:.4f} "
            f"F1={test_report['macro avg']['f1-score']:.4f}"
        )

        self.log_dict["test_report"] = test_report
        self.log_dict["confusion_matrix"] = conf_matrix_dict
        self.save_log()

        return self.model

    def validate(self, epoch: int) -> tuple[float, dict[str, Any]]:
        self.model.eval()

        val_loss = 0.0
        all_idxs: list[int] = []
        all_preds: list[int] = []
        
        val_bar = tqdm(
            self.val_loader,
            desc=f'Epoch {epoch}/{self.epochs} [Val]',
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
        )
        with torch.no_grad():
            for inputs, _, class_idxs in val_bar:  # get batch [inputs, labels, class_idxs]
                inputs, class_idxs = inputs.to(self.device), class_idxs.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, class_idxs)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)

                all_idxs.extend(class_idxs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                
                val_bar.set_postfix({'val_loss': f'{loss.item():.4f}'})

        val_loss = round(val_loss / len(self.val_loader.dataset), 4)

        report = classification_report(
            all_idxs,
            all_preds,
            target_names=CLASS_LABELS,
            zero_division=0,
            digits=4,
            output_dict=True
        )

        return val_loss, report

    def test(self) -> tuple[float, dict[str, Any]]:
        self.model.eval()

        all_idxs: list[int] = []
        all_preds: list[int] = []

        test_bar = tqdm(
            self.test_loader,
            desc='Final Testing',
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
        )

        with torch.no_grad():
            for inputs, _, class_idxs in test_bar:  # get batch [inputs, labels, class_idxs]
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                all_idxs.extend(class_idxs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        report = classification_report(
            all_idxs,
            all_preds,
            target_names=CLASS_LABELS,
            zero_division=0,
            digits=4,
            output_dict=True
        )

        # Generate confusion matrix
        conf_matrix = confusion_matrix(all_idxs, all_preds)
        
        # Convert to dictionary for JSON serialization
        conf_matrix_dict = {
            "matrix": conf_matrix.tolist(),
            "labels": CLASS_LABELS
        }

        # Save confusion matrix plot
        self._save_confusion_matrix_plot(conf_matrix=conf_matrix)

        return report, conf_matrix_dict

    def to_config(self) -> TrainerConfig:
        """
        Возвращает текущий конфиг (можно сохранить в JSON).
        """
        criterion_config = CriterionConfig(
            name=self.criterion.__class__.__name__,
            params=self._get_criterion_params()
        )

        optimizer_config = OptimizerConfig(
            name=self.optimizer.__class__.__name__,
            params=self._get_optimizer_params()
        )

        scheduler_config = SchedulerConfig(
            name=self.scheduler.__class__.__name__,
            params=self._get_scheduler_params()
        )

        train_loader_config = None
        if self.train_loader is not None:
            train_loader_config = DataLoaderConfig(
                batch_size=self.train_loader.batch_size,
                shuffle=True,
                num_workers=self.train_loader.num_workers,
                pin_memory=self.train_loader.pin_memory
            )
        
        val_loader_config = None
        if self.val_loader is not None:
            val_loader_config = DataLoaderConfig(
                batch_size=self.val_loader.batch_size,
                shuffle=False,
                num_workers=self.val_loader.num_workers,
                pin_memory=self.val_loader.pin_memory
            )

        test_loader_config = None
        if self.test_loader is not None:
            test_loader_config = DataLoaderConfig(
                batch_size=self.test_loader.batch_size,
                shuffle=False,
                num_workers=self.test_loader.num_workers,
                pin_memory=self.test_loader.pin_memory
            )

        return TrainerConfig(
            nn_model_config=self.model.to_config(),
            criterion_config=criterion_config,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            train_loader_config=train_loader_config,
            val_loader_config=val_loader_config,
            test_loader_config=test_loader_config,
            epochs=self.epochs,
            device=self.device,
            exp_results_dir=self.exp_results_dir
        )

    @classmethod
    def from_config(cls, config: TrainerConfig) -> "Trainer":
        """
        Создает Trainer из конфига.
        
        Args:
            config: Конфигурация тренера
            
        Returns:
            Trainer с созданными объектами из конфига
        """
        # Создаем модель из конфига
        model = InteriorClassifier.from_config(config.nn_model_config)
        
        # Создаем criterion из конфига
        criterion_class = getattr(torch.nn, config.criterion_config.name)
        criterion = criterion_class(**config.criterion_config.params)
        
        # Создаем optimizer из конфига
        optimizer_class = getattr(torch.optim, config.optimizer_config.name)
        optimizer = optimizer_class(model.parameters(), **config.optimizer_config.params)
        
        # Создаем scheduler из конфига (если есть)
        scheduler_class = getattr(torch.optim.lr_scheduler, config.scheduler_config.name)
        
        # Устанавливаем initial_lr для всех param_groups перед созданием scheduler
        for param_group in optimizer.param_groups:
            param_group['initial_lr'] = param_group['lr']
        
        scheduler = scheduler_class(optimizer, **config.scheduler_config.params)
        
        return cls(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=config.epochs,
            device=config.device,
            exp_results_dir=config.exp_results_dir
        )

    def set_data_loaders(
            self,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            test_loader: torch.utils.data.DataLoader
        ) -> None:
        """
        Устанавливает DataLoader'ы для тренера.
        
        Args:
            train_dataset: Dataset для обучения
            val_dataset: Dataset для валидации
            test_dataset: Dataset для тестирования
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def save_checkpoint(self, epoch: int, val_loss: float, val_accuracy: float) -> None:
        save_obj = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': val_loss,
            'accuracy': val_accuracy
        }
        torch.save(save_obj, self.checkpoint_path)
    
    def save_model(self) -> None:
        torch.save(self.model, self.model_path)
    
    def save_log(self) -> None:
        with open(self.log_path, "w") as f:
            json.dump(self.log_dict, f, indent=4)
    
    @property
    def checkpoint_path(self) -> Path:
        if self._current_epoch is None:
            return self.exp_results_dir / "ckpt.pth"
        return self.exp_results_dir / f"ckpt_epoch{self._current_epoch:02d}.pth"

    @property
    def model_path(self) -> Path:
        if self._current_epoch is None:
            return self.exp_results_dir / "model.pth"
        return self.exp_results_dir / f"model_epoch{self._current_epoch:02d}.pth"

    def _load_log(self) -> None:
        if self.log_path.exists():
            with open(self.log_path, "r") as f:
                self.log_dict = json.load(f)
            self.best_val_loss = self.log_dict.get("best_val_loss", float("inf"))

    def _save_confusion_matrix_plot(self, conf_matrix: np.ndarray) -> None:
        plt.figure(figsize=(10, 8))
        df_cm = pd.DataFrame(
            conf_matrix, 
            index=CLASS_LABELS,
            columns=CLASS_LABELS
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
            index=CLASS_LABELS,
            columns=CLASS_LABELS
        )
        sns.heatmap(df_cm_norm, annot=True, fmt='.2f', cmap='Blues')
        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        plot_norm_path = self.exp_results_dir / "confusion_matrix_normalized.png"
        plt.savefig(plot_norm_path, bbox_inches='tight')
        plt.close()
    
    def _get_criterion_params(self):
        """
        Возвращает только гиперпараметры для конструктора функции потерь (фильтрует только допустимые ключи).
        """
        import inspect
        criterion_class = self.criterion.__class__
        sig = inspect.signature(criterion_class.__init__)
        valid_keys = set(sig.parameters.keys()) - {"self"}

        params = {}
        for key, value in self.criterion.__dict__.items():
            if key in valid_keys:
                params[key] = value
        return params
    
    def _get_optimizer_params(self):
        """
        Возвращает только гиперпараметры первой param_group оптимайзера,
        пригодные для передачи в конструктор (фильтрует только допустимые ключи).
        """
        import inspect
        skip_keys = {"params"}
        group = self.optimizer.param_groups[0]
        optimizer_class = self.optimizer.__class__
        sig = inspect.signature(optimizer_class.__init__)
        valid_keys = set(sig.parameters.keys()) - {"self", "params"}
        hyperparams = {k: v for k, v in group.items() if k in valid_keys and k not in skip_keys}
        return hyperparams

    def _get_scheduler_params(self):
        """
        Возвращает только гиперпараметры для конструктора lr-scheduler (фильтрует только допустимые ключи).
        """
        import inspect
        scheduler_class = self.scheduler.__class__
        sig = inspect.signature(scheduler_class.__init__)
        valid_keys = set(sig.parameters.keys()) - {"self", "optimizer"}

        params = {}
        for key, value in self.scheduler.__dict__.items():
            if key in valid_keys:
                params[key] = value
        return params
