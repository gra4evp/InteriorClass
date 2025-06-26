from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from src.config import TrainingConfig
from src.dataset.interior_dataset import InteriorDataset, get_transforms
from src.models.interior_classifier_EfficientNet_B3 import InteriorClassifier
from src.dataset.splitter import DatasetSplitter
from pydantic import BaseModel

from torch.optim.optimizer import Optimizer

class HyperparametersConfig(BaseModel):
    batch_size: int
    epochs: int
    lr: float
    img_size: int
    random_seed: int
    device: str
    criterion: torch.nn.CrossEntropyLoss
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler



class Trainer:
    def __init__(
            self,
            batch_size: int,
            epochs: int,
            lr: float,
            img_size: int,
            random_seed: int,
            device: str,
            dataset_dir: Path,
            exp_dir: Path
        ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.img_size = img_size
        self.random_seed = random_seed
        self.device = device
        self.dataset_dir = dataset_dir
        self.exp_dir = exp_dir

        # Save paths
        self.results_dir = exp_dir / "results"
        self.checkpoint_path = self.results_dir / "best_model.pth"
        self.log_path = self.results_dir / "training_log.json"

        self._init_paths()
        self._init_datasets()
        self._init_loaders()
        self._init_model()
        self.log_dict: Dict[str, Any] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "best_val_loss": float("inf")
        }
        self.best_val_loss = float("inf")
        self._load_log()

    def _init_paths(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.exp_dir = Path(self.exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = Path(self.checkpoint_path) if self.checkpoint_path else self.exp_dir / "best_model.pth"
        self.log_path = Path(self.log_path) if self.log_path else self.exp_dir / "training_log.json"

    def _init_datasets(self) -> None:
        samples = InteriorDataset.collect_samples(dataset_dir=self.data_dir)
        splitter = DatasetSplitter(
            class_labels=self.class_labels,
            split_config=self.split_ratio,
            random_seed=self.random_seed
        )
        train_samples, val_samples, test_samples = splitter.split(samples, shuffle=True)
        self.train_dataset = InteriorDataset(
            train_samples,
            transform=get_transforms(mode='train'),
            mode='train'
        )
        self.val_dataset = InteriorDataset(
            val_samples,
            transform=get_transforms(mode='val'),
            mode='val'
        )
        self.test_dataset = InteriorDataset(
            test_samples,
            transform=get_transforms(mode='test'),
            mode='test'
        )

    def _init_loaders(self) -> None:
        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        self.val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        self.test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    def _init_model(self) -> None:
        self.model = InteriorClassifier(num_classes=len(self.class_labels)).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)

    def _load_log(self) -> None:
        if self.log_path.exists():
            with open(self.log_path, "r") as f:
                self.log = json.load(f)
            self.best_val_loss = self.log.get("best_val_loss", float("inf"))

    def save_log(self) -> None:
        with open(self.log_path, "w") as f:
            json.dump(self.log, f, indent=4)

    def save_checkpoint(self, epoch: int, val_loss: float, val_accuracy: float) -> None:
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': val_loss,
                'accuracy': val_accuracy
            }
            , self.checkpoint_path
        )

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        train_loss = 0.0
        train_bar = tqdm(
            self.train_loader,
            desc=f'Epoch {epoch}/{self.config.epochs} [Train]',
            postfix={'loss': '?', 'lr': self.optimizer.param_groups[0]['lr']},
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
        )
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        self.scheduler.step()
        return train_loss / len(self.train_loader.dataset)
    
    def train(self) -> None:
        for epoch in range(1, self.config.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_accuracy, report = self.validate(epoch)
            self.log["train_loss"].append(train_loss)
            self.log["val_loss"].append(val_loss)
            self.log["val_accuracy"].append(val_accuracy)
            print(f"\nEpoch {epoch} Summary:")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.4f}")
            print(
                f"Macro Avg: P={report['macro avg']['precision']:.4f} "
                f"R={report['macro avg']['recall']:.4f} "
                f"F1={report['macro avg']['f1-score']:.4f}\n"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.log["best_val_loss"] = self.best_val_loss
                self.save_checkpoint(epoch, val_loss, val_accuracy)
                print(f"Checkpoint saved to {self.checkpoint_path} (Val Loss improved to {val_loss:.4f})")
            self.save_log()
        test_accuracy, final_report = self.test()
        print("\nFinal Test Results:")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(
            f"Macro Avg: P={final_report['macro avg']['precision']:.4f} "
            f"R={final_report['macro avg']['recall']:.4f} "
            f"F1={final_report['macro avg']['f1-score']:.4f}"
        )
        self.log["test_accuracy"] = test_accuracy
        self.log["test_report"] = final_report
        self.save_log()

    def validate(self, epoch: int) -> Tuple[float, float, Dict[str, Any]]:
        self.model.eval()
        val_loss = 0.0
        all_preds: List[int] = []
        all_labels: List[int] = []
        val_bar = tqdm(
            self.val_loader,
            desc=f'Epoch {epoch}/{self.config.epochs} [Val]',
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
        )
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                val_bar.set_postfix({'val_loss': f'{loss.item():.4f}'})
        val_loss = val_loss / len(self.val_loader.dataset)
        report = classification_report(
            all_labels, all_preds,
            target_names=self.config.class_labels,
            zero_division=0,
            digits=4,
            output_dict=True
        )
        val_accuracy = report['accuracy']
        return val_loss, val_accuracy, report

    def test(self) -> Tuple[float, Dict[str, Any]]:
        self.model.eval()
        test_preds: List[int] = []
        test_labels: List[int] = []
        test_bar = tqdm(
            self.test_loader,
            desc='Final Testing',
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
        )
        with torch.no_grad():
            for inputs, labels in test_bar:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
        final_report = classification_report(
            test_labels, test_preds,
            target_names=self.config.class_labels,
            digits=4,
            output_dict=True
        )
        test_accuracy = final_report['accuracy']
        return test_accuracy, final_report

    def to_config(self) -> TrainingConfig:
        """Возвращает текущий конфиг (можно сохранить в JSON)."""
        return TrainingConfig(
            batch_size=self.batch_size,
            epochs=self.epochs,
            lr=self.lr,
            img_size=self.img_size,
            random_seed=self.random_seed,
            class_labels=self.class_labels,
        )
    
    def from_config(self, config: TrainingConfig) -> None:
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.lr = config.lr
        self.img_size = config.img_size
        self.random_seed = config.random_seed
        self.class_labels = config.class_labels
        self.split_ratio = config.split_ratio
        self.device = config.device
