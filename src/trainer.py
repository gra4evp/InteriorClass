from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
from tqdm import tqdm
import torch

from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from src.config import TrainingConfig
from pydantic import BaseModel


class OptimizerConfig(BaseModel):
    name: str  # "Adam", "SGD", etc.
    params: Dict[str, Any]  # {"lr": 0.001, "weight_decay": 0.01}


class SchedulerConfig(BaseModel):
    name: str  # "StepLR", "CosineAnnealingLR", etc.
    params: Dict[str, Any]  # {"step_size": 30, "gamma": 0.1}


class HyperParametersConfig(BaseModel):
    batch_size: int
    epochs: int
    img_size: int
    random_seed: int
    device: str
    criterion: str  # "CrossEntropyLoss", "MSELoss", etc.
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig | None = None



class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            criterion: torch.nn.CrossEntropyLoss,
            optimizer: torch.optim.Optimizer,
            sheduler: torch.optim.lr_scheduler.LRScheduler,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            test_loader: torch.utils.data.DataLoader,
            epochs: int,
            device: str,
            exp_results_dir: Path
        ):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = sheduler

        self.epochs = epochs
        self.device = device

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Save paths
        self.exp_results_dir = exp_results_dir
        self.checkpoint_path = exp_results_dir / "best_model.pth"
        self.log_path = exp_results_dir/ "training_log.json"

        self.log_dict: Dict[str, Any] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "best_val_loss": float("inf")
        }
        self.best_val_loss = float("inf")
        self._load_log()

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
    
    def train(self) -> torch.nn.Module:
        for epoch in range(1, self.config.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_accuracy, report = self.validate(epoch)
            self.log_dict["train_loss"].append(train_loss)
            self.log_dict["val_loss"].append(val_loss)
            self.log_dict["val_accuracy"].append(val_accuracy)

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
                self.log_dict["best_val_loss"] = self.best_val_loss
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
        self.log_dict["test_accuracy"] = test_accuracy
        self.log_dict["test_report"] = final_report
        self.save_log()
        return self.model

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
        """
        Возвращает текущий конфиг (можно сохранить в JSON).
        """
    
    def from_config(self, config: TrainingConfig) -> None:
        pass

    def save_checkpoint(self, epoch: int, val_loss: float, val_accuracy: float) -> None:
        save_obj = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': val_loss,
            'accuracy': val_accuracy
        }
        torch.save(save_obj, self.checkpoint_path)
    
    def save_log(self) -> None:
        with open(self.log_path, "w") as f:
            json.dump(self.log_dict, f, indent=4)

    def _load_log(self) -> None:
        if self.log_path.exists():
            with open(self.log_path, "r") as f:
                self.log_dict = json.load(f)
            self.best_val_loss = self.log_dict.get("best_val_loss", float("inf"))
