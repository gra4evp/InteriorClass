# src/trainer.py
from pathlib import Path
from typing import Generator
from tqdm import tqdm

import torch
from src.schemas.configs import (
    TrainerConfig,
    CriterionConfig,
    OptimizerConfig,
    SchedulerConfig,
    DataLoaderConfig
)
from src.models.interior_classifier_EfficientNet import InteriorClassifier
from src.schemas.training import TrainEpochEvent, TestEvent, ValidationEvent, EventType


class Trainer:
    def __init__(
            self,
            model: InteriorClassifier,
            criterion: torch.nn.CrossEntropyLoss,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.LRScheduler,
            epochs: int,
            device: str
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

        self._current_epoch: int | None = None

    def train_epoch(self, epoch: int) -> TrainEpochEvent:
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

        event = TrainEpochEvent(
            type=EventType.epoch_end,
            epoch=epoch,
            loss_value=train_loss
        )
        return event
    
    def train(self) -> Generator[TrainEpochEvent, None, None]:
        if not all([self.train_loader, self.val_loader, self.test_loader]):
            raise ValueError(
                "All three dataloaders must be set for training: train_loader, val_loader, test_loader."
            )

        for epoch in range(1, self.epochs + 1):
            self._current_epoch = epoch
            yield self.train_epoch(epoch)

    def validate(self) -> ValidationEvent:
        self.model.eval()
        
        val_bar = tqdm(
            self.val_loader,
            desc=f'Epoch {self._current_epoch}/{self.epochs} [Val]',
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
        )

        val_loss = 0.0
        targets: list[int] = []
        predictions: list[int] = []
        with torch.no_grad():
            for inputs, _, class_idxs in val_bar:  # get batch [inputs, labels, class_idxs]
                inputs, class_idxs = inputs.to(self.device), class_idxs.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, class_idxs)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)

                targets.extend(class_idxs.cpu().numpy())
                predictions.extend(preds.cpu().numpy())
                
                val_bar.set_postfix({'val_loss': f'{loss.item():.4f}'})

        val_loss = round(val_loss / len(self.val_loader.dataset), 4)

        event = ValidationEvent(
            type=EventType.validation,
            epoch=self._current_epoch,
            loss_value=val_loss,
            artifacts={'targets': targets, 'predictions': predictions}
        )
        return event

    def test(self, checkpoint_path: Path | None = None) -> TestEvent:
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print("Testing current model state (no checkpoint loaded).")

        self.model.eval()

        test_bar = tqdm(
            self.test_loader,
            desc='Final Testing',
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
        )

        targets: list[int] = []
        predictions: list[int] = []
        with torch.no_grad():
            for inputs, _, class_idxs in test_bar:  # get batch [inputs, labels, class_idxs]
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                targets.extend(class_idxs.cpu().numpy())
                predictions.extend(preds.cpu().numpy())

        event = TestEvent(
            type=EventType.test,
            loss_value=None,
            artifacts={'targets': targets, 'predictions': predictions}
        )
        return event

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
            device=self.device
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
            device=config.device
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
