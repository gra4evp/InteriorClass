# src/shemas/configs.py
from pathlib import Path
from typing import List, Literal, Any
from decimal import Decimal
from src.schemas.types import RatioFloat

from pydantic import BaseModel, Field, field_validator


class SampleCollectorConfig(BaseModel):
    dataset_dir: Path
    class_labels: List[str]
    file_extensions: tuple[str, ...] = ('.jpg', '.jpeg', '.png')
    samples_count: int | None = None


class SplitConfig(BaseModel):
    """Configuration for splitting dataset samples for a single class.
    
    Attributes:
        train_ratio: Ratio of samples for training (0 < value ≤ 1)
        val_ratio: Ratio of samples for validation (0 < value ≤ 1)
        test_ratio: Ratio of samples for testing (0 < value ≤ 1)
        min_samples: Minimum number of samples guaranteed for validation and testing
    """
    train_ratio: RatioFloat
    val_ratio: RatioFloat
    test_ratio: RatioFloat
    min_samples: int = Field(default=20, ge=1)  # Minimum samples in val/test sets

    @field_validator('train_ratio', 'val_ratio', 'test_ratio', mode='before')
    @classmethod
    def round_values(cls, v: float) -> float:
        """Round float values to 2 decimal places for precision.
        
        Args:
            v: Input value to round
            
        Returns:
            Value rounded to 2 decimal places
        """
        if isinstance(v, float):
            return float(round(Decimal(str(v)), 2))
        return v

    @field_validator('test_ratio')
    @classmethod
    def validate_split(cls, v: float, values) -> float:
        """Validate that the sum of all ratios doesn't exceed 1.
        
        Args:
            v: Test set ratio value
            values: Other field values
            
        Returns:
            Original test ratio if validation passes
            
        Raises:
            ValueError: If sum of train/val/test ratios exceeds 1
        """
        if 'train_ratio' in values.data and 'val_ratio' in values.data:
            total = values.data['train_ratio'] + values.data['val_ratio'] + v
            if total > 1.01:  # Small tolerance for floating point rounding
                raise ValueError(f"Sum of train/val/test ratios must not exceed 1. Got: {total}")
        return v


class DatasetSplitterConfig(BaseModel):
    splits: dict[str, SplitConfig]
    random_seed: int = 42
    train_samples_count: int | None = None
    val_samples_count: int | None = None
    test_samples_count: int | None = None


class DatasetConfig(BaseModel):
    transforms_filepath: Path | None


class DataLoaderConfig(BaseModel):
    batch_size: int
    shuffle: bool
    num_workers: int
    pin_memory: bool


class HeadConfig(BaseModel):
    hidden_dim: int = Field(512, description="Размер скрытого слоя head")
    dropout: float = Field(0.3, description="Dropout в head")
    activation: Literal['relu', 'gelu'] = Field('relu', description="Активация в head")


class ModelConfig(BaseModel):
    backbone_name: str = Field('efficientnet_b3', description="Название backbone модели")
    num_classes: int = Field(8, description="Количество классов")
    pretrained: bool = Field(True, description="Использовать ли pretrain")
    head: HeadConfig | None = None


class CriterionConfig(BaseModel):
    name: str
    params: dict[str, Any]


class OptimizerConfig(BaseModel):
    name: str  # "Adam", "SGD", etc.
    params: dict[str, Any]  # {"lr": 0.001, "weight_decay": 0.01}


class SchedulerConfig(BaseModel):
    name: str  # "StepLR", "CosineAnnealingLR", etc.
    params: dict[str, Any]  # {"step_size": 30, "gamma": 0.1}


# class HyperParametersConfig(BaseModel):
#     batch_size: int
#     epochs: int
#     img_size: int
#     random_seed: int
#     device: str
#     criterion: str  # "CrossEntropyLoss", "MSELoss", etc.
#     optimizer: OptimizerConfig
#     scheduler: SchedulerConfig | None = None


class TrainerConfig(BaseModel):
    model_config: ModelConfig
    criterion_config: CriterionConfig
    optimizer_config: OptimizerConfig
    scheduler_config: SchedulerConfig | None
    epochs: int
    device: str
    exp_results_dir: Path
    train_loader_config: DataLoaderConfig | None
    val_loader_config: DataLoaderConfig | None
    test_loader_config: DataLoaderConfig | None


class ExperimentConfig(BaseModel):
    exp_dir: Path
    exp_number: int
    epochs: int
    img_size: int
    start_lr: float
    random_seed: int
    collector_config: SampleCollectorConfig
    splitter_config: DatasetSplitterConfig
    train_dataset_config: DatasetConfig
    val_dataset_config: DatasetConfig
    test_dataset_config: DatasetConfig
    trainer_config: TrainerConfig
