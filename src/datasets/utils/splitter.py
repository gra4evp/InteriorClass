from src.config import RANDOM_SEED, SPLIT_CONFIG, MIN_VAL_TEST_PER_CLASS, CLASS_LABELS
from pathlib import Path
import random
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
from pydantic import BaseModel, Field, field_validator
from typing import Annotated
from decimal import Decimal



# Custom type for ratio values between 0 and 1 (inclusive)
RatioFloat = Annotated[float, Field(gt=0, le=1)]


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
    pass



class DatasetSplitter:
    """Класс для разделения датасета на train/val/test с учетом дисбаланса классов."""
    
    def __init__(
        self,
        class_labels: List[str],
        split_config: Dict[str, Dict[str, float]],
        random_seed: int | None = None
    ):
        """
        Args:
            class_labels: Список имен классов
            split_config: Конфигурация разделения (словарь с параметрами)
            random_seed: Seed для воспроизводимости
        """
        self.class_labels = class_labels
        self.split_config = self._validate_config(split_config)
        self.random_seed = random_seed
        
        if random_seed is not None:
            random.seed(random_seed)
        
        self._train_samples_count: int | None = None
        self._val_samples_count: int | None = None
        self._test_samples_count: int | None = None
    
    def __call__(
        self, 
        samples: List[Tuple[Path, str]],
        shuffle: bool = True
    ) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]], List[Tuple[Path, int]]]:
        """Основной метод для разделения данных."""
        label2samples = self._group_samples_by_class(samples)
        
        train, val, test = [], [], []
        for label, samples in label2samples.items():
            config = self.split_config[label]
            t, v, te = self._split(
                samples,
                train_ratio=config.train_ratio,
                val_ratio=config.val_ratio,
                test_ratio=config.test_ratio,
                min_samples=config.min_samples,
                shuffle=shuffle
            )
            train.extend(t)
            val.extend(v)
            test.extend(te)
        
        self._train_samples_count = len(train)
        self._val_samples_count = len(val)
        self._test_samples_count = len(test)
            
        return train, val, test
    
    def to_config(self) -> DatasetSplitterConfig:
        pass
    
    def _group_samples_by_class(
        self, 
        samples: List[Tuple[Path, str]]
    ) -> Dict[str, List[Tuple[Path, str]]]:
        """Группирует выборку по классам."""
        label2samples = defaultdict(list)
        for falepath, label in samples:
            label2samples[label].append((falepath, label))
        return label2samples
    
    @staticmethod
    def _validate_config(config: Dict) -> Dict[str, SplitConfig]:
        """Валидирует конфиг и преобразует в SplitConfig."""
        validated = {}
        for class_name, ratios in config.items():
            validated[class_name] = SplitConfig(**ratios)
        return validated

    @staticmethod
    def _split(
        samples: list,
        train_ratio: RatioFloat,
        val_ratio: RatioFloat,
        test_ratio: RatioFloat,
        min_samples: int,
        shuffle: bool = True
    ) -> Tuple[list, list, list]:
        """Разделяет выборки одного класса."""
        if shuffle:
            random.shuffle(samples)
        
        n_total = len(samples)
        
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Гарантируем минимальное количество в val/test
        n_val = max(n_val, min_samples)
        n_test = max(n_total - n_train - n_val, min_samples)
        
        train_set = samples[:n_train]
        val_set = samples[n_train : n_train + n_val]
        test_set = samples[n_train + n_val : n_train + n_val + n_test]
        
        return train_set, val_set, test_set
