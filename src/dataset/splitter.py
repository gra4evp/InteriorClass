from src.config import RANDOM_SEED, SPLIT_RATIO, MIN_VAL_TEST_PER_CLASS, CLASS_LABELS
from pathlib import Path
import random
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
from pydantic import BaseModel, Field, field_validator
from typing import Annotated
from decimal import Decimal


random.seed(RANDOM_SEED)

# Custom type for ratio values between 0 and 1 (inclusive)
RatioFloat = Annotated[float, Field(gt=0, le=1)]


class SplitConfig(BaseModel):
    """Configuration for splitting dataset samples for a single class.
    
    Attributes:
        train: Ratio of samples for training (0 < value ≤ 1)
        val: Ratio of samples for validation (0 < value ≤ 1)
        test: Ratio of samples for testing (0 < value ≤ 1)
        min_samples: Minimum number of samples guaranteed for validation and testing
    """
    train: RatioFloat
    val: RatioFloat
    test: RatioFloat
    min_samples: int = Field(default=20, ge=1)  # Minimum samples in val/test sets

    @field_validator('train', 'val', 'test', mode='before')
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

    @field_validator('test')
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
        if 'train' in values.data and 'val' in values.data:
            total = values.data['train'] + values.data['val'] + v
            if total > 1.01:  # Small tolerance for floating point rounding
                raise ValueError(f"Sum of train/val/test ratios must not exceed 1. Got: {total}")
        return v


class DatasetSplitter:
    """Класс для разделения датасета на train/val/test с учетом дисбаланса классов."""
    
    def __init__(
        self,
        class_labels: List[str],
        split_config: Dict[str, Dict[str, float]],
        random_seed: Optional[int] = None
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
    
    def split(
        self, 
        samples: List[Tuple[Path, int]],
        shuffle: bool = True
    ) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]], List[Tuple[Path, int]]]:
        """Основной метод для разделения данных."""
        class_to_samples = self._group_samples_by_class(samples)
        
        train, val, test = [], [], []
        for class_name, class_samples in class_to_samples.items():
            cfg = self.split_config[class_name]
            t, v, te = self._split_class_samples(class_samples, cfg, shuffle=shuffle)
            train.extend(t)
            val.extend(v)
            test.extend(te)
            
        return train, val, test
    
    def _group_samples_by_class(
        self, 
        samples: List[Tuple[Path, int]]
    ) -> Dict[str, List[Tuple[Path, int]]]:
        """Группирует выборки по классам."""
        class_to_samples = defaultdict(list)
        for img_path, class_idx in samples:
            class_name = self.class_labels[class_idx]
            class_to_samples[class_name].append((img_path, class_idx))
        return class_to_samples
    
    @staticmethod
    def _validate_config(config: Dict) -> Dict[str, SplitConfig]:
        """Валидирует конфиг и преобразует в SplitConfig."""
        validated = {}
        for class_name, ratios in config.items():
            validated[class_name] = SplitConfig(**ratios)
        return validated

    @staticmethod
    def _split_class_samples(
        samples: List[Tuple[Path, int]], 
        config: SplitConfig,
        shuffle: bool = True
    ) -> Tuple[List, List, List]:
        """Разделяет выборки одного класса."""
        if shuffle:
            random.shuffle(samples)
        
        n_total = len(samples)
        
        n_train = int(n_total * config.train)
        n_val = int(n_total * config.val)
        
        # Гарантируем минимальное количество в val/test
        n_val = max(n_val, config.min_samples)
        n_test = max(n_total - n_train - n_val, config.min_samples)
        
        train = samples[:n_train]
        val = samples[n_train : n_train + n_val]
        test = samples[n_train + n_val : n_train + n_val + n_test]
        
        return train, val, test


# Пример использования:
if __name__ == "__main__":
    # 1. Собираем все пути
    data_root = Path("./data/interior_dataset")
    samples = []
    for class_dir in data_root.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            class_idx = InteriorDataset.CLASSES.index(class_name)
            for img_path in class_dir.glob("*.jpg"):
                samples.append((img_path, class_idx))

    # Создание сплиттера
    splitter = DatasetSplitter(
        class_labels=["A0", "A1", "B0", "B1", "C0", "C1", "D0", "D1"],
        split_config=SPLIT_RATIO,
        random_seed=RANDOM_SEED
    )

    # Разделение данных
    train_samples, val_samples, test_samples = splitter.split(samples)