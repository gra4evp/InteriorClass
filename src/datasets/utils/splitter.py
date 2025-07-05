# src/datasets/utils/splitter.py
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, DefaultDict
from src.schemas import DatasetSplitterConfig, RatioFloat, SampleItem, DatasetSplit
import random


class DatasetSplitter:
    """Класс для разделения датасета на train/val/test с учетом дисбаланса классов."""
    
    def __init__(
        self,
        splits: dict[str, dict[str, int | float]],
        random_seed: int | None = None
    ):
        """
        Args:
            split_dict: Конфигурация разделения (словарь с параметрами)
            random_seed: Seed для воспроизводимости
        """
        self.splits = splits
        self.random_seed = random_seed
        
        if random_seed is not None:
            random.seed(random_seed)
        
        self._config = self.to_config()
        self._train_samples_count: int | None = None
        self._val_samples_count: int | None = None
        self._test_samples_count: int | None = None
    
    def __call__(self, sample_items: list[SampleItem], shuffle: bool = True) -> DatasetSplit:
        """Основной метод для разделения данных."""
        label2sample_items = self._group_samples_by_class(sample_items)
        
        train, val, test = [], [], []
        for label, s_items in label2sample_items.items():
            split = self._config.splits[label]
            train_set, val_set, test_set = self._split(
                s_items,
                train_ratio=split.train_ratio,
                val_ratio=split.val_ratio,
                test_ratio=split.test_ratio,
                min_samples=split.min_samples,
                shuffle=shuffle
            )

            train.extend(train_set)
            val.extend(val_set)
            test.extend(test_set)
        
        self._train_samples_count = len(train)
        self._val_samples_count = len(val)
        self._test_samples_count = len(test)
            
        return train, val, test
    
    def to_config(self) -> DatasetSplitterConfig:
        return DatasetSplitterConfig(
            splits=self.splits,
            random_seed=self.random_seed,
            train_samples_count=self._train_samples_count,
            val_samples_count=self._val_samples_count,
            test_samples_count=self._test_samples_count
        )

    @classmethod
    def from_config(cls, config: DatasetSplitterConfig):
        kwargs = config.model_dump()
        # Deleting the service fields that are needed when saving
        _ = kwargs.pop('train_samples_count', default=None)
        _ = kwargs.pop('val_samples_count', default=None)
        _ = kwargs.pop('test_samples_count', default=None)
        return cls(**kwargs)
    
    def _group_samples_by_class(self, samples: List[SampleItem]) -> dict[str, List[SampleItem]]:
        """Группирует выборку по классам."""
        label2samples: DefaultDict[str, list[SampleItem]] = defaultdict(list)
        for sample in samples:
            label2samples[sample.label].append(sample)
        return dict(label2samples)

    @staticmethod
    def _split(
        samples: list[SampleItem],
        train_ratio: RatioFloat,
        val_ratio: RatioFloat,
        test_ratio: RatioFloat,
        min_samples: int,
        shuffle: bool = True
    ) -> DatasetSplit:
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
