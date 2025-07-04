# src/datasets/utils/collector.py
from pathlib import Path
from typing import List, Tuple, Any
from pydantic import BaseModel
from tqdm import tqdm


class DataCollectorConfig(BaseModel):
    dataset_dir: Path
    class_labels: List[str]
    file_extensions: tuple[str, ...] = ('.jpg', '.jpeg', '.png')
    samples_count: int | None = None


class DataCollector:
    def __init__(
            self,
            dataset_dir: Path,
            class_labels: List[str],
            file_extensions: tuple[str, ...] = ('.jpg', '.jpeg', '.png')
        ):

        self.dataset_dir = dataset_dir
        self.class_labels = class_labels
        self.file_extensions = file_extensions
        self._samples_count: int | None = None

    def __call__(self) -> List[tuple[Path, str]]:
        """Collect samples with both path and numerical index.
        
        Args:
            dataset_dir: Root directory containing class folders
            extensions: Allowed file extensions (default: .jpg, .jpeg, .png)
                
        Returns:
            List of (filepath, class_label) tuples
        """
        
        allowed_extensions = {ext.lower() for ext in self.file_extensions}
        class_dirs = sorted(f for f in self.dataset_dir.iterdir() if f.is_dir())

        samples = []
        for class_dir in tqdm(class_dirs, desc="Collecting samples..."):
            label = class_dir.name
            
            if label in self.class_labels:
                class_samples = []
                filepaths = sorted(f for f in class_dir.iterdir() if f.is_file())
                for filepath in filepaths:
                    if filepath.suffix.lower() in allowed_extensions:
                        class_samples.append((filepath, label))
                
                samples.extend(class_samples)
        
        self._samples_count = len(samples)
        
        return samples

    @classmethod
    def from_config(cls, config: DataCollectorConfig):
        kwargs = config.model_dump()
        _ = kwargs.pop('samples_count', default=None)
        return cls(**config.model_dump())
    
    def to_config(self) -> DataCollectorConfig:
        return DataCollectorConfig(
            dataset_dir=self.dataset_dir,
            class_labels=self.class_labels,
            file_extensions=self.file_extensions,
            samples_count=self._samples_count
        )

