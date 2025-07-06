# src/datasets/utils/collector.py
from pathlib import Path
from src.schemas import SampleItem, SampleCollectorConfig
from tqdm import tqdm


class SampleCollector:
    def __init__(
            self,
            dataset_dir: Path,
            class_labels: list[str],
            file_extensions: tuple[str, ...] = ('.jpg', '.jpeg', '.png')
        ):

        self.dataset_dir = dataset_dir
        self.class_labels = class_labels
        self.file_extensions = file_extensions
        self._samples_count: int | None = None
        self._label2idx_map = {label: idx for idx, label in enumerate(class_labels)}

    def __call__(self) -> list[SampleItem]:
        """
        Collect samples
        
        Args:
            dataset_dir: Root directory containing class folders
            extensions: Allowed file extensions (default: .jpg, .jpeg, .png)
                
        Returns:
            List of SampleItem(filepath, class_label, class_idx)
        """
        
        allowed_extensions = {ext.lower() for ext in self.file_extensions}
        class_dirs = sorted(f for f in self.dataset_dir.iterdir() if f.is_dir())

        sample_items = []
        for class_dir in tqdm(class_dirs, desc="Collecting samples..."):
            label = class_dir.name

            if label in self._label2idx_map:
                class_sample_items = []
                filepaths = sorted(f for f in class_dir.iterdir() if f.is_file())
                for filepath in filepaths:
                    if filepath.suffix.lower() in allowed_extensions:
                        
                        class_sample_items.append(
                            SampleItem(
                                filepath=filepath,
                                label=label,
                                class_idx=self._label2idx_map[label]
                            )
                        )
                
                sample_items.extend(class_sample_items)
        
        self._samples_count = len(sample_items)
        
        return sample_items

    def to_config(self) -> SampleCollectorConfig:
        return SampleCollectorConfig(
            dataset_dir=self.dataset_dir,
            class_labels=self.class_labels,
            file_extensions=self.file_extensions,
            samples_count=self._samples_count
        )

    @classmethod
    def from_config(cls, config: SampleCollectorConfig):
        kwargs = config.model_dump()
        # Deleting the service fields that are needed when saving
        _ = kwargs.pop('samples_count', None)
        return cls(**kwargs)
