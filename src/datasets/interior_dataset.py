# src/datasets/interior_dataset.py
from pathlib import Path
import warnings
from torch.utils.data import Dataset
import numpy as np
import torch
from PIL import Image
import albumentations as A
from src.schemas import SampleItem, DatasetConfig


# Убираем только конкретное предупреждение Pillow о палитровых изображениях
warnings.filterwarnings(
    "ignore", 
    category=UserWarning, 
    message="Palette images with Transparency.*",
    module="PIL.Image"
)


class InteriorDataset(Dataset):
    """
    Датасет с поддержкой Albumentations аугментаций (PIL.Image версия)
    """
    
    def __init__(
            self,
            transforms: A.Compose | None = None,
            transforms_filepath: Path | None = None
        ):
        """
        Args:
            samples: список кортежей (путь_к_изображению, индекс_класса)
            transform: albumentations трансформации
        """
        self.transforms = transforms
        self.transforms_filepath = transforms_filepath
        self.sample_items: list[SampleItem] | None = None
    
    def prepare(self, sample_items: list[SampleItem]):
        self.sample_items = sample_items
        
    def __len__(self):
        if self.sample_items is None:
            return 0
        return len(self.sample_items)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, int]:
        if self.sample_items is None:
            raise RuntimeError("Dataset not prepared!")
        
        s_item = self.sample_items[idx]
        
        # Загрузка через PIL
        image_pil = Image.open(s_item.filepath).convert('RGB')
        
        # Конвертация в numpy array для Albumentations
        image_np = np.array(image_pil)
        
        if self.transforms is not None:
            augmented = self.transforms(image=image_np)
            image = augmented['image']
            
            # Если трансформации включают ToTensorV2, то image уже будет тензором
            if isinstance(image, torch.Tensor):
                return image, s_item.label, s_item.class_idx
            
            # Иначе конвертируем в тензор вручную
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
            return image_tensor, s_item.label, s_item.class_idx
            
        # Без трансформаций - конвертируем в тензор
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
        return image_tensor, s_item.label, s_item.class_idx

    def to_config(self) -> DatasetConfig:
        if not self.transforms.is_serializable():
            raise ValueError
        
        if self.transforms_filepath is not None:
            A.save(self.transforms, self.transforms_filepath)
        
        return DatasetConfig(transform_filepath=self.transforms_filepath)
    
    @classmethod
    def from_config(cls, config: DatasetConfig) -> 'InteriorDataset':
        transform = None
        if config.transforms_filepath is not None:
            transform = A.load(config.transforms_filepath)
        return cls(transform=transform, transforms_filepath=config.transforms_filepath)


def get_transforms(mode='train', img_size=380) -> A.Compose:
    if mode == 'train':
        return A.Compose([
            # Обязательные для всех изображений
            A.Resize(img_size, img_size),
            
            # Группа 1: Геометрические искажения (выбираем ТОЛЬКО ОДНО)
            A.OneOf([
                A.Affine(
                    translate_percent=0.1,
                    scale=(0.85, 1.15),
                    rotate=(-30, 30),
                    p=1.0
                ),
                A.Perspective(scale=(0.05, 0.1), p=1.0),
                A.RandomSizedCrop(
                    min_max_height=(int(img_size*0.7), img_size),
                    size=(img_size, img_size),
                    p=1.0
                ),
            ], p=0.7),  # 70% вероятность применить геометрию
            
            # Группа 2: Цветовые искажения (выбираем ТОЛЬКО ОДНО)
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, p=1.0),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                A.CLAHE(clip_limit=2, p=1.0),
            ], p=0.5),  # 50% вероятность применить цвет
            
            # Группа 3: Шумы/Артефакты (выбираем ТОЛЬКО ОДНО)
            A.OneOf([
                A.GaussNoise(std_range=(0.1, 0.2), p=1.0),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
                A.Blur(blur_limit=3, p=1.0),
                A.ImageCompression(quality_range=(20, 40), p=1.0),
            ], p=0.3),  # 30% вероятность применить шум
            
            # Группа 4: Локальные повреждения (выбираем ТОЛЬКО ОДНО)
            A.OneOf([
                A.CoarseDropout(
                    num_holes_range=(3, 6),     # Диапазон количества "дыр" (бывший max_holes)
                    hole_height_range=(16, 32),  # Диапазон высоты (бывший max_height)
                    hole_width_range=(16, 32),   # Диапазон ширины (бывший max_width)
                    fill=0,                # Значение для заливки (0 для чёрного)
                    p=0.3
                ),
                A.RandomShadow(
                    shadow_roi=(0, 0, 1, 1),
                    num_shadows_limit=(1, 3),
                    shadow_dimension=5,
                    p=1.0
                ),
                # Для синтетических повреждений используй внешние маски:
                # A.Lambda(name='mold_effect', ...)
            ], p=0.4),  # 40% вероятность
            
            # Всегда применяемые (без конфликтов)
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            A.ToTensorV2()
        ])
    else:  # val/test
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            A.ToTensorV2()
        ])
