from pathlib import Path
import warnings
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import torch
from typing import List
from PIL import Image
# from sklearn.metrics import classification_report
import albumentations as A


# Убираем только конкретное предупреждение Pillow о палитровых изображениях
warnings.filterwarnings(
    "ignore", 
    category=UserWarning, 
    message="Palette images with Transparency.*",
    module="PIL.Image"
)


class InteriorDataset(Dataset):
    """Датасет с поддержкой Albumentations аугментаций (PIL.Image версия)"""
    
    CLASSES = ["A0", "A1", "B0", "B1", "C0", "C1", "D0", "D1"]
    
    def __init__(self, samples, transform=None, mode='train'):
        """
        Args:
            samples: список кортежей (путь_к_изображению, индекс_класса)
            transform: albumentations трансформации
            mode: режим работы ('train'/'val'/'test')
        """
        self.samples = samples
        self.transform = transform
        self.mode = mode
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        
        # Загрузка через PIL
        image = Image.open(img_path).convert('RGB')
        
        # Конвертация в numpy array для Albumentations
        image_np = np.array(image)
        
        if self.transform:
            augmented = self.transform(image=image_np)
            image_np = augmented['image']
            
            # Если трансформации включают ToTensorV2, то image_np уже будет тензором
            if isinstance(image_np, torch.Tensor):
                return image_np, class_idx
            
            # Иначе конвертируем в тензор вручную
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
            return image_tensor, class_idx
            
        # Без трансформаций - конвертируем в тензор
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
        return image_tensor, class_idx

    @classmethod
    def collect_samples(
        cls, 
        dataset_dir: Path, 
        extensions: tuple[str, ...] = ('.jpg', '.jpeg', '.png')
    ) -> List[tuple[Path, int]]:
        """Collect samples with both path and numerical index.
        
        Args:
            dataset_dir: Root directory containing class folders
            extensions: Allowed file extensions (default: .jpg, .jpeg, .png)
                
        Returns:
            List of (image_path, class_index) tuples
        """
        samples = []
        allowed_extensions = {ext.lower() for ext in extensions}
        
        for class_dir in tqdm(sorted(dataset_dir.iterdir()), desc="Collecting samples..."):
            if class_dir.is_dir() and class_dir.name in cls.CLASSES:
                image_paths = sorted(class_dir.iterdir())
                class_idx = cls.CLASSES.index(class_dir.name)

                class_samples = []
                for filepath in image_paths:
                    if filepath.is_file() and filepath.suffix.lower() in allowed_extensions:
                        class_samples.append((filepath, class_idx))
                
                samples.extend(class_samples)
        
        return samples


# def get_transforms(mode='train', img_size=380):
#     """Аугментации для разных этапов"""
#     if mode == 'train':
#         return A.Compose([
#             A.Resize(img_size, img_size),
#             A.HorizontalFlip(p=0.5),
#             A.Affine(
#                 translate_percent=0.1,  # аналог shift_limit
#                 scale=(0.85, 1.15),     # аналог scale_limit
#                 rotate=(-30, 30),       # аналог rotate_limit
#                 p=0.5
#             ),
#             A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
#             A.CoarseDropout(
#                 num_holes_range=(3, 6),     # Диапазон количества "дыр" (бывший max_holes)
#                 hole_height_range=(16, 32),  # Диапазон высоты (бывший max_height)
#                 hole_width_range=(16, 32),   # Диапазон ширины (бывший max_width)
#                 fill=0,                # Значение для заливки (0 для чёрного)
#                 p=0.3
#             ),
#             A.Normalize(
#                 mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225]
#             ),
#             A.ToTensorV2()
#         ])
#     else:  # val/test
#         return A.Compose([
#             A.Resize(img_size, img_size),
#             A.Normalize(
#                 mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225]
#             ),
#             A.ToTensorV2()
#         ])


def get_transforms(mode='train', img_size=380):
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
