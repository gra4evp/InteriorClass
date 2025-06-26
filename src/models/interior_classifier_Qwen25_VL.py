import os
import random
from PIL import Image
from tqdm import tqdm
import pandas as pd
from transformers import pipeline
import warnings
import csv
import shutil
from typing import Dict, List, Any
from pathlib import Path
import torch
import traceback


warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")


class InteriorImageClassifier:
    """Классификатор состояния изображений интерйера с использованием Qwen2.5-VL модели через pipeline"""
    
    def __init__(
            self,
            model_name: str,
            base_prompt: str,
            device_map: str | Dict[str, int | str | torch.device]= "auto",  # GPU/CPU
            torch_dtype: str | torch.dtype | None = "auto"  # "auto" is torch.bfloat16
        ):
        """
        Инициализация классификатора
        
        Args:
            model_name: Название модели из Hugging Face Hub
        """
        self.pipe = pipeline(
            task="image-text-to-text",  # (мультимодальная задача)
            model=model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,  # torch.float16,  # Автоматически выберет подходящий тип данных
            quantization_config={
                "quant_method": "bitsandbytes",
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
            },
            attn_implementation="flash_attention_2",
            use_fast=True,  # `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor.
            trust_remote_code=True  # Необходимо для кастомных моделей
        )
        
        # Системный промпт для классификации (модель лучше понимает английский)
        self.base_prompt = base_prompt

    def build_few_shot_prompt(
        self,
        target_image: Image.Image,
        class_label2ref_images: Dict[str, List[Image.Image]],
        max_examples_per_class: int = 1
    ) -> List[dict]:
        """
        Формирует few-shot запрос с динамически выбираемыми примерами
        
        Args:
            target_image_path: Путь к классифицируемому изображению
            class_label2ref_images: Словарь {метка класса: [PIL изображения]}
            max_examples_per_class: Количество примеров на класс (по умолчанию 1)
            
        Returns:
            Готовый messages для pipeline
        """
        # Начало промпта
        content = [
            {"type": "text", "text": self.base_prompt},
            {"type": "text", "text": "\nReference examples:"}
        ]
        
        # Добавляем примеры для каждого класса
        for class_label, images in class_label2ref_images.items():
            if images:
                selected_images = images[:max_examples_per_class]
                
                # Добавляем в контент
                for image in selected_images:
                    content.extend([
                        {"type": "image", "image": image},
                        {"type": "text", "text": f"Class: {class_label}"}
                    ])
        
        # Добавляем целевое изображение
        content.extend([
            {"type": "text", "text": "\nNow classify THIS image:"},
            {"type": "image", "image": target_image},
            {"type": "text", "text": "Answer ONLY with class label:"}
        ])
        
        return [{"role": "user", "content": content}]
    
    def parse_model_response(self, raw_response: str) -> str:
        """
        Парсит ответ модели и извлекает класс интерьера
        
        Args:
            raw_response: Сырой текстовый ответ от модели
            
        Returns:
            Класс интерьера (A0, A1, ..., D1) или "ERROR" если не удалось распознать
        """
        for token in ["A0", "A1", "B0", "B1", "C0", "C1", "D0", "D1"]:
            if token in raw_response:
                return token
        for token in ["1", "0"]:
            if token in raw_response:
                return token
        return "ERROR"
    
    def classify_image(
            self,
            image: Image.Image,
            custom_prompt: str | None = None,
            class_label2ref_images: Dict[str, List[Image.Image]] | None = None,
            max_examples_per_class: int = 1,
            max_new_tokens: int = 10
        ) -> dict[str, str]:
        """
        Классифицирует изображение интерьера (объединяет get_model_response и parse_model_response)
        
        Args:
            image: Путь к изображению
            custom_prompt: Опциональный кастомный промпт
            
        Returns:
            Словарь с результатами:
            - image: имя файла
            - class: предсказанный класс
            - raw_response: полный ответ модели
        """
        if class_label2ref_images is not None:
            messages = self.build_few_shot_prompt(
                target_image=image,
                class_label2ref_images=class_label2ref_images,
                max_examples_per_class=max_examples_per_class
            )
        else:
            prompt = self.base_prompt
            if custom_prompt is not None:
                prompt = custom_prompt
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }]

        try:
            raw_response = self._get_model_response(inputs=messages, max_new_tokens=max_new_tokens)
            predicted_class = self.parse_model_response(raw_response)
            return {"class": predicted_class, "raw_response": raw_response}
            
        except Exception as e:
            traceback.print_exc()
            error_msg = f"Error during processing: {str(e)}"
            print(error_msg)
            # return {"class": "ERROR", "raw_response": error_msg}
    
    def __call__(
            self,
            image: Image.Image,
            custom_prompt: str | None = None,
            class_label2ref_images: Dict[str, List[Image.Image]] | None = None,
            max_examples_per_class: int = 1,
            max_new_tokens: int = 10
        ) -> dict[str, str]:
        """
        Прямой вызов классификатора
        
        Args:
            image: PIL Image объект
            custom_prompt: Опциональный кастомный промпт
            class_label2ref_images: Словарь с референсными изображениями
            max_examples_per_class: Максимальное количество примеров на класс
            
        Returns:
            Словарь с результатами классификации
        """
        # Проверяем только тип изображения
        if not isinstance(image, Image.Image):
            raise ValueError(f"Expected PIL.Image, got {type(image)}")

        return self.classify_image(
            image=image,
            custom_prompt=custom_prompt,
            class_label2ref_images=class_label2ref_images,
            max_examples_per_class=max_examples_per_class,
            max_new_tokens=max_new_tokens
        )

    def _get_model_response(
            self,
            inputs: List[Dict],
            max_new_tokens: int = 10,
            do_samples: bool = False,
            temperature: float = 0.01
        ) -> str:
        """
        Инкапсулированная логика получения ответа
        ====================== FULL RESPONSE EXAMPLE FOR list inputs messages with len = 1 ===========================
        [
          {
            "input_text":[
              {
                "role":"user",
                "content":[
                  {
                    "type":"image",
                    "image":"<image_path>"
                  },
                  {
                    "type":"text",
                    "text":"<my_prompt>"
                  }
                ]
              }
            ],
            "generated_text":[
              {
                "role":"user",
                "content":[
                  {
                    "type":"image",
                    "image":"<image_path>"
                  },
                  {
                    "type":"text",
                    "text":"<my_prompt>"
                  }
                ]
              },
              {
                "role":"assistant",
                "content":"B1"
              }
            ]
          }
        ]
        ===========================================================================================
        """
        outputs = self.pipe(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_samples,
            temperature=temperature
        )

        response_content = ""
        if outputs:
            generated_text: list[dict] = outputs[0]["generated_text"]
            assistant_message = generated_text[-1]
            response_content = assistant_message["content"]
        
        return response_content



def generate_class_label2ref_images(
    ref_images_dirpath: Path,
    max_files_per_class: int | None = None,
    shuffle_files: bool = True
) -> Dict[str, List[Image.Image]]:
    """
    Генерирует словарь {метка_класса: [изображения]} с возможностью ограничения количества
    и случайного выбора файлов.
    
    Args:
        ref_images_dirpath: Путь к директории с эталонными изображениями
        max_files_per_class: Максимальное количество файлов на класс (None - без ограничений)
        shuffle_files: Если True, выбирает случайные файлы (в пределах max_files_per_class)
    
    Returns:
        Словарь с алфавитно отсортированными классами и изображениями
    """
    class_label2ref_images = {}
    
    # Получаем и сортируем папки классов по имени
    class_dirs = sorted(
        [d for d in ref_images_dirpath.iterdir() if d.is_dir()],
        key=lambda x: x.name
    )
    
    for class_dir in class_dirs:
        # Находим все подходящие файлы в папке класса
        image_paths = []
        for ext in ("*.jpg", "*.png"):
            image_paths.extend(class_dir.glob(ext))
        
        # Применяем случайный выбор и ограничение количества
        if shuffle_files:
            random.shuffle(image_paths)
        if max_files_per_class is not None:
            image_paths = image_paths[:max_files_per_class]

        images = []
        for path in sorted(image_paths, key=lambda x: Path(x).name):
            images.append(Image.open(path))

        # Сохраняем отсортированный результат и сортируем по имени файла
        class_label2ref_images[class_dir.name] = images
    
    return class_label2ref_images
