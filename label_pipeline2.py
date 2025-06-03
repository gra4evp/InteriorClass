import os
import random
from PIL import Image
from tqdm import tqdm
import pandas as pd
from transformers import pipeline
import warnings
import csv
import shutil
from typing import Dict, List
from pathlib import Path


warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")


class InteriorClassifier:
    """Классификатор состояния интерйера с использованием Qwen2.5-VL модели через pipeline"""
    
    def __init__(self, model_name: str, base_prompt: str):
        """
        Инициализация классификатора
        
        Args:
            model_name: Название модели из Hugging Face Hub
        """
        # Инициализируем pipeline для работы с изображениями и текстом
        # Указываем:
        # - task: "image-text-to-text" (мультимодальная задача)
        # - model: имя модели
        # - device_map: "auto" для автоматического выбора GPU/CPU
        # - torch_dtype: float16 для экономии памяти
        self.pipe = pipeline(
            "image-text-to-text",
            model=model_name,
            device_map="auto",
            torch_dtype="auto",  # Автоматически выберет подходящий тип данных
            trust_remote_code=True  # Необходимо для кастомных моделей
        )
        
        # Системный промпт для классификации на английском (модель лучше понимает английский)
        self.base_prompt = base_prompt

    def build_few_shot_prompt(
        self,
        target_image_path: str,
        reference_images: Dict[str, List[str]],  # {"A0": ["path1.jpg", ...]}
        custom_prompt: str | None = None,
        max_examples_per_class: int = 2
    ) -> List[dict]:
        """
        Формирует few-shot запрос с динамически выбираемыми примерами
        
        Args:
            target_image_path: Путь к классифицируемому изображению
            reference_images: Словарь {класс: [пути_к_изображениям]}
            max_examples_per_class: Количество примеров на класс (по умолчанию 2)
            
        Returns:
            Готовый messages для pipeline
        """
        # Начало промпта
        content = [
            {"type": "text", "text": self.base_prompt},
            {"type": "text", "text": "\nReference examples:"}
        ]
        
        # Добавляем примеры для каждого класса
        for class_label, paths in reference_images.items():
            if not paths:
                continue

            selected_paths = paths[:max_examples_per_class]
            
            # Добавляем в контент
            for path in selected_paths:
                content.extend([
                    {"type": "image", "image": path},
                    {"type": "text", "text": f"Class: {class_label}"}
                ])
        
        # Добавляем целевое изображение
        content.extend([
            {"type": "text", "text": "\nNow classify THIS image:"},
            {"type": "image", "image": target_image_path},
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
        return "ERROR"
    
    def classify_image(
            self,
            image_path: str,
            custom_prompt: str | None = None,
            reference_images: Dict[str, List[str]] | None = None,
            max_examples_per_class: int = 2
        ) -> dict[str, str]:
        """
        Классифицирует изображение интерьера (объединяет get_model_response и parse_model_response)
        
        Args:
            image_path: Путь к изображению
            custom_prompt: Опциональный кастомный промпт
            
        Returns:
            Словарь с результатами:
            - image: имя файла
            - class: предсказанный класс
            - raw_response: полный ответ модели
        """
        if reference_images is not None:
            messages = self.build_few_shot_prompt(
                target_image_path=image_path,
                reference_images=reference_images,
                max_examples_per_class=max_examples_per_class
            )
        else:
            prompt = self.base_prompt
            if custom_prompt is not None:
                prompt = custom_prompt
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt}
                ]
            }]

        try:
            raw_response = self._get_model_response(imputs=messages)
            predicted_class = self.parse_model_response(raw_response)
            
            return {
                "image_filename": os.path.basename(image_path),
                "class": predicted_class,
                "raw_response": raw_response
            }
            
        except Exception as e:
            error_msg = f"Error processing {image_path}: {str(e)}"
            print(error_msg)
            return {
                "image_filename": os.path.basename(image_path),
                "class": "ERROR",
                "raw_response": error_msg
            }
    
    def process_directory(
            self,
            image_dir: str,
            output_dir: str,
            output_csv: str = "results.csv",
            extensions: tuple = (".jpg", ".png", ".jpeg")
        ) -> None:
        """
        Обрабатывает изображения и сортирует по папкам классов
        с постепенной записью результатов в CSV.
        """
        # Создаем корневую директорию для результатов
        os.makedirs(output_dir, exist_ok=True)
        
        # Путь к итоговому CSV
        csv_path = os.path.join(output_dir, output_csv)
        
        # Заголовки CSV
        fieldnames = ["image_filename", "class", "raw_response"]
        
        # Открываем CSV для постепенной записи
        with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            
            # Обрабатываем каждое изображение
            for img_file in tqdm(
                [f for f in os.listdir(image_dir) if f.lower().endswith(extensions)],
                desc="Обработка изображений"
            ):
                image_path = os.path.join(image_dir, img_file)
                result = self.classify_image(image_path)
                
                # Создаем папку класса (A0, A1... ERROR)
                class_dir = os.path.join(output_dir, result["class"])
                os.makedirs(class_dir, exist_ok=True)
                
                # Копируем изображение в папку класса
                shutil.copy2(
                    image_path,
                    os.path.join(class_dir, img_file)
                )
                
                # Записываем результат в CSV
                writer.writerow(result)
        
        print(f"\nГотово! Результаты в {output_dir}")
    
    def __call__(
            self,
            image_path: str,
            custom_prompt: str | None = None,
            reference_images: Dict[str, List[str]] | None = None,
            max_examples_per_class: int = 2
        ) -> dict[str, str]:

        # Проверяем существование файла
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Проверяем валидность изображения
        with Image.open(image_path) as img:
            img.verify()

        return self.classify_image(
            image_path=image_path,
            custom_prompt=custom_prompt,
            reference_images=reference_images,
            max_examples_per_class=max_examples_per_class
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
) -> Dict[str, List[str]]:
    """
    Генерирует словарь {класс: [пути_к_изображениям]} с возможностью ограничения количества
    и случайного выбора файлов.
    
    Args:
        ref_images_dirpath: Путь к директории с эталонными изображениями
        max_files_per_class: Максимальное количество файлов на класс (None - без ограничений)
        shuffle_files: Если True, выбирает случайные файлы (в пределах max_files_per_class)
    
    Returns:
        Словарь с алфавитно отсортированными классами и путями к изображениям
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
        
        # Преобразуем Path в строки и сортируем по имени файла
        image_paths = sorted([str(p) for p in image_paths], key=lambda x: Path(x).name)
        
        # Применяем случайный выбор и ограничение количества
        if shuffle_files:
            random.shuffle(image_paths)
        if max_files_per_class is not None:
            image_paths = image_paths[:max_files_per_class]
        
        # Сохраняем отсортированный результат
        class_label2ref_images[class_dir.name] = sorted(image_paths)
    
    return class_label2ref_images




def main():
    # Конфигурация
    config = {
        "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
        "image_dir": "data/cosmetic/cosmetic",
        "output_csv": "labeled_results.csv"
    }
    
    # Инициализация классификатора
    classifier = InteriorClassifier(config["model_name"])
    result = classifier.classify_image(image_path="/home/little-garden/CodeProjects/InteriorClass/data/cosmetic/cosmetic+/153001513_5.jpg")
    for key, value in result.items():
        print(key, value)
    # # Запуск обработки
    # classifier.process_directory(
    #     image_dir=config["image_dir"],
    #     output_csv=config["output_csv"]
    # )


if __name__ == "__main__":
    main()
