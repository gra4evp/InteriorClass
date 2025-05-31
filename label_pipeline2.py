import os
from PIL import Image
from tqdm import tqdm
import pandas as pd
from transformers import pipeline
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")


class InteriorClassifier:
    """Классификатор состояния интерйера с использованием Qwen2.5-VL модели через pipeline"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
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
        self.prompt_template = """
        Analyze the interior photo and classify its condition. Choose ONLY ONE class from:
        - A0: No finish (bare walls, concrete)
        - A1: Needs major renovation (severe wear)
        - B0: Basic finish (white-box)
        - B1: Needs cosmetic repairs
        - C0: Good condition
        - C1: Excellent condition (with furniture)
        - D0: Premium renovation (design project)
        - D1: Luxury (exclusive)
        Respond STRICTLY with ONLY the class label (e.g. "C0").
        """
    
    def get_model_response(self, image_path: str, custom_prompt: str | None = None) -> str:
        """
        Получает сырой ответ от модели для изображения
        
        Args:
            image_path: Путь к изображению
            custom_prompt: Опциональный кастомный промпт
            
        Returns:
        =========================== FULL RESPONSE EXAMPLE FOR list input messages with len = 1 ================================
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
        ==================================================================================================================      
        Полный текстовый ответ от модели
            
        Raises:
            FileNotFoundError: Если изображение не найдено
            Exception: При других ошибках обработки изображения
        """
        # Проверяем существование файла
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Проверяем валидность изображения
        with Image.open(image_path) as img:
            img.verify()
        
        # Формируем сообщение для модели
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": custom_prompt or self.prompt_template}
                ]
            }
        ]
        
        # Получаем ответ от модели
        response = self.pipe(
            messages,
            max_new_tokens=10,
            do_sample=False,
            temperature=0.01
        )
        
        #return response[0]["generated_text"] if response else ""
        return response
    
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
    
    def classify_image(self, image_path: str, custom_prompt: str | None = None) -> dict[str, str]:
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
        try:
            raw_response = self.get_model_response(image_path, custom_prompt)
            predicted_class = self.parse_model_response(raw_response)
            
            return {
                "image": os.path.basename(image_path),
                "class": predicted_class,
                "raw_response": raw_response
            }
            
        except Exception as e:
            error_msg = f"Error processing {image_path}: {str(e)}"
            print(error_msg)
            return {
                "image": os.path.basename(image_path),
                "class": "ERROR",
                "raw_response": error_msg
            }
    
    def process_directory(
        self, 
        image_dir: str, 
        output_csv: str = "results.csv",
        extensions: tuple = (".jpg", ".png", ".jpeg")
    ) -> pd.DataFrame:
        """Обработка всех изображений в директории"""
        results = []
        image_files = [
            f for f in os.listdir(image_dir) 
            if f.lower().endswith(extensions)
        ]
        
        for img_file in tqdm(image_files, desc="Обработка изображений"):
            image_path = os.path.join(image_dir, img_file)
            result = self.classify_image(image_path)
            results.append(result)
        
        # Сохраняем результаты
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"Результаты сохранены в {output_csv}")
        return df


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
