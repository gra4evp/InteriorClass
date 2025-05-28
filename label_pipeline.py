import os
from PIL import Image
from tqdm import tqdm
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer


class InteriorClassifier:
    """Классификатор состояния интерьера с помощью Qwen-VL"""
    
    def __init__(self, model_name: str = "Qwen/Qwen-VL-Chat"):
        """Инициализация модели и токенизатора"""
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            bnb_4bit_compute_dtype="float16"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Системный промпт для классификации
        self.prompt = """
        Оцени состояние ремонта на фото. Выбери один класс:
        - A0: Без отделки (голые стены, бетон)
        - A1: Требует кап. ремонта (сильный износ)
        - B0: Под чистовую (white-box)
        - B1: Требует косметического ремонта
        - C0: Хорошее состояние
        - C1: Отличное состояние (с мебелью)
        - D0: Евроремонт (дизайн-проект)
        - D1: Luxury (эксклюзив)
        Ответь ТОЛЬКО классом (например, "C0").
        """
    
    def classify_image(self, image_path: str) -> dict[str, str]:
        """Классификация одного изображения"""
        try:
            image = Image.open(image_path).convert("RGB")
            
            # Формируем запрос
            query = self.tokenizer.from_list_format([
                {'image': image_path},
                {'text': self.prompt},
            ])
            
            # Генерируем ответ
            inputs = self.tokenizer(query, return_tensors='pt').to(self.model.device)
            output = self.model.generate(**inputs, max_new_tokens=10)
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Парсим ответ
            predicted_class = response.strip().split()[-1]
            
            return {
                "image": os.path.basename(image_path),
                "class": predicted_class,
                "raw_response": response
            }
        except Exception as e:
            print(f"Ошибка обработки {image_path}: {str(e)}")
            return {
                "image": os.path.basename(image_path),
                "class": "ERROR",
                "raw_response": str(e)
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
        "model_name": "Qwen/Qwen-VL-Chat",
        "image_dir": "path/to/your/images",
        "output_csv": "labeled_results.csv"
    }
    
    # Инициализация классификатора
    classifier = InteriorClassifier(config["model_name"])
    
    # Запуск обработки
    classifier.process_directory(
        image_dir=config["image_dir"],
        output_csv=config["output_csv"]
    )


if __name__ == "__main__":
    main()
