import os
from PIL import Image
from tqdm import tqdm
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

class InteriorClassifier:
    """Классификатор состояния интерьера с помощью Qwen-VL"""
    
    def __init__(self, model_name: str = "Qwen/Qwen-VL-Chat"):
        # quant_config = BitsAndBytesConfig(
        #     load_in_4bit=True,                     # Активирует 4-битную квантзацию весов
        #     bnb_4bit_compute_dtype=torch.float16,  # Тип данных для вычислений (bf16/fp16)
        #     bnb_4bit_quant_type="nf4"              # Тип квантзации
        # )
        """Инициализация модели и токенизатора"""
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,                       # Название модели (например, "Qwen/Qwen-VL-Chat")
            device_map="auto",                # Автораспределение слоёв по GPU/CPU
            trust_remote_code=True,           # Разрешает выполнение кастомного кода модели
            # torch_dtype=torch.float16,       # Основной тип данных модели (bf16/fp16/fp32)
            # quantization_config=quant_config  # Конфиг квантзации (None, если не нужно)
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Системный промпт для классификации
        # self.prompt = """
        # Ты эксперт по оценке состояния интерьера. Оцени состояние ремонта на фото. Выбери один класс:
        # - A0: Без отделки (голые стены, бетон)
        # - A1: Требует кап. ремонта (сильный износ)
        # - B0: Под чистовую (white-box)
        # - B1: Требует косметического ремонта
        # - C0: Хорошее состояние
        # - C1: Отличное состояние (с мебелью)
        # - D0: Евроремонт (дизайн-проект)
        # - D1: Luxury (эксклюзив)
        # Ответь ТОЛЬКО меткой класса (например "C0").
        # """

        self.prompt = """
        You are an interior condition assessment expert. Analyze the photo and select ONE class:
        - A0: No finish (bare walls, concrete)
        - A1: Needs major renovation (severe wear)
        - B0: Basic finish (white-box)
        - B1: Needs cosmetic repairs
        - C0: Good condition
        - C1: Excellent condition (with furniture)
        - D0: Premium renovation (design project)
        - D1: Luxury (exclusive)
        Respond ONLY with the class label (e.g. "C0"). Do not add any other text.
        """

        # self.prompt = "Describe image"
    
    def classify_image(self, image_path: str, prompt: str | None = None) -> dict[str, str]:
        """Классификация одного изображения"""
        try:
            if prompt is None:
                prompt = self.prompt
            
            # Формируем запрос
            query = [
                {'image': image_path},
                {'text': prompt}
            ]

            # Подготовка входа
            inputs = self.tokenizer.from_list_format(query)
            inputs = self.tokenizer(inputs, return_tensors='pt').to(self.model.device)

            # Генерация с ограничениями
            outputs = self.model.generate(
                **inputs,
                num_beams=3,          # Оптимальный баланс качества/скорости
                do_sample=False,      # Для детерминированных результатов
                early_stopping=True,  # Остановка, когда все лучи завершены
                max_new_tokens=350,      # Для формата "A0"-"D1"
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            # Декодирование и очистка ответа
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace(prompt, "").strip()
            
            # Извлекаем только класс (первое слово после промпта)
            predicted_class = response.split()[0] if response else "ERROR"
            
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
