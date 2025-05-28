# label_pipeline.py
import os
from PIL import Image
from tqdm import tqdm
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

# Конфигурация
MODEL_NAME = "Qwen/Qwen-VL-Chat"
IMAGE_DIR = "path/to/your/images"  # Папка с картинками
OUTPUT_CSV = "labeled_results.csv"
PROMPT = """
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

# Загрузка модели (4-битная квантизация для экономии VRAM)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    bnb_4bit_compute_dtype="float16"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Обработка изображений
results = []
for img_file in tqdm(os.listdir(IMAGE_DIR)):
    if not img_file.lower().endswith((".jpg", ".png")):
        continue
    
    image_path = os.path.join(IMAGE_DIR, img_file)
    image = Image.open(image_path).convert("RGB")
    
    # Запрос к модели
    query = tokenizer.from_list_format([
        {'image': image_path},
        {'text': PROMPT},
    ])
    inputs = tokenizer(query, return_tensors='pt').to(model.device)
    output = model.generate(**inputs, max_new_tokens=10)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Извлекаем класс (например, "C0")
    predicted_class = response.strip().split()[-1]  # Простейший парсинг
    
    results.append({
        "image": img_file,
        "class": predicted_class,
        "raw_response": response
    })

# Сохраняем в CSV
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Результаты сохранены в {OUTPUT_CSV}")