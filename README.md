# Interior Image Classification Pipeline

![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)
![HuggingFace](https://img.shields.io/badge/Powered%20by-HuggingFace-yellow.svg)

Проект для автоматической классификации изображений интерьеров с использованием мультимодальной модели Qwen-VL. Поддерживает:

- Фильтрацию "мусорных" изображений (улицы, фасады и т.д.)
- Few-shot классификацию состояния интерьеров (классы A0-D1)
- Пакетную обработку директорий
- Гибкую настройку промптов

## 📦 Установка

1. **Создание виртуального окружения** (рекомендуется):
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/MacOS
.\.venv\Scripts\activate   # Windows
```

2. **Установка зависимостей**:
```bash
pip install -r requirements.txt
```

## 🛠 Требования

- Python 3.12
- Подключение к интернету (для загрузки моделей)
- Для GPU: CUDA 11.8+ и драйверы NVIDIA
- RAM: минимум 16GB (32GB рекомендуется)

## ⚙️ Технические детали

- **Модель**: Qwen-VL (7B параметров)
- **Препроцессинг**:
  - Ресайз до 448x448 с сохранением пропорций
  - Автоматический паддинг белым цветом