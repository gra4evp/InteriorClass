MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

RANDOM_SEED = 42  # Фиксируем seed для воспроизводимости

CLASS_LABELS = ["A0", "A1", "B0", "B1", "C0", "C1", "D0", "D1"]

# Конфигурация разделения для каждого класса
SPLIT_CONFIG = {
    # Классы с малым количеством данных (<1000)
    "A1": {"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15},
    "C0": {"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15},
     
    # Классы со средним количеством данных (1000-5000)
    "B0": {"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15},
    "C1": {"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1},
    "D1": {"train_ratio": 0.9, "val_ratio": 0.05, "test_ratio": 0.05},
    
    # Классы с большим количеством данных (>5000)
    "A0": {"train_ratio": 0.94, "val_ratio": 0.03, "test_ratio": 0.03},
    "B1": {"train_ratio": 0.94, "val_ratio": 0.03, "test_ratio": 0.03},
    "D0": {"train_ratio": 0.94, "val_ratio": 0.03, "test_ratio": 0.03},
}

# Минимальное количество примеров для val и test
MIN_VAL_TEST_PER_CLASS = 20  # Гарантируем хотя бы по 20 на val и test
