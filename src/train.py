from pathlib import Path
import sys
import json
from src.config import TrainingConfig
from src.trainer import Trainer

if __name__ == "__main__":
    # Пример: инициализация конфига (можно заменить на чтение из файла)
    config = TrainingConfig(
        data_dir=str(Path.cwd().parent.parent / "data" / "interior_dataset"),
        exp_dir=str(Path.cwd().parent.parent / "experiments" / "exp001_baseline" / "results"),
        batch_size=32,
        epochs=10,
        lr=3e-5,
        img_size=380,
        # Можно добавить другие параметры при необходимости
    )

    # Сохраняем конфиг эксперимента в JSON
    config_path = Path(config.exp_dir) / "training_config.json"
    with open(config_path, "w") as f:
        f.write(config.json(indent=4, ensure_ascii=False))
    print(f"Training config saved to {config_path}")

    # Запуск обучения
    trainer = Trainer(config)
    trainer.train()
