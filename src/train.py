# src/train.py
from pathlib import Path
import json

import torch
from src.config import RANDOM_SEED, SPLIT_CONFIG, MIN_VAL_TEST_PER_CLASS, CLASS_LABELS
from src.schemas.configs import ExperimentConfig
from src.experiment import Experiment


if __name__ == "__main__":
    # 1. =========================== Define hyperparameters ===========================
    EXP_NUMBER = 11
    BATCH_SIZE = 32
    EPOCHS = 1
    START_LR = 3e-5
    IMG_SIZE = 448
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


    # 2. ================================ Define paths =================================
    project_root = Path.cwd()
    data_dir = project_root / "data"
    print(f"data_dir: {data_dir}")
    dataset_dir = data_dir / "interior_dataset"

    exp_dir = project_root / "experiments" / f"exp{EXP_NUMBER:03d}"

    # exp = Experiment(
    #     dataset_dir=dataset_dir,
    #     exp_dir=exp_dir,
    #     class_labels=CLASS_LABELS,
    #     splits=SPLIT_CONFIG,
    #     exp_number=EXP_NUMBER,
    #     batch_size=BATCH_SIZE,
    #     epochs=EPOCHS,
    #     img_size=IMG_SIZE,
    #     start_lr=START_LR,
    #     random_seed=RANDOM_SEED,
    #     device=DEVICE
    # )

    # model = exp.run()

    exp_config_path = exp_dir / "config.json"
    exp_config = Experiment.load_config(exp_config_path)
    exp = Experiment.from_config(exp_config)
