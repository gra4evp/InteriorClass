# src/schemas/training.py

from pydantic import BaseModel
from typing import Dict, Any
from enum import Enum

class EventType(str, Enum):
    epoch_end = "epoch_end"
    validation = "validation"
    test = "test"


class Metric(BaseModel):
    name: str
    value: float
    extra: Dict[str, Any] | None = None


class TrainerEvent(BaseModel):
    type: EventType
    epoch: int | None = None
    loss_value: float = None
    metrics: dict[str, Metric] | None = None
    artifacts: dict[str, Any] | None = None


class TrainEpochEvent(TrainerEvent):
    pass


class ValidationEvent(TrainerEvent):
    pass


class TestEvent(TrainerEvent):
    pass
