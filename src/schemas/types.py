# src/schemas/types.py
from pathlib import Path
from typing import NamedTuple, TypeAlias, Annotated
from dataclasses import dataclass
from pydantic import Field


# @dataclass(frozen=True)
class SampleItem(NamedTuple):
    filepath: Path
    label: str
    class_idx: int


DatasetSplit: TypeAlias = tuple[list[SampleItem], list[SampleItem], list[SampleItem]]

# Custom type for ratio values between 0 and 1 (inclusive)
RatioFloat: TypeAlias = Annotated[float, Field(gt=0, le=1)]
