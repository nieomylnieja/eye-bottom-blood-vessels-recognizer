from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    DatasetBasePath: Path = Path().absolute().joinpath("dataset")
