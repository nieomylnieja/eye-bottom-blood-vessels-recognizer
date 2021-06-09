from enum import Enum, auto
from os import PathLike


class Classifier(Enum):
    __metaclass__ = PathLike

    def _generate_next_value_(name, *_):
        return name.lower().replace("_", " ").capitalize()

    @classmethod
    def list(cls) -> list[str]:
        return list(map(lambda c: c.value, cls))

    @property
    def name(self) -> str:
        return self._name_.lower()

    def __fspath__(self) -> str:
        return self.name


class ImageType(Classifier):
    INPUT = auto()
    FOV = auto()
    MANUAL = auto()

    @property
    def f_ext(self) -> str:
        return "jpg" if self == ImageType.INPUT else "tif"


class Dataset(Classifier):
    HEALTHY = auto()
    GLAUCOMATOUS = auto()
    DIABETIC_RETINOPATHY = auto()
