from typing import Union

from skimage import io

from classifiers import Dataset, ImageType
from config import Config


class RetinalImage:

    def __init__(self, dataset: Dataset, name: Union[int, str]):
        self.index: int = name if type(name) is int else self.index_from_name(name)
        self.dataset: Dataset = dataset

    @staticmethod
    def index_from_name(name: str) -> int:
        return int(name.split(".")[0])

    def _f_name(self, typ: ImageType) -> str:
        return f"{'0' if self.index < 10 else ''}{self.index}.{typ.f_ext}"

    def open(self, typ: ImageType = ImageType.INPUT, as_gray: bool = False):
        path = Config.DatasetBasePath.joinpath(self.dataset).joinpath(typ).joinpath(self._f_name(typ))
        return io.imread(path.__str__(), as_gray=as_gray)
