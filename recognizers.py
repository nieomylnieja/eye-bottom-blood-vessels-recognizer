from __future__ import annotations

from abc import abstractmethod, ABCMeta
from collections import namedtuple

import numpy as np
import pandas as pd
from skimage.filters import frangi
from skimage.color import rgb2gray, rgb2hsv, hsv2rgb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from classifiers import ImageType
from retinal_image import RetinalImage


class Recognizer(metaclass=ABCMeta):

    def __init__(self, img: RetinalImage):
        self.img: RetinalImage = img

    @abstractmethod
    def run(self, *args):
        raise NotImplementedError


class FilterRecognizer(Recognizer):
    Result = namedtuple("Result", ["plt", "statistics"])

    def run(self, hsv_v_threshold: float, visibility_correction: int) -> FilterRecognizer.Result:
        image = self.f_img(self.img)

        fig, ax = plt.subplots(ncols=3)
        fig.set_facecolor("#2E3440")

        ax[0].imshow(image.get(), cmap="gray")
        ax[0].set_title("Original image", color="white")

        limit_to_one = 1
        image \
            .extract_green_channel() \
            .rgb2hsv() \
            .extract_value_channel() \
            .hsv2rgb() \
            .rgb2gray() \
            .apply_threshold(hsv_v_threshold) \
            .frangi() \
            .trim_to_fov() \
            .enhance_white_visibility(visibility_correction) \
            .apply_threshold(limit_to_one)

        result = image.get()
        ax[1].imshow(result, cmap="gray")
        ax[1].set_title("Frangi filter result", color="white")

        manual = self.img.open(ImageType.MANUAL, as_gray=True)
        ax[2].imshow(manual, cmap="gray")
        ax[2].set_title("Manual gold standard", color="white")

        for a in ax:
            a.axis("off")

        plt.tight_layout()
        return self.Result(plt, self.Statistics(manual, result).get_report())

    class Statistics:
        _tolerances = [1e-1, 1e-2, 1e-3, 0]
        _columns = ["accuracy", "sensitivity", "specificity"]

        def __init__(self, base: np.ndarray, pred: np.ndarray):
            self.b: np.ndarray = (base.copy() == 255).flatten()
            self.p: np.ndarray = pred.copy().flatten()

        def get_report(self) -> pd.DataFrame:
            cms = [self._calc_cm(t) for t in self._tolerances]
            return pd.DataFrame(
                [[self._accuracy(cm), self._sensitivity(cm), self._specificity(cm)] for cm in cms],
                columns=self._columns,
                index=[str(t) for t in self._tolerances])

        def _calc_cm(self, t: float) -> np.ndarray:
            return confusion_matrix(self.b, self.p.copy() > t)

        @staticmethod
        def _accuracy(cm: np.ndarray) -> list[float]:
            return (cm[0, 0] + cm[1, 1]) / cm.sum()

        @staticmethod
        def _sensitivity(cm: np.ndarray) -> list[float]:
            return cm[0, 0] / (cm[0, 0] + cm[0, 1])

        @staticmethod
        def _specificity(cm: np.ndarray) -> list[float]:
            return cm[1, 1] / (cm[1, 0] + cm[1, 1])

    class f_img:

        def __init__(self, img: RetinalImage):
            self._img: np.ndarray = img.open()
            self._fov: np.ndarray = img.open(ImageType.FOV, as_gray=True)

        def extract_green_channel(self) -> f_img:
            self._img[:, :, 0] = 0
            self._img[:, :, 2] = 0
            return self

        def extract_value_channel(self) -> f_img:
            self._img[:, :, 0] = 0
            self._img[:, :, 1] = 0
            return self

        def rgb2hsv(self) -> f_img:
            self._img = rgb2hsv(self._img)
            return self

        def hsv2rgb(self) -> f_img:
            self._img = hsv2rgb(self._img)
            return self

        def rgb2gray(self) -> f_img:
            self._img = rgb2gray(self._img)
            return self

        def apply_threshold(self, th: float) -> f_img:
            np.putmask(self._img, self._img > th, th)
            return self

        def frangi(self) -> f_img:
            self._img = frangi(self._img)
            return self

        def trim_to_fov(self) -> f_img:
            self._img = self._img * self._fov
            return self

        def enhance_white_visibility(self, v_correction: int) -> f_img:
            self._img = self._img * v_correction
            return self

        def get(self) -> np.ndarray:
            return self._img.copy()
