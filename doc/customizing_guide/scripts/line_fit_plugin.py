from glue.core.fitters import BaseFitter1D
from glue.config import fit_plugin
import numpy as np


@fit_plugin
class LineFit(BaseFitter1D):
    label = "Line"

    def fit(self, x, y, dy, constraints):
        return np.polyfit(x, y, 1)

    def predict(self, fit_result, x):
        return np.polyval(fit_result, x)
