from glue.core.fitters import BaseFitter1D
from glue.core.simpleforms import IntOption
from glue.config import fit_plugin
import numpy as np


@fit_plugin
class PolynomialFitter(BaseFitter1D):
    label = "Polynomial"
    degree = IntOption(min=0, max=5, default=3, label="Polynomial Degree")

    def fit(self, x, y, dy, constraints, degree=2):
        return np.polyfit(x, y, degree)

    def predict(self, fit_result, x):
        return np.polyval(fit_result, x)

    def summarize(self, fit_result, x, y, dy=None):
        return "Coefficients:\n" + "\n".join("%e" % coeff
                                             for coeff in fit_result.tolist())
