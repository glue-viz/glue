class BaseFitter(object):

    def fit(self, x, y, z=None, weights=None):
        raise NotImplementedError()

    def predict(self, x):
        raise NotImplementedError()


class AstropyModelFitter(object):

    def __init__(self, model, fitter):
        self._model = model
        self._fitter = fitter

    def fit(self, x, y, z=None, weights=None, **kwargs):
        result = self._fitter(self._model, x, y, z=z, weights=weights,
                              **kwargs)
        self._model = result
        return result

    def predict(self, *args):
        return self._model(*args)

    @classmethod
    def gaussian_fitter(cls, amplitude=1, mean=0, stddev=1):
        from astropy.modeling import models, fitting
        m = models.Gaussian1D(amplitude=amplitude,
                              mean=mean,
                              stddev=stddev)
        f = fitting.NonLinearLSQFitter()
        return cls(m, f)
