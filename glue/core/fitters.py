"""
Glue's fitting classes are designed to be easily subclassed for performing
custom model fitting in Glue.

See the guide on :ref:`writing custom fit plugins <fit_plugins>` for
help with using custom fitting utilities in Glue.
"""

from __future__ import absolute_import, division, print_function

import numpy as np

from glue.core.simpleforms import IntOption, Option

__all__ = ['BaseFitter1D',
           'PolynomialFitter',
           'AstropyFitter1D',
           'SimpleAstropyGaussianFitter',
           'BasicGaussianFitter']


class BaseFitter1D(object):

    """
    Base class for 1D fitters.

    This abstract class must be overwritten.
    """

    label = "Fitter"
    """A short label for the fit, used by the GUI"""

    param_names = []
    """list of parameter names that support restrictions"""

    def __init__(self, **params):
        self._constraints = {}

        for k, v in params.items():
            if k in self.param_names:
                self.set_constraint(k, value=v)
            else:
                setattr(self, k, v)

    def plot(self, fit_result, axes, x):
        """
        Plot the result of a fit.

        :param fit_result: The output from fit
        :param axes: The Matplotlib axes to add the fit to
        :param x: The values of X at which to visualize the model

        :returns: A list of matplotlib artists. **This is important:**
                  plots will not be properly cleared if this isn't provided
        """
        y = self.predict(fit_result, x)
        result = axes.plot(x, y, '#4daf4a',
                           lw=3, alpha=0.8,
                           scalex=False, scaley=False)
        return result

    def _sigma_to_weights(self, dy):
        if dy is not None:
            return 1. / np.asarray(dy) ** 2

    @property
    def options(self):
        """
        A dictionary of the current setting of each model hyperparameter.

        Hyperparameters are defined in subclasses by creating class-level
        :mod:`Option <glue.core.simpleforms>` attributes. This attribute
        dict maps ``{hyperparameter_name: current_value}``
        """
        result = []
        for typ in type(self).mro():
            result.extend(k for k, v in typ.__dict__.items()
                          if isinstance(v, Option))
        return dict((o, getattr(self, o)) for o in result)

    def summarize(self, fit_result, x, y, dy=None):
        """
        Return a textual summary of the fit.

        :param fit_result: The return value from :meth:`fit`
        :param x: The x values passed to :meth:`fit`
        :returns: A description of the fit result
        :rtype: str
        """
        return str(fit_result)

    @property
    def constraints(self):
        """
        A dict of the constraints on each parameter in :attr:`param_names`.
        Each value is itself a dict with 3 items:

        :key value: The default value
        :key fixed: True / False, indicating whether the parameter is fixed
        :key bounds: [min, max] or None, indicating lower/upper limits
        """
        result = {}
        for p in self.param_names:
            result[p] = dict(value=None, fixed=False, limits=None)
            result[p].update(self._constraints.get(p, {}))
        return result

    def set_constraint(self, parameter_name, value=None,
                       fixed=None, limits=None):
        """
        Update a constraint.

        :param parameter_name: name of the parameter to update
        :type parameter_name: str
        :param value: Set the default value (optional)
        :param limits: Set the limits to[min, max] (optional)
        :param fixed: Set whether the parameter is fixed (optional)
        """
        c = self._constraints.setdefault(parameter_name, {})
        if value is not None:
            c['value'] = value
        if fixed is not None:
            c['fixed'] = fixed
        if limits is not None:
            c['limits'] = limits

    def build_and_fit(self, x, y, dy=None):
        """
        Method which builds the arguments to fit, and calls that method
        """
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        if dy is not None:
            dy = np.asarray(dy).ravel()

        return self.fit(x, y, dy=dy,
                        constraints=self.constraints,
                        **self.options)

    def fit(self, x, y, dy, constraints, **options):
        """
        Fit the model to data.

        *This must be overriden by a subclass.*

        :param x: The x values of the data
        :type x:  :class:`numpy.ndarray`
        :param y: The y values of the data
        :type y:  :class:`numpy.ndarray`
        :param dy: 1 sigma uncertainties on each datum (optional)
        :type dy: :class:`numpy.ndarray`
        :param constraints: The current value of :attr:`constraints`
        :param options: kwargs for model hyperparameters.

        :returns: An object representing the fit result.
        """

        raise NotImplementedError()

    def predict(self, fit_result, x):
        """
        Evaulate the model at a set of locations.

        **This must be overridden in a subclass.**

        :param fit_result: The result from the fit method
        :param x: Locations to evaluate model at
        :type x: :class:`numpy.ndarray`

        :returns: model(x)
        :rtype: :class:`numpy.ndarray`
        """
        raise NotImplementedError()


class AstropyFitter1D(BaseFitter1D):

    """
    A base class for wrapping :mod:`astropy.modeling`.

    Subclasses must override :attr:`model_cls` :attr:`fitting_cls`
    to point to the desired Astropy :mod:`model <astropy.modeling>`
    and :mod:`fitter <astropy.modeling.fitting>` classes.

    In addition, they should override :attr:`label` with a better label,
    and :meth:`parameter_guesses` to generate initial guesses
    """

    model_cls = None
    """class describing the model"""

    fitting_cls = None
    """class to fit the model"""

    label = "Base Astropy Fitter"
    """UI Label"""

    @property
    def param_names(self):
        return self.model_cls.param_names

    def predict(self, fit_result, x):
        model, _ = fit_result
        return model(x)

    def summarize(self, fit_result, x, y, dy=None):
        model, fitter = fit_result
        result = [_report_fitter(fitter), ""]
        pnames = list(sorted(model.param_names))
        maxlen = max(map(len, pnames))
        result.extend("%s = %e" % (p.ljust(maxlen), getattr(model, p).value)
                      for p in pnames)
        return "\n".join(result)

    def fit(self, x, y, dy, constraints):
        m, f = self._get_model_fitter(x, y, dy, constraints)

        dy = self._sigma_to_weights(dy)
        return f(m, x, y, weights=dy), f

    def _get_model_fitter(self, x, y, dy, constraints):
        if self.model_cls is None or self.fitting_cls is None:
            raise NotImplementedError("Model or fitting class is unspecified.")

        params = dict((k, v['value']) for k, v in constraints.items())

        # update unset parameters with guesses from data
        for k, v in self.parameter_guesses(x, y, dy).items():
            if params[k] is not None or constraints[k]['fixed']:
                continue
            params[k] = v

        m = self.model_cls(**params)
        f = self.fitting_cls()

        for param_name, constraint in constraints.items():
            param = getattr(m, param_name)
            if constraint['fixed']:
                param.fixed = True
            if constraint['limits']:
                param.min, param.max = constraint['limits']
        return m, f

    def parameter_guesses(self, x, y, dy):
        """
        Provide initial guesses for each model parameter.

        **The base implementation does nothing, and should be overridden**

        :param x: X - values of the data
        :type x: :class:`numpy.ndarray`
        :param y: Y - values of the data
        :type y: :class:`numpy.ndarray`
        :param dy: ncertainties on Y(assumed to be 1 sigma)
        :type dy: :class:`numpy.ndarray`

        :returns: A dict maping ``{parameter_name: value guess}`` for each
                  parameter
        """
        return {}


def _gaussian_parameter_estimates(x, y, dy):

    amplitude = np.percentile(y, 95)
    y = np.maximum(y / y.sum(), 0)
    mean = (x * y).sum()
    stddev = np.sqrt((y * (x - mean) ** 2).sum())
    return dict(mean=mean, stddev=stddev, amplitude=amplitude)


class BasicGaussianFitter(BaseFitter1D):

    """
    Fallback Gaussian fitter, for astropy < 0.3.

    If :mod:`astropy.modeling` is installed, this class is replaced by
    :class:`SimpleAstropyGaussianFitter`
    """
    label = "Gaussian"

    def _errorfunc(self, params, x, y, dy):
        yp = self.eval(x, *params)
        result = (yp - y)
        if dy is not None:
            result /= dy
        return result

    @staticmethod
    def eval(x, amplitude, mean, stddev):
        return np.exp(-(x - mean) ** 2 / (2 * stddev ** 2)) * amplitude

    @staticmethod
    def fit_deriv(x, amplitude, mean, stddev):
        """
        Gaussian1D model function derivatives.
        """

        d_amplitude = np.exp(-0.5 / stddev ** 2 * (x - mean) ** 2)
        d_mean = amplitude * d_amplitude * (x - mean) / stddev ** 2
        d_stddev = amplitude * d_amplitude * (x - mean) ** 2 / stddev ** 3
        return [d_amplitude, d_mean, d_stddev]

    def fit(self, x, y, dy, constraints):
        from scipy import optimize
        init_values = _gaussian_parameter_estimates(x, y, dy)
        init_values = [init_values[p] for p in ['amplitude', 'mean', 'stddev']]
        farg = (x, y, dy)
        dfunc = None
        fitparams, status, dinfo, mess, ierr = optimize.leastsq(
            self._errorfunc, init_values, args=farg, Dfun=dfunc,
            full_output=True)
        return fitparams

    def predict(self, fit_result, x):
        return self.eval(x, *fit_result)

    def summarize(self, fit_result, x, y, dy=None):
        return ("amplitude = %e\n"
                "mean      = %e\n"
                "stddev    = %e" % tuple(fit_result))


GaussianFitter = BasicGaussianFitter


try:
    from astropy.modeling import models, fitting

    class SimpleAstropyGaussianFitter(AstropyFitter1D):

        """
        Guassian fitter using astropy.modeling.
        """
        model_cls = models.Gaussian1D
        try:
            fitting_cls = fitting.LevMarLSQFitter
        except AttributeError:  # astropy v0.3
            fitting_cls = fitting.NonLinearLSQFitter

        label = "Gaussian"

        parameter_guesses = staticmethod(_gaussian_parameter_estimates)

    GaussianFitter = SimpleAstropyGaussianFitter

except ImportError:
    pass


class PolynomialFitter(BaseFitter1D):

    """
    A polynomial model.

    The degree of the polynomial is specified by :attr:`degree`
    """
    label = "Polynomial"
    degree = IntOption(min=0, max=5, default=3, label="Polynomial Degree")

    def fit(self, x, y, dy, constraints, degree=2):
        """
        Fit a ``degree``-th order polynomial to the data.
        """
        w = self._sigma_to_weights(dy)

        return np.polyfit(x, y, degree, w=w)

    def predict(self, fit_result, x):
        return np.polyval(fit_result, x)

    def summarize(self, fit_result, x, y, dy=None):
        return "Coefficients:\n" + "\n".join("%e" % coeff
                                             for coeff in fit_result.tolist())


def _report_fitter(fitter):
    if "nfev" in fitter.fit_info:
        return "Converged in %i iterations" % fitter.fit_info['nfev']
    return 'Converged'

__FITTERS__ = [PolynomialFitter, GaussianFitter]
