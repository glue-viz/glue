"""
Glue's fitting classes are designed to be easily subclassed for using
custom model fitting in Glue.

A subclass of :class:`BaseFitter1D` must, at a minimum, override the
fit and predict methods. In addition, a subclass can optionally add:

 * A label for the fit. This is highly recommended, and is used
   when selecting between fitters.

 * A class-level list of param_names. These list the parameters
   fit by a model that can be constrained (either by setting to a fixed
   value or reticting to a given range). When such a list is present,
   Glue provides a UI for setting these constraints)

 * An overridden plot and/or summarize method. These are used to
   visualize the result of a model fit. Basic implementations already
   exist for both methods

 * Zero or more class-level Option descriptors, to let users interactively
   set the hyperparameters of a model
"""

import numpy as np

from .simpleforms import IntOption, Option


class BaseFitter1D(object):
    label = "Fitter"  # short name used to identify this Fit class in the UI
    param_names = []  # list of parameter names that support restrictions

    def __init__(self, **params):
        self._constraints = {}

        for k, v in params.items():
            if k in self.param_names:
                self.set_constraint(k, value=v)
            else:
                setattr(self, k, v)

    def plot(self, fit_result, axes, x):
        """
        Plot the result of a fit

        :param fit_result: The output from fit
        :param axes: The Matplotlib axes to add the fit to
        :param x: The values of X at which to visualize the model

        :returns: A list of matplotlib artists. This is important,
        and fit plots will not be properly cleared if this isn't provided
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
        result = []
        for typ in type(self).mro():
            result.extend(k for k, v in typ.__dict__.items()
                          if isinstance(v, Option))
        return dict((o, getattr(self, o)) for o in result)

    def summarize(self, fit_result, x, y, dy=None):
        return str(fit_result)

    @property
    def constraints(self):
        result = {}
        for p in self.param_names:
            result[p] = dict(value=None, fixed=False, limits=None)
            result[p].update(self._constraints.get(p, {}))
        return result

    def set_constraint(self, parameter_name, value=None,
                       fixed=None, limits=None):

        c = self._constraints.setdefault(parameter_name, {})
        if value is not None:
            c['value'] = value
        if fixed is not None:
            c['fixed'] = fixed
        if limits is not None:
            c['limits'] = limits

    def build_and_fit(self, x, y, dy=None):
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

        This must be ovverriden by a subclass

        :param x: The X-values of the data
        :type x: numpy array
        :param y: The Y-values of the data
        :param y: numpy array
        :param dy: The uncertainties (assumed to be 1sigma) on each datum
        :type dy: numpy array, or None
        :param constraints: A dictionary of constraints on model
        parameters. Each value is itself a dictionary, with 4 keys:
            - value: The requested initial value for this parameter
            - fixed: if True, this parameter should be held fixed
            - limits: None, or [min, max], giving the allowed range
                      for this parameter
        **options: A dict of {option name: option value} for each
                   hyperparameter of this model

        :returns: An object representing the fit. This is
        passed to predict and summarize so, as long as these methods
        know what to do with the result, there are no other restrictions
        on what the return value should be.
        """
        raise NotImplementedError()

    def predict(self, fit_result, x):
        """
        Evaulate the model at a set of locations.

        This must be overridden in a subclass.


        :param fit_result: The result from the fit method
        :param x: Locations to evaluate model at
        :type x: Numpy array

        :returns: model(x)
        :rtype: numpy array
        """
        raise NotImplementedError()


class AstropyFitter1D(BaseFitter1D):

    """
    A generic class for wrapping astropy model classes in
    Glue fitters

    Subclasses must override:

    - The model_cls class variable with an Astropy.modeling.model class
    - The fitting_cls class variable with an Astropy.modeling.fitters class

    In addition, they should override:
    - The label with a better label
    - The _parameter_guesses method to generate initial guesses for
      model parameters
    """

    model_cls = None  # class describing the model
    fitting_cls = None  # class to fit the model
    label = "Base Astropy Fitter"

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
        for k, v in self._parameter_guesses(x, y, dy).items():
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

    def _parameter_guesses(self, x, y, dy):
        """
        Provide initial guesses for each model parameter

        The base implementation does nothing, and should be overridden

        :param x: X-values of the data
        :type x: numpy array
        :param y: Y-values of the data
        :type y: numpy array
        :param dy: Uncertainties on Y (assumed to be 1 sigma)
        :type dy: numpy array

        :returns: A dict maping {parameter_name: value guess} for each
                  parameter
        """
        pass


def _gaussian_parameter_estimates(x, y, dy):

    y = np.maximum(y / y.sum(), 0)
    mean = (x * y).sum()
    stddev = np.sqrt((y * (x - mean) ** 2).sum())
    amplitude = np.percentile(y, 95)
    return dict(mean=mean, stddev=stddev, amplitude=amplitude)


class BasicGaussianFitter(BaseFitter1D):

    """
    Fallback Gaussian fitter, for astropy < 0.3

    If astropy.modeling exists, this class is replaced by
    SimpleAstropyGaussianFitter
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
        model_cls = models.Gaussian1D
        fitting_cls = fitting.NonLinearLSQFitter
        label = "Gaussian"

        _parameter_guesses = staticmethod(_gaussian_parameter_estimates)

    GaussianFitter = SimpleAstropyGaussianFitter


except ImportError:
    pass


class PolynomialFitter(BaseFitter1D):
    label = "Polynomial"
    degree = IntOption(min=0, max=5, default=3, label="Polynomial Degree")

    def fit(self, x, y, dy, constraints, degree=2):
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
