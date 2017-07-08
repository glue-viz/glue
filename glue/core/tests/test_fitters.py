

from __future__ import absolute_import, division, print_function

import pytest
import numpy as np
from mock import MagicMock

from glue.tests.helpers import requires_scipy, requires_astropy, ASTROPY_INSTALLED

from ..fitters import (PolynomialFitter, IntOption,
                       BasicGaussianFitter)
needs_modeling = pytest.mark.skipif("False", reason='')


if ASTROPY_INSTALLED:
    from astropy.modeling.models import Gaussian1D
    try:
        from astropy.modeling.fitting import NonLinearLSQFitter
    except ImportError:
        from astropy.modeling.fitting import LevMarLSQFitter as NonLinearLSQFitter
    from ..fitters import SimpleAstropyGaussianFitter


@requires_astropy
@requires_scipy
class TestAstropyFitter(object):

    def test_fit(self):

        f = SimpleAstropyGaussianFitter(amplitude=1, mean=2, stddev=3)
        f.fitting_cls = fitter = MagicMock()

        f.build_and_fit([1, 2, 3], [2, 3, 4])
        (model, x, y), kwargs = fitter().call_args

        assert model.amplitude == 1
        assert model.mean == 2
        assert model.stddev == 3

        np.testing.assert_array_equal(x, [1, 2, 3])
        np.testing.assert_array_equal(y, [2, 3, 4])

    def test_fit_converts_errors_to_weights(self):
        f = SimpleAstropyGaussianFitter(amplitude=1, mean=2, stddev=3)
        f.fitting_cls = fitter = MagicMock()
        f.build_and_fit([1, 2, 3], [2, 3, 4], [3, 4, 5])

        args, kwargs = fitter().call_args
        np.testing.assert_array_equal(kwargs['weights'],
                                      1. / np.array([3., 4., 5]) ** 2)

    def test_fit_returns_model_and_fitter(self):
        f = SimpleAstropyGaussianFitter(amplitude=1, mean=2, stddev=3)
        model, fitter = f.build_and_fit([1, 2, 3], [2, 3, 4])
        assert isinstance(model, Gaussian1D)
        assert isinstance(fitter, NonLinearLSQFitter)

    def test_predict_uses_model(self):
        f = SimpleAstropyGaussianFitter(amplitude=1, mean=2, stddev=3)
        result = f.build_and_fit([1, 2, 3], [2, 3, 4])
        model, _ = result
        np.testing.assert_array_equal(model([3]),
                                      f.predict(result, [3]))

    def test_summarize(self):
        f = SimpleAstropyGaussianFitter(amplitude=1, mean=2, stddev=3)
        model, fitter = f.build_and_fit([1, 2, 3], [2, 3, 4])

        expected = ["Converged in %i iterations" % fitter.fit_info['nfev'],
                    "",
                    "amplitude = %e" % model.amplitude.value,
                    "mean      = %e" % model.mean.value,
                    "stddev    = %e" % model.stddev.value
                    ]
        expected = '\n'.join(expected)
        actual = f.summarize((model, fitter), [1, 2, 3], [2, 3, 4])
        assert expected == actual

    def test_range_constraints(self):
        f = SimpleAstropyGaussianFitter(amplitude=1, mean=0, stddev=1)
        x = np.linspace(0, 10, 10)
        y = np.exp(-x ** 2 / 2)
        f.set_constraint('mean', limits=[1, 2])

        model, fitter = f.build_and_fit(x, y)
        np.testing.assert_almost_equal(model.mean.value, 1)

    def test_fixed_constraints(self):
        f = SimpleAstropyGaussianFitter(amplitude=1.5, mean=0, stddev=1)
        x = np.linspace(-5, 5, 10)
        y = np.exp(-x ** 2 / 2)
        f.set_constraint('amplitude', fixed=True)

        model, fitter = f.build_and_fit(x, y)
        np.testing.assert_almost_equal(model.amplitude.value, 1.5)


class TestPolynomialFitter(object):

    def test_fit(self):
        f = PolynomialFitter()
        result = f.build_and_fit([1, 2, 3, 4, 5, 6], [2, 3, 4, 4, 3, 4])
        expected = np.polyfit([1, 2, 3, 4, 5, 6], [2, 3, 4, 4, 3, 4], 3)
        np.testing.assert_array_equal(result, expected)

    def test_predict(self):
        f = PolynomialFitter(degree=1)
        fit = f.build_and_fit([1, 2, 3], [2, 3, 4])
        result = f.predict(fit, [4])
        expected = [5]
        np.testing.assert_almost_equal(result, expected)

    def test_summarize(self):
        f = PolynomialFitter(degree=1)
        fit = f.build_and_fit([1, 2, 3], [2, 3, 4])

        assert f.summarize(
            fit, [1, 2, 3], [2, 3, 4]) == "Coefficients:\n%e\n%e" % (1, 1)


class TestOptions(object):

    def test_options(self):
        p = PolynomialFitter(degree=3)
        assert p.options == {'degree': 3}

    def test_inherited(self):
        class Inherit(PolynomialFitter):
            test = IntOption(default=5)
        assert Inherit().options == dict(degree=3, test=5)

    def test_options_passed_to_fit(self):

        p = PolynomialFitter(degree=4)
        p.fit = MagicMock()

        p.build_and_fit([1, 2, 3], [2, 3, 4])
        assert p.fit.call_args[1]['degree'] == 4


@requires_astropy
class TestFitWrapper(object):

    def setup_method(self, method):
        f = SimpleAstropyGaussianFitter(amplitude=1, mean=2, stddev=3)
        f.fit = MagicMock()
        self.x = [1, 2, 3]
        self.y = [2, 3, 4]
        self.f = f

    def call_info(self):
        return self.f.fit.call_args

    def test_basic(self):
        self.f.build_and_fit(self.x, self.y)
        (x, y), kwargs = self.call_info()
        assert kwargs['constraints'] == {'amplitude': dict(value=1,
                                                           fixed=False,
                                                           limits=None),
                                         'mean': dict(value=2,
                                                      fixed=False,
                                                      limits=None),
                                         'stddev': dict(value=3,
                                                        fixed=False,
                                                        limits=None)}
        np.testing.assert_array_equal(x, self.x)
        np.testing.assert_array_equal(y, self.y)


@requires_astropy
class TestSetConstraints(object):

    def test(self):
        f = SimpleAstropyGaussianFitter(amplitude=1, mean=2, stddev=3)
        f.set_constraint('amplitude', value=3, fixed=True)
        f.set_constraint('mean', limits=[1, 2])
        assert f.constraints == {
            'amplitude': dict(value=3, fixed=True, limits=None),
            'mean': dict(value=2, fixed=False, limits=[1, 2]),
            'stddev': dict(value=3, fixed=False, limits=None)
        }


@requires_scipy
class TestBasicGaussianFitter(object):

    def test(self):

        f = BasicGaussianFitter()

        x = np.linspace(-10, 10)
        y = np.exp(-x ** 2)
        r = f.build_and_fit(x, y)

        expected = [3.67879441e-01, 1.83156389e-02, 1.23409804e-04]
        np.testing.assert_array_almost_equal(f.predict(r, [1, 2, 3]),
                                             expected)
