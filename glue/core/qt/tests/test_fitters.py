from unittest.mock import MagicMock

from glue.core.fitters import SimpleAstropyGaussianFitter, PolynomialFitter

from ..fitters import ConstraintsWidget, FitSettingsWidget


class TestConstraintsWidget(object):

    def setup_method(self, method):
        self.constraints = dict(a=dict(fixed=True, value=1, limits=None))
        self.widget = ConstraintsWidget(self.constraints)

    def test_settings(self):
        assert self.widget.settings('a') == dict(fixed=True, value=1,
                                                 limits=None)

    def test_update_settings(self):
        self.widget._widgets['a'][2].setChecked(False)
        assert self.widget.settings('a')['fixed'] is False

    def test_update_constraints(self):
        self.widget._widgets['a'][2].setChecked(False)
        fitter = MagicMock()
        self.widget.update_constraints(fitter)
        fitter.set_constraint.assert_called_once_with('a',
                                                      fixed=False, value=1,
                                                      limits=None)


class TestFitSettingsWidget(object):

    def test_option(self):
        f = PolynomialFitter()
        f.degree = 1
        w = FitSettingsWidget(f)
        w.widgets['degree'].setValue(5)
        w.update_fitter_from_settings()
        assert f.degree == 5

    def test_set_constraints(self):
        f = SimpleAstropyGaussianFitter()
        w = FitSettingsWidget(f)
        w.constraints._widgets['amplitude'][2].setChecked(True)
        w.update_fitter_from_settings()
        assert f.constraints['amplitude']['fixed']
