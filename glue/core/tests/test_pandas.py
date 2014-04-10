import operator

from mock import MagicMock
import pytest
import numpy as np
import pandas as pd
from pandas.util.testing import (assert_series_equal,
                                    assert_frame_equal)

from ..data import (Component, ComponentID, Data,
                    DerivedComponent, CoordinateComponent,
                    CategoricalComponent)
from ... import core


class TestPandasConversion(object):

    def test_Component_conversion(self):

        comp = Component(np.arange(5))
        series = pd.Series(np.arange(5))

        assert_series_equal(series, comp.to_series())

    def test_DerivedComponent_conversion(self):

        data = MagicMock()
        link = MagicMock()
        link.compute.return_value = np.arange(5)
        comp = DerivedComponent(data, link)

        series = pd.Series(np.arange(5))
        assert_series_equal(series, comp.to_series())

    def test_CategoricalComponent_conversion(self):

        comp = CategoricalComponent(np.array(['a', 'b', 'c', 'd']))
        series = pd.Series(['a', 'b', 'c', 'd'])

        assert_series_equal(series, comp.to_series())

    def test_CoordinateComponent_conversion(self):

        comp = CoordinateComponent(core.Data(), 2)
        with pytest.raises(NotImplementedError):
            comp.to_series()

