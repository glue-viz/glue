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

        pass
        #Chris, you'll have to put some basic testing logic here.

    def test_Data_conversion(self):

        d = Data(n=[4, 5, 6, 7])
        cat_comp = CategoricalComponent(np.array(['a', 'b', 'c', 'd']))
        d.add_component(cat_comp, 'c')
        link = MagicMock()
        link.compute.return_value = np.arange(4)
        deriv_comp = DerivedComponent(d, link)
        d.add_component(deriv_comp, 'd')

        frame = pd.DataFrame({
                                'n': [4, 5, 6, 7],
                                'c': ['a', 'b', 'c', 'd'],
                                'd': np.arange(4),
                                'Pixel Axis 0': np.arange(4),
                                'World 0': np.arange(4)
                              })

        assert_frame_equal(d.to_dataframe(), frame)


