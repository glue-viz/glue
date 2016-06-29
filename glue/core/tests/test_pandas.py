from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
from mock import MagicMock
from pandas.util.testing import (assert_series_equal,
                                 assert_frame_equal)

from ..component import (Component, DerivedComponent, CoordinateComponent,
                         CategoricalComponent)
from ..data import Data


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

        d = Data(x=[1, 2, 3])
        series = pd.Series(np.array([0, 1, 2], dtype=np.int))
        comp = d.get_component(d.get_pixel_component_id(0))
        assert_series_equal(series, comp.to_series())

    def test_Data_conversion(self):

        d = Data(n=np.array([4, 5, 6, 7], dtype=np.int64))
        cat_comp = CategoricalComponent(np.array(['a', 'b', 'c', 'd']))
        d.add_component(cat_comp, 'c')
        link = MagicMock()
        link.compute.return_value = np.arange(4)
        deriv_comp = DerivedComponent(d, link)
        d.add_component(deriv_comp, 'd')
        order = [comp.label for comp in d.components]

        frame = pd.DataFrame({
            'n': [4, 5, 6, 7],
            'c': ['a', 'b', 'c', 'd'],
            'd': np.arange(4),
            'Pixel Axis 0 [x]': np.arange(4),
            'World 0': np.arange(4)
        })
        out_frame = d.to_dataframe()
        assert_frame_equal(out_frame, frame)
        assert list(out_frame.columns) == order

    def test_multi_dimensional(self):

        a = np.array([[2, 3], [5, 4], [6, 7]])
        comp = Component(a)
        series = pd.Series(a.ravel())

        assert_series_equal(series, comp.to_series())
