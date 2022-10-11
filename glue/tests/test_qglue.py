import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

from .. import qglue
from ..core import BaseCartesianData, Data
from ..core.exceptions import IncompatibleAttribute
from ..core.registry import Registry
from .helpers import requires_astropy, requires_qt


@requires_qt
@requires_astropy
class TestQGlue(object):

    def setup_method(self, method):

        from astropy.table import Table
        from astropy.io.fits import HDUList, ImageHDU

        Registry().clear()

        x = [1, 2, 3]
        y = [2, 3, 4]

        u = [10, 20, 30, 40]
        v = [20, 40, 60, 80]

        self.xy = {'x': x, 'y': y}
        self.dict_data = {'u': u, 'v': v}
        self.recarray_data = np.rec.array([(0, 1), (2, 3)],
                                          dtype=[(str('a'), int), (str('b'), int)])
        self.astropy_table = Table({'x': x, 'y': y})
        self.bad_data = {'x': x, 'u': u}
        self.hdulist = HDUList([ImageHDU(x, name='PRIMARY')])

        self.x = np.array(x)
        self.y = np.array(y)
        self.u = np.array(u)
        self.v = np.array(v)

    def check_setup(self, dc, expected):
        # assert that the assembled data collection returned
        # form qglue matches expected structure

        # test for expected data, components
        for data in dc:
            components = set(c.label for c in data.components)
            e = expected.pop(data.label)
            for component in e:
                assert component in components
        assert len(expected) == 0

    def test_qglue_starts_application(self):
        pandas_data = pd.DataFrame(self.xy)
        with patch('glue.app.qt.GlueApplication') as ga:
            qglue(data1=pandas_data)
            ga.assert_called_once()

    def test_single_pandas(self):
        with patch('glue.app.qt.GlueApplication') as ga:
            qglue(data1=self.xy)
            dc = ga.call_args[0][0]
        self.check_setup(dc, {'data1': ['x', 'y']})

    def test_single_pandas_nonstring_column(self):
        with patch('glue.app.qt.GlueApplication') as ga:
            qglue(data1=pd.DataFrame({1: [1, 2, 3]}))
            dc = ga.call_args[0][0]
        self.check_setup(dc, {'data1': ['1']})

    def test_single_numpy(self):
        with patch('glue.app.qt.GlueApplication') as ga:
            qglue(data1=np.array([1, 2, 3]))
            dc = ga.call_args[0][0]
        self.check_setup(dc, {'data1': ['data1']})

    def test_single_list(self):
        with patch('glue.app.qt.GlueApplication') as ga:
            qglue(data1=[1, 2, 3])
            dc = ga.call_args[0][0]
        self.check_setup(dc, {'data1': ['data1']})

    def test_single_dict(self):
        with patch('glue.app.qt.GlueApplication') as ga:
            qglue(data2=self.dict_data)
            dc = ga.call_args[0][0]
        self.check_setup(dc, {'data2': ['u', 'v']})

    def test_recarray(self):
        with patch('glue.app.qt.GlueApplication') as ga:
            qglue(data3=self.recarray_data)
            dc = ga.call_args[0][0]
        self.check_setup(dc, {'data3': ['a', 'b']})

    def test_astropy_table(self):
        with patch('glue.app.qt.GlueApplication') as ga:
            qglue(data4=self.astropy_table)
            dc = ga.call_args[0][0]
        self.check_setup(dc, {'data4': ['x', 'y']})

    def test_multi_data(self):
        with patch('glue.app.qt.GlueApplication') as ga:
            qglue(data1=self.dict_data, data2=self.xy)
            dc = ga.call_args[0][0]
        self.check_setup(dc, {'data1': ['u', 'v'],
                              'data2': ['x', 'y']})

    def test_hdulist(self):
        with patch('glue.app.qt.GlueApplication') as ga:
            qglue(data1=self.hdulist).data_collection
            dc = ga.call_args[0][0]
        self.check_setup(dc, {'data1': ['PRIMARY']})

    def test_glue_data(self):
        d = Data(x=[1, 2, 3])
        with patch('glue.app.qt.GlueApplication') as ga:
            qglue(x=d)
            dc = ga.call_args[0][0]
        assert d.label == 'x'

    def test_base_data(self):

        class CustomData(BaseCartesianData):

            def get_kind(self):
                pass

            def compute_histogram(self):
                pass

            def compute_statistic(self):
                pass

            def get_mask(self):
                pass

            @property
            def shape(self):
                return ()

            @property
            def main_components(self):
                return []

        d = CustomData()
        with patch('glue.app.qt.GlueApplication') as ga:
            qglue(x=d)
            dc = ga.call_args[0][0]

        assert dc[0] is d

    def test_simple_link(self):
        using = lambda x: x * 2
        links = [['data1.x', 'data2.u', using]]
        with patch('glue.app.qt.GlueApplication') as ga:
            qglue(data1=self.xy, data2=self.dict_data, links=links)
            dc = ga.call_args[0][0]

        links = [[['x'], 'u', using]]
        self.check_setup(dc, {'data1': ['x', 'y'],
                              'data2': ['u', 'v']})

        d = dc[0] if dc[0].label == 'data1' else dc[1]
        np.testing.assert_array_equal(d['x'], self.x)
        np.testing.assert_array_equal(d['u'], self.x * 2)
        d = dc[0] if dc[0].label == 'data2' else dc[1]
        with pytest.raises(IncompatibleAttribute) as exc:
            d['x']

    def test_multi_link(self):

        def forwards(x, y):
            return x * 2, y * 3

        def backwards(x, y):
            return x / 2, y / 3

        links = [[['Data1.x', 'Data1.y'],
                  ['Data2.u', 'Data2.v'], forwards, backwards]]
        with patch('glue.app.qt.GlueApplication') as ga:
            qglue(Data1=self.xy, Data2=self.dict_data, links=links)
            dc = ga.call_args[0][0]

        self.check_setup(dc, {'Data1': ['x', 'y'],
                              'Data2': ['u', 'v']})

        for d in dc:
            if d.label == 'Data1':
                np.testing.assert_array_equal(d['x'], self.x)
                np.testing.assert_array_equal(d['y'], self.y)
                np.testing.assert_array_equal(d['u'], self.x * 2)
                np.testing.assert_array_equal(d['v'], self.y * 3)
            else:
                np.testing.assert_array_equal(d['x'], self.u / 2)
                np.testing.assert_array_equal(d['y'], self.v / 3)
                np.testing.assert_array_equal(d['u'], self.u)
                np.testing.assert_array_equal(d['v'], self.v)

    def test_implicit_identity_link(self):
        links = [('Data1.x', 'Data2.v'),
                 ('Data1.y', 'Data2.u')]
        with patch('glue.app.qt.GlueApplication') as ga:
            qglue(Data1=self.xy, Data2=self.dict_data, links=links)
            dc = ga.call_args[0][0]
        # currently, identity links rename the second link to first,
        # so u/v disappear
        for d in dc:
            if d.label == 'Data1':
                np.testing.assert_array_equal(d['x'], self.x)
                np.testing.assert_array_equal(d['y'], self.y)
            else:
                np.testing.assert_array_equal(d['y'], self.u)
                np.testing.assert_array_equal(d['x'], self.v)

    def test_bad_link(self):
        forwards = lambda *args: args
        links = [(['Data1.a'], ['Data2.b'], forwards)]
        with pytest.raises(ValueError) as exc:
            qglue(Data1=self.xy, Data2=self.dict_data, links=links)
        assert exc.value.args[0] == "Invalid link (no component named Data1.a)"

    def test_bad_data_shape(self):
        with pytest.raises(ValueError) as exc:
            qglue(d=self.bad_data)
        assert exc.value.args[0].startswith("Invalid format for data 'd'")

    def test_bad_data_format(self):
        with pytest.raises(TypeError) as exc:
            qglue(d=5)
        assert exc.value.args[0].startswith("Invalid data description")

    def test_malformed_data_dict(self):
        with pytest.raises(ValueError) as exc:
            qglue(d={'x': 'bad'})
        assert exc.value.args[0].startswith("Invalid format for data 'd'")
