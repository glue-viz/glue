import numpy as np
from astropy.table import Table

from mock import patch
import pytest

from .. import qglue
from ..core.registry import Registry
from ..core.exceptions import IncompatibleAttribute

def has_pandas():
    try:
        import pandas
        return True
    except:
        return False

class TestQGlue(object):
    def setup_method(self, method):
        Registry().clear()

        x = [1, 2, 3]
        y = [2, 3, 4]

        u = [10, 20, 30, 40]
        v = [20, 40, 60, 80]

        self.xy = {'x':x, 'y':y}
        self.dict_data = {'u': u, 'v': v}
        self.recarray_data = np.rec.array([(0, 1), (2, 3)],
                                          dtype=[('a', 'i'), ('b', 'i')])
        self.astropy_table = Table({'x': x, 'y': y})
        self.bad_data = {'x': x, 'u': u}

        self.x = np.array(x)
        self.y = np.array(y)
        self.u = np.array(u)
        self.v = np.array(v)

    def check_setup(self, dc, expected):
        #assert that the assembled data collection returned
        #form qglue matches expected structure

        #test for expected data, components
        for data in dc:
            components = set(c.label for c in data.components)
            e = expected.pop(data.label)
            for component in e:
                assert component in components
        assert len(expected) == 0

    @pytest.mark.skipif("not has_pandas()")
    def test_qglue_starts_application(self):
        import pandas as pd
        pandas_data = pd.DataFrame(self.xy)

        with patch('glue.qt.glue_application.GlueApplication') as ga:
            dc = qglue(data1=pandas_data)
            ga.assert_called_once_with(dc)
            ga().start.assert_called_once_with()

    def test_single_pandas(self):
        with patch('glue.qt.glue_application.GlueApplication') as ga:
            dc = qglue(data1=self.xy)
            self.check_setup(dc, {'data1': ['x', 'y']})

    def test_single_dict(self):
        with patch('glue.qt.glue_application.GlueApplication') as ga:
            dc = qglue(data2=self.dict_data)
            self.check_setup(dc, {'data2': ['u', 'v']})

    def test_recarray(self):
        with patch('glue.qt.glue_application.GlueApplication') as ga:
            dc = qglue(data3=self.recarray_data)
            self.check_setup(dc, {'data3': ['a', 'b']})

    def test_astropy_table(self):
        with patch('glue.qt.glue_application.GlueApplication') as ga:
            dc = qglue(data4=self.astropy_table)
            self.check_setup(dc, {'data4': ['x', 'y']})

    def test_multi_data(self):
        with patch('glue.qt.glue_application.GlueApplication') as ga:
            dc = qglue(data1=self.dict_data, data2=self.xy)
            self.check_setup(dc, {'data1': ['u', 'v'],
                                  'data2': ['x', 'y']})

    def test_simple_link(self):
        with patch('glue.qt.glue_application.GlueApplication') as ga:
            using = lambda x: x * 2
            links = [['data1.x', 'data2.u', using]]
            dc = qglue(data1=self.xy, data2=self.dict_data,
                       links=links)

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
        with patch('glue.qt.glue_application.GlueApplication') as ga:
            forwards = lambda *args: (args[0] * 2, args[1] * 3)
            backwards = lambda *args: (args[0] / 2, args[1] / 3)

            links = [[['Data1.x', 'Data1.y'],
                      ['Data2.u', 'Data2.v'], forwards, backwards]]
            dc = qglue(Data1=self.xy, Data2=self.dict_data,
                       links=links)

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
        with patch('glue.qt.glue_application.GlueApplication') as ga:
            links = [('Data1.x', 'Data2.v'),
                     ('Data1.y', 'Data2.u')]
            dc = qglue(Data1=self.xy, Data2=self.dict_data,
                       links=links)
        #currently, identity links rename the second link to first,
        #so u/v disappear
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
            dc = qglue(Data1=self.xy, Data2=self.dict_data,
                       links=links)
        assert exc.value.args[0] == "Invalid link (no component named Data1.a)"

    def test_bad_data_shape(self):
        with pytest.raises(ValueError) as exc:
            dc = qglue(d=self.bad_data)
        assert exc.value.args[0].startswith("Invalid format for data 'd'")

    def test_bad_data_format(self):
        with pytest.raises(TypeError) as exc:
            dc = qglue(d=np.array([1, 2, 3]))
        assert exc.value.args[0].startswith("Invalid data description")

    def test_malformed_data_dict(self):
        with pytest.raises(ValueError) as exc:
            dc = qglue(d={'x': 'bad'})
        assert exc.value.args[0].startswith("Invalid format for data 'd'")
