# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

import operator

import pytest
import numpy as np
from unittest.mock import MagicMock

from astropy.wcs import WCS

from glue import core
from glue.tests.helpers import requires_astropy

from ..coordinates import Coordinates
from ..component import (Component, DerivedComponent, CoordinateComponent,
                         CategoricalComponent)
from ..component_id import ComponentID
from ..data import Data
from ..parse import ParsedCommand, ParsedComponentLink
from ..data_collection import DataCollection

VIEWS = (np.s_[:], np.s_[1], np.s_[::-1], np.s_[0, :])


class TestComponent(object):

    def setup_method(self, method):
        self.data = MagicMock()
        self.data.shape = [1, 2]
        self.component = Component(self.data)

    def test_data(self):
        assert self.component.data is self.data

    def test_shape(self):
        assert self.component.shape is self.data.shape

    def test_ndim(self):
        assert self.component.ndim is len(self.data.shape)


class TestComponentID(object):

    def setup_method(self, method):
        self.cid = ComponentID('test')

    def test_label(self):
        assert self.cid.label == 'test'

    def test_str(self):
        """ str should return """
        str(self.cid)

    def test_repr(self):
        """ str should return """
        repr(self.cid)


class TestDerivedComponent(object):

    def setup_method(self, method):
        data = MagicMock()
        link = MagicMock()
        self.cid = DerivedComponent(data, link)
        self.link = link
        self.data = data

    def test_data(self):
        """ data property should wrap to links compute method """
        self.cid.data
        self.link.compute.assert_called_once_with(self.data)

    def test_link(self):
        assert self.cid.link == self.link


class TestCategoricalComponent(object):

    def setup_method(self, method):
        self.list_data = ['a', 'a', 'b', 'b']
        self.array_data = np.array(self.list_data)

    def test_autodetection(self):
        assert Component.autotyped(self.array_data).categorical
        assert Component.autotyped(self.list_data).categorical

        x = np.array([True, False, True, False])
        assert not Component.autotyped(x).categorical

        x = np.array([1, 2, 3, 4])
        assert not Component.autotyped(x).categorical

        x = np.array(['1', '2', '3', '4'])
        assert not Component.autotyped(x).categorical

        d = Data(x=['a', 'b', 'c'])
        assert d.get_component('x').categorical

    def test_basic_properties(self):
        data = ['a', 'b', 'c', 'b', 'b', 'c', 'a', 'c', 'd']
        cat_comp = CategoricalComponent(data)
        np.testing.assert_equal(cat_comp.data, data)
        np.testing.assert_equal(cat_comp.labels, data)
        np.testing.assert_equal(cat_comp.codes, [0, 1, 2, 1, 1, 2, 0, 2, 3])
        np.testing.assert_equal(cat_comp.categories, ['a', 'b', 'c', 'd'])

    def test_accepts_numpy(self):
        cat_comp = CategoricalComponent(self.array_data)
        assert cat_comp.data.shape == (4,)

    def test_accepts_pandas(self):
        # Regression test for a bug that caused read-only
        # pandas DataFrame columns to cause issues
        from pandas import DataFrame
        df = DataFrame()
        df['a'] = ['a', 'b', 'c']
        CategoricalComponent(df['a'].fillna(''))

    def test_accepts_list(self):
        """Should accept a list and convert to numpy!"""
        cat_comp = CategoricalComponent(self.list_data)
        np.testing.assert_equal(cat_comp.data, self.array_data)

    def test_multi_nans(self):
        cat_comp = CategoricalComponent(['', '', 'a', 'b', 'c', 'zanthia'])
        np.testing.assert_equal(cat_comp.codes,
                                np.array([0, 0, 1, 2, 3, 4]))
        np.testing.assert_equal(cat_comp.categories,
                                np.asarray(['', 'a', 'b', 'c', 'zanthia']))

    def test_calculate_grouping(self):
        cat_comp = CategoricalComponent(self.array_data)
        np.testing.assert_equal(cat_comp.categories, np.asarray(['a', 'b']))
        np.testing.assert_equal(cat_comp.codes, np.array([0, 0, 1, 1]))
        assert cat_comp.codes.dtype == float

    def test_accepts_provided_grouping(self):
        ncategories = ['b', 'c']
        cat_data = list('aaabbbcccddd')
        cat_comp = CategoricalComponent(cat_data, categories=ncategories)

        assert cat_comp.categories == ncategories
        assert np.all(np.isnan(cat_comp.codes[:3]))
        assert np.all(cat_comp.codes[3:6] == 0)
        assert np.all(cat_comp.codes[6:9] == 1)
        assert np.all(np.isnan(cat_comp.codes[9:]))

    def test_uniform_jitter(self):
        cat_comp = CategoricalComponent(self.array_data)
        second_comp = CategoricalComponent(self.array_data)
        cat_comp.jitter(method='uniform')
        assert np.all(cat_comp.codes != second_comp.codes), "Didn't jitter data!"

    def test_unjitter_data(self):
        cat_comp = CategoricalComponent(self.array_data)
        second_comp = CategoricalComponent(self.array_data)

        cat_comp.jitter(method='uniform')
        delta = np.abs(cat_comp.codes - second_comp.codes).sum()
        assert delta > 0

        cat_comp.jitter(method=None)
        np.testing.assert_equal(cat_comp.codes,
                                second_comp.codes,
                                "Didn't un-jitter data!")

    def test_jitter_on_init(self):
        cat_comp = CategoricalComponent(self.array_data, jitter='uniform')
        second_comp = CategoricalComponent(self.array_data)
        assert np.all(cat_comp.codes != second_comp.codes)

    def test_object_dtype(self):
        d = np.array([1, 3, 3, 1, 'a', 'b', 'a'], dtype=object)
        c = CategoricalComponent(d)

        np.testing.assert_array_equal(c.categories,
                                      np.array([1, 3, 'a', 'b'], dtype=object))
        np.testing.assert_array_equal(c.codes, [0, 1, 1, 0, 2, 3, 2])

    def test_valueerror_on_bad_jitter(self):

        with pytest.raises(ValueError):
            cat_comp = CategoricalComponent(self.array_data)
            cat_comp.jitter(method='this will never be a jitter method')


def test_nd_categorical_component():
    data = [['a', 'b'], ['c', 'b']]
    cat_comp = CategoricalComponent(data)
    np.testing.assert_equal(cat_comp.data, data)
    np.testing.assert_equal(cat_comp.labels, data)
    np.testing.assert_equal(cat_comp.codes, [[0, 1], [2, 1]])
    np.testing.assert_equal(cat_comp.categories, ['a', 'b', 'c'])


def test_nd_categorical_component_autotype():
    x = np.array([['M', 'F'], ['F', 'M']])
    assert Component.autotyped(x).categorical

    d = Data(x=[['male', 'female', 'male'], ['female', 'female', 'male']])
    assert d.get_component('x').categorical


class TestCoordinateComponent(object):

    def setup_method(self, method):

        class TestCoords(Coordinates):

            def __init__(self):
                super().__init__(pixel_n_dim=3, world_n_dim=3)

            def pixel_to_world_values(self, *args):
                return [a * (i + 1) for i, a in enumerate(args)]

            def world_to_pixel_values(self, *args):
                return [a / (i + 1) for i, a in enumerate(args)]

            @property
            def axis_correlation_matrix(self):
                return np.identity(3).astype(bool)

        data = core.Data()
        data.add_component(Component(np.zeros((3, 3, 3))), 'test')
        data.coords = TestCoords()
        self.data = data
        self.px = CoordinateComponent(data, 2)
        self.py = CoordinateComponent(data, 1)
        self.pz = CoordinateComponent(data, 0)
        self.wx = CoordinateComponent(data, 2, world=True)
        self.wy = CoordinateComponent(data, 1, world=True)
        self.wz = CoordinateComponent(data, 0, world=True)

    def test_data(self):
        z, y, x = np.mgrid[0:3, 0:3, 0:3]
        np.testing.assert_array_equal(self.px.data, x)
        np.testing.assert_array_equal(self.py.data, y)
        np.testing.assert_array_equal(self.pz.data, z)
        np.testing.assert_array_equal(self.wx.data, x * 1)
        np.testing.assert_array_equal(self.wy.data, y * 2)
        np.testing.assert_array_equal(self.wz.data, z * 3)

    @pytest.mark.parametrize(('view'), VIEWS)
    def test_view(self, view):
        z, y, x = np.mgrid[0:3, 0:3, 0:3]

        np.testing.assert_array_equal(self.px[view], x[view])
        np.testing.assert_array_equal(self.py[view], y[view])
        np.testing.assert_array_equal(self.pz[view], z[view])
        np.testing.assert_array_equal(self.wx[view], x[view] * 1)
        np.testing.assert_array_equal(self.wy[view], y[view] * 2)
        np.testing.assert_array_equal(self.wz[view], z[view] * 3)


def check_binary(result, left, right, op):
    assert isinstance(result, core.subset.InequalitySubsetState)
    assert result.left is left
    assert result.right is right
    assert result.operator is op


def check_link(result, left, right):
    assert isinstance(result, core.component_link.ComponentLink)
    if isinstance(left, ComponentID):
        assert left in result.get_from_ids()
    if isinstance(right, ComponentID):
        assert right in result.get_from_ids()


# componentID overload
COMPARE_OPS = (operator.gt, operator.ge, operator.lt, operator.le)
NUMBER_OPS = (operator.add, operator.mul, operator.truediv, operator.sub)


@pytest.mark.parametrize(('op'), COMPARE_OPS)
def test_inequality_scalar(op):
    cid = ComponentID('test')
    result = op(cid, 3)
    check_binary(result, cid, 3, op)


@pytest.mark.parametrize(('op'), COMPARE_OPS)
def test_inequality_id(op):
    cid = ComponentID('test')
    cid2 = ComponentID('test2')
    result = op(cid, cid2)
    check_binary(result, cid, cid2, op)


@pytest.mark.parametrize(('op'), NUMBER_OPS)
def test_arithmetic_scalar(op):
    cid = ComponentID('test')
    result = op(cid, 3)
    check_link(result, cid, 3)


@pytest.mark.parametrize(('op'), NUMBER_OPS)
def test_arithmetic_scalar_right(op):
    cid = ComponentID('test')
    result = op(3, cid)
    check_link(result, 3, cid)


@pytest.mark.parametrize(('op'), NUMBER_OPS)
def test_arithmetic_cid(op):
    cid = ComponentID('test')
    cid2 = ComponentID('test2')
    result = op(cid, cid2)
    check_link(result, cid, cid2)


def test_pow_scalar():
    cid = ComponentID('test')
    result = cid ** 3
    check_link(result, cid, 3)


@pytest.mark.parametrize(('view'), VIEWS)
def test_view(view):
    comp = Component(np.array([[1, 2, 3], [2, 3, 4]]))
    np.testing.assert_array_equal(comp[view], comp.data[view])


@pytest.mark.parametrize(('view'), VIEWS)
def test_view_derived(view):
    comp = Component(np.array([[1, 2, 3], [2, 3, 4]]))
    d = core.Data()
    cid = d.add_component(comp, 'primary')
    cid2 = ComponentID("derived")
    link = core.ComponentLink([cid], cid2, using=lambda x: x * 3)
    dc = DerivedComponent(d, link)

    np.testing.assert_array_equal(dc[view], comp.data[view] * 3)


@requires_astropy
def test_units():

    # Make sure that units get converted to strings. At the moment if these
    # are set to Astropy units for example, things can go wrong for example
    # when writing out the datasets. Once we settle on a units framework, we
    # can then use that instead of converting units to strings.

    from astropy import units as u

    comp = Component([1, 2, 3], units='m')
    assert comp.units == 'm'
    assert isinstance(comp.units, str)

    comp = Component([1, 2, 3], units=u.m)
    assert comp.units == 'm'
    assert isinstance(comp.units, str)


def test_scalar_derived():
    # Regression test for a bug that caused expressions without components to
    # fail. Note that the data collection is needed here as the bug occurs in
    # the link manager set up by the data collection.
    data = Data(a=[3, 4, 1])
    dc = DataCollection([data])  # noqa
    pc = ParsedCommand('3.5', {})
    link = ParsedComponentLink(ComponentID('b'), pc)
    data.add_component_link(link)
    np.testing.assert_equal(data['b'], [3.5, 3.5, 3.5])


def test_rename_cid_used_in_derived():
    # Unit test to make sure expressions still work when renaming components
    data = Data(a=[3, 4, 1])
    dc = DataCollection([data])  # noqa
    pc = ParsedCommand('1 + {a}', {'a': data.id['a']})
    link = ParsedComponentLink(ComponentID('b'), pc)
    data.add_component_link(link)
    np.testing.assert_equal(data['b'], [4, 5, 2])
    data.id['a'].label = 'x'
    np.testing.assert_equal(data['b'], [4, 5, 2])


@pytest.mark.xfail
def test_update_cid_used_in_derived():
    # Unit test to make sure expressions still work when updating the component
    # id for components
    data = Data(a=[3, 4, 1])
    dc = DataCollection([data])  # noqa
    pc = ParsedCommand('1 + {a}', {'a': data.id['a']})
    link = ParsedComponentLink(ComponentID('b'), pc)
    data.add_component_link(link)
    np.testing.assert_equal(data['b'], [4, 5, 2])
    data.update_id(data.id['a'], ComponentID('x'))
    np.testing.assert_equal(data['b'], [4, 5, 2])


def test_coordinate_component_1d_coord():

    # Regression test for a bug that caused incorrect world coordinate values
    # for 1D coordinates.

    wcs = WCS(naxis=1)
    wcs.wcs.ctype = ['FREQ']
    wcs.wcs.crpix = [1]
    wcs.wcs.crval = [1]
    wcs.wcs.cdelt = [1]

    data = Data(flux=np.random.random(5), coords=wcs, label='data')
    np.testing.assert_equal(data['Frequency'], [1, 2, 3, 4, 5])
