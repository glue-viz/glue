#pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103
import operator

from mock import MagicMock
import pytest
import numpy as np

from ..data import (Component, ComponentID,
                    DerivedComponent, CoordinateComponent,
                    CategoricalComponent)
from ... import core


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

    def test_accepts_numpy(self):
        cat_comp = CategoricalComponent(self.array_data)
        assert cat_comp._categorical_data.shape == (4,)

    def test_accepts_list(self):
        """Should accept a list and convert to numpy!"""
        cat_comp = CategoricalComponent(self.list_data)
        assert np.all(cat_comp._categorical_data == self.array_data)

    def test_multi_nans(self):
        cat_comp = CategoricalComponent([np.nan, np.nan, 'a', 'b', 'c', 'zanthia'])
        np.testing.assert_equal(cat_comp._data,
                                np.array([0, 0, 1, 2, 3, 4]))
        np.testing.assert_equal(cat_comp._categories,
                                np.asarray([np.nan, 'a', 'b', 'c', 'zanthia'],
                                           dtype=np.object))

    def test_calculate_grouping(self):
        cat_comp = CategoricalComponent(self.array_data)
        assert np.all(cat_comp._categories == np.asarray(['a', 'b']))
        assert np.all(cat_comp._data == np.array([0, 0, 1, 1]))

    def test_accepts_provided_grouping(self):
        ncategories = ['b', 'c']
        cat_comp = CategoricalComponent(self.array_data, categories=ncategories)

        assert cat_comp._categories == ncategories
        assert np.all(np.isnan(cat_comp._data[:1]))
        assert np.all(cat_comp._data[2:] == 0)
        assert not np.any(cat_comp._data == 1)


class TestCoordinateComponent(object):

    def setup_method(self, method):
        class TestCoords(object):
            def pixel2world(self, *args):
                return [a * (i + 1) for i, a in enumerate(args)]

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

#componentID overload
COMPARE_OPS = (operator.gt, operator.ge, operator.lt, operator.le)
NUMBER_OPS = (operator.add, operator.mul, operator.div, operator.sub)


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
