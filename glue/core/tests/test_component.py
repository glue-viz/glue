import operator

from mock import MagicMock
import pytest

from ..data import Component, ComponentID, DerivedComponent
from ... import core


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
