# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103,W0612

from __future__ import absolute_import, division, print_function

import pytest
import numpy as np
from numpy.testing import assert_array_equal

from ..component_link import ComponentLink, BinaryComponentLink
from ..data import ComponentID, Data, Component
from ..data_collection import DataCollection
from ..link_helpers import LinkSame
from ..subset import InequalitySubsetState


class TestComponentLink(object):

    def toy_data(self):
        data = Data()
        from_comp = Component(np.array([1, 2, 3]))
        to_comp = Component(np.array([4, 5, 6]))
        return data, from_comp, to_comp

    def test_valid_init(self):
        ComponentLink([ComponentID('from')], ComponentID('to'))

    def test_valid_init_using(self):
        data, from_, to_ = self.toy_data()
        using = lambda x: x
        ComponentLink([ComponentID('from')], ComponentID('to'), using)

    def test_invalid_init_multi_from_no_using(self):
        with pytest.raises(TypeError) as exc:
            ComponentLink([ComponentID('a'), ComponentID('b')],
                          ComponentID('c'))
        assert exc.value.args[0] == ("comp_from must have only 1 element, "
                                     "or a 'using' function must be provided")

    def test_invalid_init_scalar_from(self):
        with pytest.raises(TypeError) as exc:
            ComponentLink(ComponentID('from'), ComponentID('to'))
        assert exc.value.args[0].startswith("comp_from must be a list")

    def test_compute_direct(self):
        data, from_, to_ = self.toy_data()
        from_id = data.add_component(from_, 'from_label')
        to_id = ComponentID('to_label')
        link = ComponentLink([from_id], to_id)

        result = link.compute(data)
        expected = from_.data
        assert_array_equal(result, expected)

    def test_compute_using(self):
        data, from_, to_ = self.toy_data()
        from_id = data.add_component(from_, 'from_label')
        to_id = ComponentID('to_label')
        using = lambda x: 3 * x
        link = ComponentLink([from_id], to_id, using)

        result = link.compute(data)
        expected = from_.data * 3
        assert_array_equal(result, expected)

    def test_getters(self):
        data, from_, to_ = self.toy_data()
        from_id = data.add_component(from_, 'from_label')
        to_id = ComponentID('to_label')
        using = lambda x: 3 * x
        link = ComponentLink([from_id], to_id, using)

        assert link.get_from_ids()[0] is from_id
        assert link.get_to_id() is to_id
        assert link.get_using() is using

    def test_str(self):
        """ str method returns without error """
        data, from_, to_ = self.toy_data()
        from_id = data.add_component(from_, 'from_label')
        to_id = ComponentID('to_label')
        link = ComponentLink([from_id], to_id)
        str(link)
        link = ComponentLink([from_id], to_id, using=lambda x: 3 * x)
        str(link)

    def test_repr(self):
        """ repr method returns without error """
        data, from_, to_ = self.toy_data()
        from_id = data.add_component(from_, 'from_label')
        to_id = ComponentID('to_label')
        link = ComponentLink([from_id], to_id)
        repr(link)

    def test_type_check(self):
        """Should raise an exception if non ComponentIDs are passed as input"""
        cid = ComponentID('test')
        with pytest.raises(TypeError) as exc:
            ComponentLink([None], cid)
        assert exc.value.args[0].startswith('from argument is not a list '
                                            'of ComponentIDs')

        with pytest.raises(TypeError) as exc:
            ComponentLink([cid], None)
        assert exc.value.args[0].startswith('to argument is not a ComponentID')

        with pytest.raises(TypeError) as exc:
            ComponentLink([cid, None], None, using=lambda x, y: None)
        assert exc.value.args[0].startswith('from argument is not a list '
                                            'of ComponentIDs')

l = ComponentLink([ComponentID('a')], ComponentID('b'))
cid = ComponentID('a')
scalar = 3


@pytest.mark.parametrize(('a', 'b'), [(l, l), (l, cid), (l, scalar)])
def test_arithmetic_overload(a, b):
    for x in [a + b, a - b, a * b, a / b, a ** b]:
        assert isinstance(x, BinaryComponentLink)
    for x in [b + a, b - a, b * a, b / a, b ** a]:
        assert isinstance(x, BinaryComponentLink)


@pytest.mark.parametrize(('a', 'b'), [(l, l), (l, cid), (l, scalar)])
def test_inequality_overload(a, b):
    for x in [a < b, a <= b, a > b, a >= b]:
        assert isinstance(x, InequalitySubsetState)
    for x in [b < a, b <= a, b > a, b >= a]:
        assert isinstance(x, InequalitySubsetState)


def test_link_bad_input():
    with pytest.raises(TypeError) as exc:
        BinaryComponentLink(ComponentID('x'), None, None)
    assert exc.value.args[0] == 'Cannot create BinaryComponentLink using None'

    with pytest.raises(TypeError) as exc:
        BinaryComponentLink(None, ComponentID('x'), None)
    assert exc.value.args[0] == 'Cannot create BinaryComponentLink using None'


def test_arithmetic_id_scalar():
    d = Data(x=[1, 2, 3, 4], y=[10, 20, 10, 20])
    assert_array_equal(d[d.id['x'] + 3], [4, 5, 6, 7])
    assert_array_equal(d[d.id['x'] - 3], [-2, -1, 0, 1])
    assert_array_equal(d[d.id['x'] * 3], [3, 6, 9, 12])
    assert_array_equal(d[d.id['y'] / 10], [1, 2, 1, 2])
    assert_array_equal(d[d.id['x'] ** 2], [1, 4, 9, 16])

    assert_array_equal(d[3 + d.id['x']], [4, 5, 6, 7])
    assert_array_equal(d[3 - d.id['x']], [2, 1, 0, -1])
    assert_array_equal(d[3 * d.id['x']], [3, 6, 9, 12])
    assert_array_equal(d[24 / d.id['x']], [24, 12, 8, 6])
    assert_array_equal(d[2 ** d.id['x']], [2, 4, 8, 16])


def test_arithmetic_id_id():
    d = Data(x=[1, 2, 3, 4], y=[10, 20, 10, 20])
    assert_array_equal(d[d.id['x'] + d.id['y']], [11, 22, 13, 24])
    assert_array_equal(d[d.id['x'] - d.id['y']], [-9, -18, -7, -16])
    assert_array_equal(d[d.id['x'] * d.id['y']], [10, 40, 30, 80])
    assert_array_equal(
        d[d.id['y'] / d.id['x']], [10, 10, 10 / 3, 5])
    assert_array_equal(d[d.id['y'] ** d.id['x']],
                       [10, 400, 1000, 20 ** 4])


def test_arithmetic_id_link():
    d = Data(x=[1, 2, 3, 4], y=[10, 20, 10, 20])
    y10 = d.id['y'] / 10
    assert_array_equal(d[d.id['x'] + y10], [2, 4, 4, 6])
    assert_array_equal(d[d.id['x'] - y10], [0, 0, 2, 2])
    assert_array_equal(d[d.id['x'] * y10], [1, 4, 3, 8])
    assert_array_equal(d[d.id['x'] / y10], [1, 1, 3, 2])
    assert_array_equal(d[d.id['x'] ** y10], [1, 4, 3, 16])

    assert_array_equal(d[y10 + d.id['x']], [2, 4, 4, 6])
    assert_array_equal(d[y10 - d.id['x']], [0, 0, -2, -2])
    assert_array_equal(d[y10 * d.id['x']], [1, 4, 3, 8])
    assert_array_equal(d[y10 / d.id['x']], [1, 1, 1 / 3., 1 / 2.])
    assert_array_equal(d[y10 ** d.id['x']], [1, 4, 1, 16])


def test_arithmetic_link_link():
    d = Data(x=[1, 2, 3, 4], y=[10, 20, 10, 20])
    x = d[d.id['x']]
    y = d[d.id['y']]
    xpy = d.id['x'] + d.id['y']
    xt3 = d.id['x'] * 3
    assert_array_equal(d[xpy + xt3], x + y + x * 3)
    assert_array_equal(d[xpy - xt3], x + y - x * 3)
    assert_array_equal(d[xpy * xt3], (x + y) * x * 3)
    assert_array_equal(d[xpy / xt3], (x + y) / (x * 3))
    assert_array_equal(d[xpy ** xt3], (x + y) ** (x * 3))


def test_inequality():
    d = Data(x=[1, 2, 3, 4], y=[10, 20, 10, 20])
    s = d.new_subset()

    xpy = d.id['x'] + d.id['y']
    twentytwo = xpy * 0 + 22
    x = d[d.id['x']]
    y = d[d.id['y']]

    s.subset_state = xpy < 22
    assert_array_equal(s.to_mask(), (x + y) < 22)

    s.subset_state = xpy <= 22
    assert_array_equal(s.to_mask(), (x + y) <= 22)

    s.subset_state = xpy >= 22
    assert_array_equal(s.to_mask(), (x + y) >= 22)

    s.subset_state = xpy > 22
    assert_array_equal(s.to_mask(), (x + y) > 22)

    s.subset_state = 22 < xpy
    assert_array_equal(s.to_mask(), 22 < (x + y))

    s.subset_state = 22 <= xpy
    assert_array_equal(s.to_mask(), 22 <= (x + y))

    s.subset_state = 22 > xpy
    assert_array_equal(s.to_mask(), 22 > (x + y))

    s.subset_state = 22 >= xpy
    assert_array_equal(s.to_mask(), 22 >= (x + y))

    s.subset_state = twentytwo < xpy
    assert_array_equal(s.to_mask(), 22 < (x + y))

    s.subset_state = twentytwo <= xpy
    assert_array_equal(s.to_mask(), 22 <= (x + y))

    s.subset_state = twentytwo > xpy
    assert_array_equal(s.to_mask(), 22 > (x + y))

    s.subset_state = twentytwo >= xpy
    assert_array_equal(s.to_mask(), 22 >= (x + y))


def test_link_fixes_shape():
    def double(x):
        return (x * 2).reshape((2, 2))

    d = Data(x=[1, 2, 3, 4])
    y = ComponentID('y')
    link = ComponentLink([d.id['x']], y, using=double)
    assert_array_equal(d[link], [2, 4, 6, 8])


def test_link_str():
    """Links should have sensible names"""
    d = Data(x=[1, 2, 3], y=[2, 3, 4])
    x = d.id['x']
    y = d.id['y']

    assert str(x + y) == ('(x + y)')
    assert str(x - y) == ('(x - y)')
    assert str(x * y) == ('(x * y)')
    assert str(x / y) == ('(x / y)')
    assert str(x ** y) == ('(x ** y)')
    assert str(x ** 3) == ('(x ** 3)')
    assert str(3 + x * y) == ('(3 + (x * y))')
    assert str(x + x + y) == ('((x + x) + y)')

    assert repr(x + y) == '<BinaryComponentLink: (x + y)>'


def test_duplicated_links_remove_first_input():
    """
    # test changes introduced for #508
    """

    d1 = Data(x=[1, 2, 3])
    d2 = Data(y=[2, 4, 6])

    x = d1.id['x']
    y = d2.id['y']

    dc = DataCollection([d1, d2])

    dc.add_link(LinkSame(x, y))
    assert y not in d2.components
    assert y not in d1.components
    assert x in d2.components
    assert x in d2.components
