#pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103,W0612
import pytest

import numpy as np

from ..data import ComponentID, Data, Component
from ..component_link import ComponentLink, BinaryComponentLink
from ..subset import InequalitySubsetState


class TestComponentLink(object):

    def toy_data(self):
        data = Data()
        from_comp = Component(np.array([1, 2, 3]))
        to_comp = Component(np.array([4, 5, 6]))
        return data, from_comp, to_comp

    def test_valid_init(self):
        link = ComponentLink([ComponentID('from')], ComponentID('to'))

    def test_valid_init_using(self):
        data, from_, to_ = self.toy_data()
        using = lambda x: x
        link = ComponentLink([ComponentID('from')], ComponentID('to'), using)

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
        np.testing.assert_array_equal(result, expected)

    def test_compute_using(self):
        data, from_, to_ = self.toy_data()
        from_id = data.add_component(from_, 'from_label')
        to_id = ComponentID('to_label')
        using = lambda x: 3 * x
        link = ComponentLink([from_id], to_id, using)

        result = link.compute(data)
        expected = from_.data * 3
        np.testing.assert_array_equal(result, expected)

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
        assert exc.value.args[0] == \
            'from argument is not a list of ComponentIDs'

        with pytest.raises(TypeError) as exc:
            ComponentLink([cid], None)
        assert exc.value.args[0] == 'to argument is not a ComponentID'

        with pytest.raises(TypeError) as exc:
            ComponentLink([cid, None], None, using=lambda x, y: None)
        assert exc.value.args[0] == \
            'from argument is not a list of ComponentIDs'

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


def test_arithmetic_id_scalar():
    d = Data(x=[1, 2, 3, 4], y=[10, 20, 10, 20])
    np.testing.assert_array_equal(d[d.id['x'] + 3], [4, 5, 6, 7])
    np.testing.assert_array_equal(d[d.id['x'] - 3], [-2, -1, 0, 1])
    np.testing.assert_array_equal(d[d.id['x'] * 3], [3, 6, 9, 12])
    np.testing.assert_array_equal(d[d.id['y'] / 10], [1, 2, 1, 2])
    np.testing.assert_array_equal(d[d.id['x'] ** 2], [1, 4, 9, 16])

    np.testing.assert_array_equal(d[3 + d.id['x']], [4, 5, 6, 7])
    np.testing.assert_array_equal(d[3 - d.id['x']], [2, 1, 0, -1])
    np.testing.assert_array_equal(d[3 * d.id['x']], [3, 6, 9, 12])
    np.testing.assert_array_equal(d[24 / d.id['x']], [24, 12, 8, 6])
    np.testing.assert_array_equal(d[2 ** d.id['x']], [2, 4, 8, 16])


def test_arithmetic_id_id():
    d = Data(x=[1, 2, 3, 4], y=[10, 20, 10, 20])
    np.testing.assert_array_equal(d[d.id['x'] + d.id['y']], [11, 22, 13, 24])
    np.testing.assert_array_equal(d[d.id['x'] - d.id['y']], [-9, -18, -7, -16])
    np.testing.assert_array_equal(d[d.id['x'] * d.id['y']], [10, 40, 30, 80])
    np.testing.assert_array_equal(
        d[d.id['y'] / d.id['x']], [10, 10, 10 / 3, 5])
    np.testing.assert_array_equal(d[d.id['y'] ** d.id['x']],
                                  [10, 400, 1000, 20 ** 4])


def test_arithmetic_id_link():
    d = Data(x=[1, 2, 3, 4], y=[10, 20, 10, 20])
    y10 = d.id['y'] / 10
    np.testing.assert_array_equal(d[d.id['x'] + y10], [2, 4, 4, 6])
    np.testing.assert_array_equal(d[d.id['x'] - y10], [0, 0, 2, 2])
    np.testing.assert_array_equal(d[d.id['x'] * y10], [1, 4, 3, 8])
    np.testing.assert_array_equal(d[d.id['x'] / y10], [1, 1, 3, 2])
    np.testing.assert_array_equal(d[d.id['x'] ** y10], [1, 4, 3, 16])

    np.testing.assert_array_equal(d[y10 + d.id['x']], [2, 4, 4, 6])
    np.testing.assert_array_equal(d[y10 - d.id['x']], [0, 0, -2, -2])
    np.testing.assert_array_equal(d[y10 * d.id['x']], [1, 4, 3, 8])
    np.testing.assert_array_equal(d[y10 / d.id['x']], [1, 1, 0, 0])
    np.testing.assert_array_equal(d[y10 ** d.id['x']], [1, 4, 1, 16])


def test_arithmetic_link_link():
    d = Data(x=[1, 2, 3, 4], y=[10, 20, 10, 20])
    x = d[d.id['x']]
    y = d[d.id['y']]
    xpy = d.id['x'] + d.id['y']
    xt3 = d.id['x'] * 3
    np.testing.assert_array_equal(d[xpy + xt3], x + y + x * 3)
    np.testing.assert_array_equal(d[xpy - xt3], x + y - x * 3)
    np.testing.assert_array_equal(d[xpy * xt3], (x + y) * x * 3)
    np.testing.assert_array_equal(d[xpy / xt3], (x + y) / (x * 3))
    np.testing.assert_array_equal(d[xpy ** xt3], (x + y) ** (x * 3))


def test_inequality():
    d = Data(x=[1, 2, 3, 4], y=[10, 20, 10, 20])
    s = d.new_subset()

    xpy = d.id['x'] + d.id['y']
    twentytwo = xpy * 0 + 22
    x = d[d.id['x']]
    y = d[d.id['y']]

    s.subset_state = xpy < 22
    np.testing.assert_array_equal(s.to_mask(), (x + y) < 22)

    s.subset_state = xpy <= 22
    np.testing.assert_array_equal(s.to_mask(), (x + y) <= 22)

    s.subset_state = xpy >= 22
    np.testing.assert_array_equal(s.to_mask(), (x + y) >= 22)

    s.subset_state = xpy > 22
    np.testing.assert_array_equal(s.to_mask(), (x + y) > 22)

    s.subset_state = 22 < xpy
    np.testing.assert_array_equal(s.to_mask(), 22 < (x + y))

    s.subset_state = 22 <= xpy
    np.testing.assert_array_equal(s.to_mask(), 22 <= (x + y))

    s.subset_state = 22 > xpy
    np.testing.assert_array_equal(s.to_mask(), 22 > (x + y))

    s.subset_state = 22 >= xpy
    np.testing.assert_array_equal(s.to_mask(), 22 >= (x + y))

    s.subset_state = twentytwo < xpy
    np.testing.assert_array_equal(s.to_mask(), 22 < (x + y))

    s.subset_state = twentytwo <= xpy
    np.testing.assert_array_equal(s.to_mask(), 22 <= (x + y))

    s.subset_state = twentytwo > xpy
    np.testing.assert_array_equal(s.to_mask(), 22 > (x + y))

    s.subset_state = twentytwo >= xpy
    np.testing.assert_array_equal(s.to_mask(), 22 >= (x + y))
