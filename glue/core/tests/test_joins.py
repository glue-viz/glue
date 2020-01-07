import pytest
from numpy.testing import assert_array_equal

from .. import Data, DataCollection
from ..exceptions import IncompatibleAttribute
from .test_state import clone


class TestSubsets(object):

    def test_basic(self):
        x = Data(id=[0, 1, 2])
        y = Data(id=[0, 1, 2, 3], x=[1, 2, 3, 4])
        x.join_on_key(y, 'id', 'id')

        s = x.new_subset()
        s.subset_state = y.id['x'] > 1

        assert_array_equal(s.to_mask(), [False, True, True])

        # Make sure this also works if we use the Data.get_mask API
        assert_array_equal(x.get_mask(s.subset_state), [False, True, True])

    def test_basic_to_index_list(self):
        x = Data(id=[0, 1, 2])
        y = Data(id=[0, 1, 2, 3], x=[1, 2, 3, 4])
        x.join_on_key(y, 'id', 'id')

        s = x.new_subset()
        s.subset_state = y.id['x'] > 1

        assert_array_equal(s.to_index_list(), [1, 2])

    def test_permute(self):
        x = Data(id=[1, 2, 1])
        y = Data(id=[2, 0, 1, 5], x=[1, 2, 3, 2])
        x.join_on_key(y, 'id', 'id')

        s = x.new_subset()
        s.subset_state = y.id['x'] < 3
        assert_array_equal(s.to_mask(), [False, True, False])

        s.subset_state = y.id['x'] > 1
        assert_array_equal(s.to_mask(), [True, False, True])

    def test_multidim(self):
        x = Data(id=[[0, 0], [1, 2]])
        y = Data(id=[2, 0, 1], x=[1, 2, 3])
        x.join_on_key(y, 'id', 'id')

        s = x.new_subset()
        s.subset_state = y.id['x'] > 1

        assert_array_equal(s.to_mask(), [[True, True], [True, False]])

    def test_mismatch(self):
        x = Data(id=[3, 4, 5])
        y = Data(id=[0, 0, 0, 0], x=[1, 2, 3, 4])
        x.join_on_key(y, 'id', 'id')

        s = x.new_subset()
        s.subset_state = y.id['x'] > 1

        assert_array_equal(s.to_mask(), [False, False, False])

    def test_inverse_match(self):

        x = Data(id=[0, 1, 2, 3], x=[5, 6, 7, 8])
        y = Data(id=[2, 1, 0], y=[1, 2, 3])
        x.join_on_key(y, 'id', 'id')

        s = y.new_subset()
        s.subset_state = x.id['x'] > 6

        assert_array_equal(s.to_mask(), [True, False, False])

    def test_join_chain(self):
        x = Data(id1=[0, 1, 2], label='x')
        y = Data(id1=[2, 1, 0, 3], id2=[3, 4, 5, 6], label='y')
        z = Data(id2=[5, 4, 5], z=[1, 2, 3], label='z')

        x.join_on_key(y, 'id1', 'id1')
        y.join_on_key(z, 'id2', 'id2')

        s = x.new_subset()
        s.subset_state = z.id['z'] > 2

        assert_array_equal(s.to_mask(), [True, False, False])
        with pytest.raises(IncompatibleAttribute):
            w = Data(w=[1, 2])
            s.subset_state = w.id['w'] > 1
            s.to_mask()

    def test_incompatible_attibute_without_join(self):
        x = Data(id1=[0, 1, 2], label='x')
        y = Data(y=[1, 2, 3, 4])

        s = x.new_subset()
        s.subset_state = y.id['y'] > 2

        with pytest.raises(IncompatibleAttribute):
            s.to_mask()

    def test_bad_join_key(self):
        x = Data(id1=[0, 1, 2], label='x')
        y = Data(id1=[1, 2, 3, 4], label='y')

        with pytest.raises(ValueError) as exc:
            x.join_on_key(y, 'bad_key', 'id1')
        assert exc.value.args[0] == 'ComponentID not found in x: bad_key'

        with pytest.raises(ValueError) as exc:
            x.join_on_key(y, 'id1', 'bad_key')
        assert exc.value.args[0] == 'ComponentID not found in y: bad_key'

    def test_clone(self):
        x = Data(id=[0, 1, 2], label='data_x')
        y = Data(id=[0, 1, 2, 3], x=[1, 2, 3, 4], label='data_y')
        x.join_on_key(y, 'id', 'id')

        dc = DataCollection([x, y])
        dc = clone(dc)

        x, y = dc
        s = x.new_subset()
        s.subset_state = y.id['x'] > 1

        assert_array_equal(s.to_mask(), [False, True, True])


def test_many_to_many():
    """
    Test the use of multiple keys to denote that combinations of components
    have to match.
    """

    d1 = Data(x=[1, 2, 3, 5, 5],
              y=[0, 0, 1, 1, 2], label='d1')
    d2 = Data(a=[2, 5, 5, 8, 4, 9],
              b=[1, 3, 2, 2, 3, 9], label='d2')
    d2.join_on_key(d1, ('a', 'b'), ('x', 'y'))

    s = d1.new_subset()
    s.subset_state = d1.id['x'] == 5
    assert_array_equal(s.to_mask(), [0, 0, 0, 1, 1])

    s = d2.new_subset()
    s.subset_state = d1.id['x'] == 5
    assert_array_equal(s.to_mask(), [0, 0, 1, 0, 0, 0])


def test_one_and_many():
    """
    Test the use of one-to-many keys or many-to-one key to indicate that any of
    the components can match the other.
    """

    d1 = Data(x=[1, 2, 3], label='d1')
    d2 = Data(a=[1, 1, 2, 5],
              b=[2, 3, 3, 5], label='d2')
    d1.join_on_key(d2, 'x', ('a', 'b'))

    s = d2.new_subset()
    s.subset_state = d2.id['a'] == 2
    assert_array_equal(s.to_mask(), [0, 0, 1, 0])

    s = d1.new_subset()
    s.subset_state = d2.id['a'] == 2
    assert_array_equal(s.to_mask(), [0, 1, 1])

    d1 = Data(x=[1, 2, 3], label='d1')
    d2 = Data(a=[1, 1, 2, 5],
              b=[2, 3, 3, 5], label='d2')
    d2.join_on_key(d1, ('a', 'b'), 'x')

    s = d1.new_subset()
    s.subset_state = d1.id['x'] == 1
    assert_array_equal(s.to_mask(), [1, 0, 0])

    s = d2.new_subset()
    s.subset_state = d1.id['x'] == 1
    assert_array_equal(s.to_mask(), [1, 1, 0, 0])


def test_mismatch():

    d1 = Data(x=[1, 1, 2],
              y=[2, 3, 3],
              z=[2, 3, 3], label='d1')
    d2 = Data(a=[1, 1, 2],
              b=[2, 3, 3], label='d2')

    with pytest.raises(Exception) as exc:
        d1.join_on_key(d2, ('x', 'y', 'z'), ('a', 'b'))
    assert exc.value.args[0] == ("Either the number of components in the key "
                                 "join sets should match, or one of the "
                                 "component sets should contain a single "
                                 "component.")
