from numpy.testing import assert_array_equal
import pytest

from .. import Data, DataCollection
from ..exceptions import IncompatibleAttribute
from .test_state import clone


class TestSubsets(object):

    def test_basic(self):
        x = Data(id=[0, 1, 2])
        y = Data(id=[0, 1, 2], x=[1, 2, 3])
        x.join_on_key(y, 'id', 'id')

        s = x.new_subset()
        s.subset_state = y.id['x'] > 1

        assert_array_equal(s.to_mask(), [False, True, True])

    def test_basic_to_index_list(self):
        x = Data(id=[0, 1, 2])
        y = Data(id=[0, 1, 2], x=[1, 2, 3])
        x.join_on_key(y, 'id', 'id')

        s = x.new_subset()
        s.subset_state = y.id['x'] > 1

        assert_array_equal(s.to_index_list(), [1, 2])

    def test_permute(self):
        x = Data(id=[1, 2, 1])
        y = Data(id=[2, 0, 1], x=[1, 2, 3])
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
        y = Data(id=[0, 0, 0], x=[1, 2, 3])
        x.join_on_key(y, 'id', 'id')

        s = x.new_subset()
        s.subset_state = y.id['x'] > 1

        assert_array_equal(s.to_mask(), [False, False, False])

    def test_inverse_match(self):

        x = Data(id=[0, 1, 2], x=[5, 6, 7])
        y = Data(id=[2, 1, 0], y=[1, 2, 3])
        x.join_on_key(y, 'id', 'id')

        s = y.new_subset()
        s.subset_state = x.id['x'] > 6

        assert_array_equal(s.to_mask(), [True, False, False])

    def test_join_chain(self):
        x = Data(id1=[0, 1, 2], label='x')
        y = Data(id1=[2, 1, 0], id2=[3, 4, 5], label='y')
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
        y = Data(y=[1, 2, 3])

        s = x.new_subset()
        s.subset_state = y.id['y'] > 2

        with pytest.raises(IncompatibleAttribute):
            s.to_mask()

    def test_bad_join_key(self):
        x = Data(id1=[0, 1, 2], label='x')
        y = Data(id1=[1, 2, 3], label='y')

        with pytest.raises(ValueError) as exc:
            x.join_on_key(y, 'bad_key', 'id1')
        assert exc.value.args[0] == 'ComponentID not found in x: bad_key'

        with pytest.raises(ValueError) as exc:
            x.join_on_key(y, 'id1', 'bad_key')
        assert exc.value.args[0] == 'ComponentID not found in y: bad_key'

    def test_clone(self):
        x = Data(id=[0, 1, 2])
        y = Data(id=[0, 1, 2], x=[1, 2, 3])
        x.join_on_key(y, 'id', 'id')

        dc = DataCollection([x, y])
        dc = clone(dc)

        x, y = dc
        s = x.new_subset()
        s.subset_state = y.id['x'] > 1

        assert_array_equal(s.to_mask(), [False, True, True])
