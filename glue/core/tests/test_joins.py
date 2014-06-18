from numpy.testing import assert_array_equal
import pytest
import numpy as np

from .. import Data
from ..component_link import ComponentLink
from ..exceptions import IncompatibleAttribute
from .test_state import clone


def test_basic_join():

    x = Data(yid=[0, 1, 2])
    y = Data(z=[3, 4, 5])

    x.join_by_lookup(y, x.id['yid'])
    assert_array_equal(x['z'], [3, 4, 5])


def test_2d_src():

    x = Data(yid=[[0, 1], [1, 0]])
    y = Data(z=[3, 4, 5])

    x.join_by_lookup(y, x.id['yid'])
    assert_array_equal(x['z'], [[3, 4], [4, 3]])


def test_2d_dest():
    x = Data(yid=[0, 1, 2, 3])
    y = Data(z=[[0, 1], [2, 3]])

    x.join_by_lookup(y, x.id['yid'])
    assert_array_equal(x['z'], [0, 1, 2, 3])


def test_join_by_name():

    x = Data(yid=[0, 1, 2])
    y = Data(z=[3, 4, 5])

    x.join_by_lookup(y, 'yid')
    assert_array_equal(x['z'], [3, 4, 5])


def test_lookup_by_id():

    x = Data(yid=[0, 1, 2])
    y = Data(z=[3, 4, 5])

    x.join_by_lookup(y, x.id['yid'])
    assert_array_equal(x[y.id['z']], [3, 4, 5])


def test_views():
    x = Data(yid=[0, 1, 2])
    y = Data(z=[3, 4, 5])

    x.join_by_lookup(y, x.id['yid'])

    assert_array_equal(x['z', ::2], [3, 5])


def test_bad_component():
    x = Data(yid=[0, 1, 2])
    y = Data(z=[3, 4, 5])

    with pytest.raises(ValueError):
        x.join_by_lookup(y, 'Does not exist')

    with pytest.raises(ValueError):
        x.join_by_lookup(y, y.id['z'])


def test_error_on_cid_conflict():

    x = Data(x=[1, 2, 3], yid=[0, 1, 0])
    y = Data()
    y.add_component([4, 5, 6], x.id['x'])

    with pytest.raises(ValueError) as exc:
        x.join_by_lookup(y, 'yid')
    assert exc.value.args[0] == 'Cannot join data. Component conflict: x'


def test_error_on_link_conflict():
    x = Data(z=[10, 20, 30], yid=[0, 1, 0])
    y = Data(z=[1, 2, 3])

    link = ComponentLink([x.id['z']], y.id['z'])
    x.add_component_link(link)

    with pytest.raises(ValueError) as exc:
        x.join_by_lookup(y, 'yid')
    assert exc.value.args[0] == 'Cannot join data. Component conflict: z'


def test_infinite_loop():

    x = Data(yid=[0, 1, 2])
    y = Data(x=[1, 2, 3], xid=[0, 1, 2])

    x.join_by_lookup(y, 'yid')
    y.join_by_lookup(x, 'xid')

    assert_array_equal(x['x'], [1, 2, 3])


def test_recursion():

    x = Data(yid=[0, 1, 2])
    y = Data(xid=[0, 1, 2])

    x.join_by_lookup(y, 'yid')
    y.join_by_lookup(x, 'xid')

    with pytest.raises(IncompatibleAttribute):
        x['x']


def test_multistep():

    x = Data(yid=[0, 0, 1])
    y = Data(zid=[2, 1, 0])
    z = Data(z=[1, 2, 3])

    x.join_by_lookup(y, 'yid')
    y.join_by_lookup(z, 'zid')

    assert_array_equal(x['z'], [3, 3, 2])
    assert_array_equal(y['z'], [3, 2, 1])


def test_multistep_recursion():

    x = Data(yid=[0, 0, 1])
    y = Data(zid=[2, 1, 0])
    z = Data(z=[1, 2, 3])

    x.join_by_lookup(y, 'yid')
    y.join_by_lookup(z, 'zid')

    with pytest.raises(IncompatibleAttribute):
        x['w']


def test_missing():
    x = Data(yid=[0, 1, -2])
    y = Data(z=[1, 2, 3])
    x.join_by_lookup(y, 'yid', missing=-2)

    assert_array_equal(x['z'], [1, 2, np.nan])


def test_restore():
    x = Data(yid=[0, 1, -2])
    y = Data(z=[1, 2, 3])
    x.join_by_lookup(y, 'yid', missing=-2)

    x2 = clone(x)

    assert_array_equal(x2['z'], [1, 2, np.nan])


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
