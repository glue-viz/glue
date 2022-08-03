from glue.plugins.join_on_key.link_helpers import Join_Link
from glue.core.data import Data
from glue.core.data_collection import DataCollection
from numpy.testing import assert_array_equal
import pytest
from glue.core.exceptions import IncompatibleAttribute


def test_remove_and_add_again():
    d1 = Data(x=[1, 2, 3, 4, 5], k1=[0, 0, 1, 1, 2], label='d1')
    d2 = Data(y=[2, 4, 5, 8, 4], k2=[1, 3, 1, 2, 3], label='d2')
    dc = DataCollection([d1, d2])

    mylink = Join_Link(cids1=[d1.id['k1']], cids2=[d2.id['k2']], data1=d1, data2=d2)
    dc.add_link(mylink)

    dc.remove_link(mylink)
    s = d1.new_subset()
    s.subset_state = d1.id['x'] > 2
    assert_array_equal(s.to_mask(), [False, False, True, True, True])
    s = d2.new_subset()
    s.subset_state = d1.id['x'] > 2
    with pytest.raises(IncompatibleAttribute):
        assert_array_equal(s.to_mask(), [True, False, True, True, False])
    mylink = Join_Link(cids1=[d1.id['k1']], cids2=[d2.id['k2']], data1=d1, data2=d2)
    dc.add_link(mylink)
    s = d2.new_subset()
    s.subset_state = d1.id['x'] > 2
    assert_array_equal(s.to_mask(), [True, False, True, True, False])


def test_remove_is_clean():
    d1 = Data(x=[1, 2, 3, 4, 5], k1=[0, 0, 1, 1, 2], label='d1')
    d2 = Data(y=[2, 4, 5, 8, 4], k2=[1, 3, 1, 2, 3], label='d2')
    dc = DataCollection([d1, d2])

    mylink = Join_Link(cids1=[d1.id['k1']], cids2=[d2.id['k2']], data1=d1, data2=d2)
    dc.add_link(mylink)

    dc.remove_link(mylink)
    s = d1.new_subset()
    s.subset_state = d1.id['x'] > 2
    assert_array_equal(s.to_mask(), [False, False, True, True, True])
    s = d2.new_subset()
    s.subset_state = d1.id['x'] > 2
    with pytest.raises(IncompatibleAttribute):
        assert_array_equal(s.to_mask(), [True, False, True, True, False])


def test_remove():
    d1 = Data(x=[1, 2, 3, 4, 5], k1=[0, 0, 1, 1, 2], label='d1')
    d2 = Data(y=[2, 4, 5, 8, 4], k2=[1, 3, 1, 2, 3], label='d2')
    dc = DataCollection([d1, d2])

    assert len(dc.links) == 0
    assert d1._key_joins == {}
    assert d2._key_joins == {}

    mylink = Join_Link(cids1=[d1.id['k1']], cids2=[d2.id['k2']], data1=d1, data2=d2)
    dc.add_link(mylink)
    assert len(dc.links) == 1
    dc.remove_link(mylink)
    assert len(dc.links) == 0
    assert d1._key_joins == {}
    assert d2._key_joins == {}


def test_using_link_index():
    d1 = Data(x=[1, 2, 3, 4, 5], k1=[0, 0, 1, 1, 2], label='d1')
    d2 = Data(y=[2, 4, 5, 8, 4], k2=[1, 3, 1, 2, 3], label='d2')
    dc = DataCollection([d1, d2])

    assert len(dc.links) == 0
    dc.add_link(Join_Link(cids1=[d1.id['k1']], cids2=[d2.id['k2']], data1=d1, data2=d2))
    assert len(dc.links) == 1

    s = d1.new_subset()
    s.subset_state = d1.id['x'] > 2
    assert_array_equal(s.to_mask(), [False, False, True, True, True])
    s = d2.new_subset()
    s.subset_state = d1.id['x'] > 2
    assert_array_equal(s.to_mask(), [True, False, True, True, False])


def test_basic_join_on_key():
    d1 = Data(x=[1, 2, 3, 4, 5], k1=[0, 0, 1, 1, 2], label='d1')
    d2 = Data(y=[2, 4, 5, 8, 4], k2=[1, 3, 1, 2, 3], label='d2')
    d2.join_on_key(d1, 'k2', 'k1')

    s = d1.new_subset()
    s.subset_state = d1.id['x'] > 2
    assert_array_equal(s.to_mask(), [False, False, True, True, True])
    s = d2.new_subset()
    s.subset_state = d1.id['x'] > 2
    assert_array_equal(s.to_mask(), [True, False, True, True, False])
