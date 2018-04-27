# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function

import tempfile
import operator as op

import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose

from mock import MagicMock

from glue.tests.helpers import requires_astropy, requires_scipy, SCIPY_INSTALLED
from ..exceptions import IncompatibleAttribute
from .. import DataCollection, ComponentLink
from ..data import Data, Component
from ..roi import CategoricalROI, RectangularROI, Projected3dROI, CircularROI
from ..message import SubsetDeleteMessage
from ..registry import Registry
from ..link_helpers import LinkSame
from ..subset import (Subset, SubsetState,
                      ElementSubsetState, RoiSubsetState, RangeSubsetState,
                      CategoricalROISubsetState, InequalitySubsetState,
                      CategorySubsetState, MaskSubsetState,
                      CategoricalROISubsetState2D, RoiSubsetState3d,
                      CategoricalMultiRangeSubsetState, FloodFillSubsetState,
                      SliceSubsetState)
from ..subset import AndState
from ..subset import InvertState
from ..subset import OrState
from ..subset import XorState
from .test_state import clone


class TestSubset(object):

    def setup_method(self, method):
        self.data = MagicMock(spec=Data)
        self.data.hub = MagicMock()
        self.data.label = "data"
        Registry().clear()

    def test_subset_mask_wraps_state(self):
        s = Subset(self.data)
        state = MagicMock(spec_set=SubsetState)
        state.to_mask.return_value = np.array([True])
        assert state.to_mask.call_count == 0
        s.subset_state = state
        s.to_mask()
        state.to_mask.assert_called_once_with(self.data, None)

    def test_subset_index_wraps_state(self):
        s = Subset(self.data)
        state = MagicMock(spec=SubsetState)
        state.to_index_list.return_value = np.array([1, 2, 3])
        s.subset_state = state
        s.to_index_list()
        state.to_index_list.assert_called_once_with(self.data)

    def test_set_label(self):
        s = Subset(self.data, label='hi')
        assert s.label == 'hi'

    def test_str(self):
        s = Subset(self.data, label="hi")
        assert str(s) == "Subset: hi (data: data)"
        s = Subset(None, label="hi")
        assert str(s) == "Subset: hi (no data)"
        s = Subset(None)
        assert str(s) == "Subset: (no label) (no data)"
        s = Subset(self.data)
        assert str(s) == "Subset: (no label) (data: data)"

    def test_set_color(self):
        s = Subset(self.data, color='blue')
        assert s.style.color == 'blue'

    def test_paste_returns_copy_of_state(self):
        s = Subset(self.data)
        state1 = MagicMock(spec=SubsetState)
        state1_copy = MagicMock(spec=SubsetState)
        state1.copy.return_value = state1_copy
        s.subset_state = state1

        s2 = Subset(self.data)

        s2.paste(s)
        assert s2.subset_state is state1_copy

    def test_register_enables_braodcasting(self):
        s = Subset(self.data)
        s.register()
        assert s._broadcasting

    def test_register_adds_subset_to_data(self):
        s = Subset(self.data)
        s.register()
        s.data.add_subset.assert_called_once_with(s)

    def test_delete_without_hub(self):
        self.data.hub = None
        s = Subset(self.data)
        s.register()
        s.delete()
        assert not s._broadcasting

    def test_delete_disables_broadcasting(self):
        """Subset no longer broadcasts after delete"""
        s = Subset(self.data)
        s.register()
        s.delete()
        assert not s._broadcasting

    def test_delete_sends_message_if_hub_present(self):
        """delete() broadcasts a SubsetDelteMessage"""
        s = Subset(self.data)
        s.register()
        s.delete()
        assert s.data.hub.broadcast.call_count == 1
        args = s.data.hub.broadcast.call_args[0]
        msg = args[0]
        assert isinstance(msg, SubsetDeleteMessage)

    def test_delete_removes_from_data(self):
        """delete method removes reference from data.subsets"""
        data = Data()
        s = data.new_subset()
        assert s in data.subsets
        s.delete()
        assert s not in data.subsets

    def test_delete_with_no_data(self):
        """delete method doesnt crash if subset has no data"""
        s = Subset(None)
        assert s.data is None
        s.delete()

    def test_double_delete_ignored(self):
        """calling delete twice doesnt crash"""
        data = Data()
        s = data.new_subset()
        assert s in data.subsets
        s.delete()
        s.delete()
        assert s not in data.subsets

    def test_broadcast_ignore(self):
        """subset doesn't broadcast until do_broadcast(True)"""
        s = Subset(self.data)
        s.broadcast('style')
        assert s.data.hub.broadcast.call_count == 0

    def test_broadcast_processed(self):
        """subset broadcasts after do_broadcast(True)"""
        s = Subset(self.data)
        s.do_broadcast(True)
        s.broadcast('style')
        assert s.data.hub.broadcast.call_count == 1

    def test_del(self):
        s = Subset(self.data)
        s.__del__()

    def test_getitem_empty(self):
        s = Subset(self.data)
        s.to_index_list = MagicMock()
        s.to_index_list.return_value = []
        get = s['test']
        assert list(get) == []

    def test_state_with_array(self):
        d = Data(x=[1, 2, 3])
        s = d.new_subset()
        s.subset_state = np.array([True, False, False])
        np.testing.assert_array_equal(s.to_mask(), [True, False, False])

    def test_state_array_bad_shape(self):
        d = Data(x=[1, 2, 3])
        s = d.new_subset()
        with pytest.raises(ValueError):
            s.subset_state = np.array([True])

    def test_state_bad_type(self):
        s = Subset(Data())
        with pytest.raises(TypeError):
            s.subset_state = 5


target_states = ((op.and_, AndState),
                 (op.or_, OrState),
                 (op.xor, XorState))


@pytest.mark.parametrize(("x"), target_states)
def test_binary_subset_combination(x):
    operator, target = x
    s1 = Subset(None)
    s2 = Subset(None)
    newsub = operator(s1, s2)
    assert isinstance(newsub, Subset)
    assert isinstance(newsub.subset_state, target)


class TestSubsetStateCombinations(object):

    def setup_method(self, method):
        self.data = None

    def test_or(self):
        s1 = Subset(self.data)
        s2 = Subset(self.data)
        s3 = s1.subset_state | s2.subset_state
        assert isinstance(s3, OrState)

    def test_and(self):
        s1 = Subset(self.data)
        s2 = Subset(self.data)
        s3 = s1.subset_state & s2.subset_state
        assert isinstance(s3, AndState)

    def test_invert(self):
        s1 = Subset(self.data)
        s3 = ~s1.subset_state
        assert isinstance(s3, InvertState)

    def test_xor(self):
        s1 = Subset(self.data)
        s2 = Subset(self.data)
        s3 = s1.subset_state ^ s2.subset_state
        assert isinstance(s3, XorState)


class TestCompositeSubsetStates(object):

    class DummyState(SubsetState):

        def __init__(self, mask):
            self._mask = mask

        def to_mask(self, data, view):
            return self._mask

        def copy(self):
            return TestCompositeSubsetStates.DummyState(self._mask)

    def setup_method(self, method):
        self.sub1 = self.DummyState(np.array([1, 1, 0, 0], dtype='bool'))
        self.sub2 = self.DummyState(np.array([1, 0, 1, 0], dtype='bool'))
        self.data = Data(x=[1, 2, 3, 4])

    def test_or(self):
        s3 = OrState(self.sub1, self.sub2)
        answer = s3.to_mask(self.data)
        expected = np.array([True, True, True, False])
        np.testing.assert_array_equal(answer, expected)

    def test_and(self):
        s3 = AndState(self.sub1, self.sub2)
        answer = s3.to_mask(self.data)
        expected = np.array([True, False, False, False])
        np.testing.assert_array_equal(answer, expected)

    def test_xor(self):
        s3 = XorState(self.sub1, self.sub2)
        answer = s3.to_mask(self.data)
        expected = np.array([False, True, True, False])
        np.testing.assert_array_equal(answer, expected)

    def test_invert(self):
        s3 = InvertState(self.sub1)
        answer = s3.to_mask(self.data)
        expected = np.array([False, False, True, True])
        np.testing.assert_array_equal(answer, expected)

    def test_multicomposite(self):
        s3 = AndState(self.sub1, self.sub2)
        s4 = XorState(s3, self.sub1)
        answer = s4.to_mask(self.data)
        expected = np.array([False, True, False, False])
        np.testing.assert_array_equal(answer, expected)


class TestElementSubsetState(object):

    def setup_method(self, method):
        self.state = ElementSubsetState()
        self.data = Data(x=[[1], [2]])

    def test_empty_mask(self):
        mask = self.state.to_mask(self.data)
        np.testing.assert_array_equal(mask, np.array([[False], [False]]))

    def test_empty_index_list(self):
        ilist = self.state.to_index_list(self.data)
        np.testing.assert_array_equal(ilist, np.array([]))

    def test_nonempty_index_list(self):
        self.state._indices = [0]
        ilist = self.state.to_index_list(self.data)
        np.testing.assert_array_equal(ilist, np.array([0]))

    def test_nonempty_mask(self):
        self.state._indices = [0]
        mask = self.state.to_mask(self.data)
        np.testing.assert_array_equal(mask, np.array([[True], [False]]))

    def test_define_on_init(self):
        ind = np.array([0, 1])
        state = ElementSubsetState(indices=ind)
        np.testing.assert_array_equal(ind, state._indices)


class TestSubsetIo(object):

    def setup_method(self, method):
        self.data = MagicMock(spec=Data)
        self.data.shape = (4, 4)
        self.data.uuid = 'abcde'
        self.subset = Subset(self.data)
        inds = np.array([1, 2, 3])
        self.subset.subset_state = ElementSubsetState(indices=inds)

    @requires_astropy
    def test_write(self):
        fobj, tmp = tempfile.mkstemp()

        self.subset.write_mask(tmp)
        from astropy.io import fits
        with fits.open(tmp) as hdulist:
            data = hdulist[0].data
        expected = np.array([[0, 1, 1, 1],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]], dtype=np.int16)
        np.testing.assert_array_equal(data, expected)

    @requires_astropy
    def test_read(self):
        fobj, tmp = tempfile.mkstemp()

        self.subset.write_mask(tmp)
        sub2 = Subset(self.data)
        sub2.read_mask(tmp)
        mask1 = self.subset.to_mask()
        mask2 = sub2.to_mask()
        np.testing.assert_array_equal(mask1, mask2)

    @requires_astropy
    def test_read_error(self):
        with pytest.raises(IOError) as exc:
            self.subset.read_mask('file_does_not_exist')
        assert exc.value.args[0] == ("Could not read file_does_not_exist "
                                     "(not a fits file?)")

    def test_write_unsupported_format(self):
        with pytest.raises(AttributeError) as exc:
            self.subset.write_mask('file_will_fail', format='.hd5')
        assert exc.value.args[0] == "format not supported: .hd5"


class TestSubsetState(object):

    def setup_method(self, method):
        self.state = SubsetState()

    def mask_check(self, mask, answer):
        self.state.to_mask = MagicMock()
        self.state.to_mask.return_value = mask
        np.testing.assert_array_equal(self.state.to_index_list(Data()), answer)

    def test_to_index_list_1d(self):
        mask = np.array([False, True])
        answer = np.array([1])
        self.mask_check(mask, answer)

    def test_to_index_list_2d(self):
        mask = np.array([[False, True], [False, True]])
        answer = np.array([1, 3])
        self.mask_check(mask, answer)

    def test_empty_to_index_1d(self):
        mask = np.array([False, False])
        answer = np.array([])
        self.mask_check(mask, answer)

    def test_empty_to_index_2d(self):
        mask = np.array([[False, False], [False, False]])
        answer = np.array([])
        self.mask_check(mask, answer)


class TestCompositeSubsetStateCopy(object):

    def assert_composite_copy(self, cls):
        """Copying composite state should create new
        state with same type, with copies of sub states"""
        state1 = MagicMock()
        state2 = MagicMock()
        s1 = cls(state1, state2)
        s2 = s1.copy()

        assert type(s1) == type(s2)
        assert s1.state1.copy() is s2.state1
        assert s1.state2.copy() is s2.state2

    def test_invert(self):
        self.assert_composite_copy(InvertState)

    def test_and(self):
        self.assert_composite_copy(AndState)

    def test_or(self):
        self.assert_composite_copy(OrState)

    def test_xor(self):
        self.assert_composite_copy(XorState)


class DummySubsetState(SubsetState):

    def to_mask(self, data, view=None):
        result = np.ones(data.shape, dtype=bool)
        if view is not None:
            result = result[view]
        return result


class TestSubsetViews(object):

    def setup_method(self, method):
        d = Data()
        c = Component(np.array([1, 2, 3, 4]))
        self.cid = d.add_component(c, 'test')
        self.s = d.new_subset()
        self.c = c
        self.s.subset_state = DummySubsetState()

    def test_cid_get(self):
        np.testing.assert_array_equal(self.s[self.cid],
                                      self.c.data)

    def test_label_get(self):
        np.testing.assert_array_equal(self.s['test'],
                                      self.c.data)

    def test_cid_slice(self):
        np.testing.assert_array_equal(self.s[self.cid, ::2],
                                      self.c.data[::2])

    def test_label_slice(self):
        np.testing.assert_array_equal(self.s['test', ::-1],
                                      self.c.data[::-1])


# Test Fancy Indexing into the various subset states


def roifac(comp, cid):

    result = RoiSubsetState()
    result.xatt = cid
    result.yatt = cid
    roi = RectangularROI()
    roi.update_limits(0.5, 0.5, 1.5, 1.5)
    result.roi = roi
    return result


def roifac3d(comp, cid):

    roi_2d = RectangularROI(0.5, 0.5, 1.5, 1.5)
    roi_3d = Projected3dROI(roi_2d=roi_2d, projection_matrix=np.arange(16).reshape((4, 4)))

    result = RoiSubsetState3d(xatt=cid, yatt=cid, zatt=cid, roi=roi_3d)

    return result


def rangefac(comp, cid):
    return RangeSubsetState(.5, 2.5, att=cid)


def compfac(comp, cid, oper):
    s1 = roifac(comp, cid)
    s2 = rangefac(comp, cid)
    return oper(s1, s2)


def orfac(comp, cid):
    return compfac(comp, cid, op.or_)


def andfac(comp, cid):
    return compfac(comp, cid, op.and_)


def xorfac(comp, cid):
    return compfac(comp, cid, op.xor)


def invertfac(comp, cid):
    return ~rangefac(comp, cid)


def elementfac(comp, cid):
    return ElementSubsetState(np.array([0, 1]))


def inequalityfac(comp, cid):
    return cid > 2.5


def basefac(comp, cid):
    return SubsetState()


def maskfac(comp, cid):
    return MaskSubsetState([[0, 1], [1, 1]], cid.parent.pixel_component_ids)


def floodfac(comp, cid):
    return FloodFillSubsetState(cid.parent, cid, [0, 0], 1.2)


def slicefac(comp, cid):
    return SliceSubsetState(cid.parent, [slice(None), slice(0, 1)])


views = (np.s_[:],
         np.s_[::-1, 0],
         np.s_[0, :],
         np.s_[:, 0],
         np.array([[True, False], [False, True]]),
         np.where(np.array([[True, False], [False, True]])),
         np.zeros((2, 2), dtype=bool),
         )
facs = [roifac, roifac3d, rangefac, orfac, andfac, xorfac, invertfac,
        elementfac, inequalityfac, basefac, maskfac, slicefac]

if SCIPY_INSTALLED:
    facs.append(floodfac)


@pytest.mark.parametrize(('statefac', 'view'), [(f, v) for f in facs
                                                for v in views])
def test_mask_of_view_is_view_of_mask(statefac, view):
    print(statefac, view)
    d = Data()
    d.edit_subset = d.new_subset()
    c = Component(np.array([[1, 2], [3, 4]]))
    cid = d.add_component(c, 'test')
    s = d.edit_subset
    s.subset_state = statefac(c, cid)

    v1 = s.to_mask(view)
    v2 = s.to_mask()[view]
    np.testing.assert_array_equal(v1, v2)

    v1 = s[cid, view]
    v2 = c.data[view][s.to_mask(view)]
    np.testing.assert_array_equal(v1, v2)


def test_inequality_state_str():
    d = Data(x=[1, 2, 3], y=[2, 3, 4])
    x = d.id['x']
    y = d.id['y']

    assert str(x == 'a') == '(x == a)'
    assert str(x > 3) == '(x > 3)'
    assert str(x < 2) == '(x < 2)'
    assert str(x < y) == '(x < y)'
    assert str((3 * x) < 5) == '((3 * x) < 5)'
    assert str((x < y) & (x < 2)) == '((x < y) & (x < 2))'
    assert str((x < y) | (x < 2)) == '((x < y) | (x < 2))'
    assert str(~(x < y)) == '(~(x < y))'
    assert repr(x < 5) == ('<InequalitySubsetState: (x < 5)>')


def test_to_mask_state():

    d = Data(x=[1, 2, 3])
    sub = d.new_subset()
    sub.subset_state = d.id['x'] > 1
    sub.subset_state = sub.state_as_mask()

    np.testing.assert_array_equal(sub.to_mask(), [False, True, True])


def test_to_mask_state_across_data():

    d = Data(x=[1, 2, 3])
    d2 = Data(x=[2, 3, 4])
    dc = DataCollection([d, d2])

    link = ComponentLink(d2.pixel_component_ids,
                         d.pixel_component_ids[0],
                         lambda x: x - 1)
    dc.add_link(link)

    sub = d.new_subset()
    sub.subset_state = d.id['x'] > 1
    sub.subset_state = sub.state_as_mask()

    sub2 = d2.new_subset()
    sub2.subset_state = sub.subset_state
    np.testing.assert_array_equal(sub2.to_mask(), [False, False, True])


def test_mask_clone():

    d = Data(x=[1, 2, 3])
    sub = d.new_subset()
    sub.subset_state = d.id['x'] > 1
    sub.subset_state = sub.state_as_mask()

    d = clone(d)
    sub = d.subsets[0]
    np.testing.assert_array_equal(sub.to_mask(), [False, True, True])


class TestAttributes(object):

    def test_empty(self):
        assert SubsetState().attributes == tuple()

    def test_roi(self):
        d = Data(x=[1], y=[2])
        s = RoiSubsetState(xatt=d.id['x'], yatt=d.id['y'])
        assert s.attributes == (d.id['x'], d.id['y'])

    def test_range(self):
        d = Data(x=[1])
        s = RangeSubsetState(0, 1, att=d.id['x'])
        assert s.attributes == (d.id['x'],)

    def test_composite(self):
        d = Data(x=[1])
        s = RangeSubsetState(0, 1, att=d.id['x'])
        assert (s & s).attributes == (d.id['x'],)

    def test_not(self):
        d = Data(x=[1])
        s = RangeSubsetState(0, 1, att=d.id['x'])
        assert (~s).attributes == (d.id['x'],)

    def test_subset(self):
        d = Data(x=[1])
        s = d.new_subset()
        s.subset_state = RangeSubsetState(0, 1, att=d.id['x'])
        assert s.attributes == (d.id['x'],)


def test_save_element_subset_state():
    # Regression test to make sure that element subset states are saved
    # correctly.
    state1 = ElementSubsetState(indices=[1, 3, 4])
    state2 = clone(state1)
    assert state2._indices == [1, 3, 4]


def test_inequality_subset_state_string():
    d = Data(x=['a', 'b', 'c', 'b'])
    state = d.id['x'] == 'b'
    np.testing.assert_equal(state.to_mask(d), np.array([False, True, False, True]))


def test_inherited_properties():

    d = Data(x=np.random.random((3, 2, 4)).astype(np.float32))
    sub = d.new_subset()
    sub.subset_state = d.id['x'] > 0.5

    assert sub.component_ids() == d.component_ids()
    assert sub.components == d.components
    assert sub.derived_components == d.derived_components
    assert sub.primary_components == d.primary_components
    assert sub.visible_components == d.visible_components
    assert sub.pixel_component_ids == d.pixel_component_ids
    assert sub.world_component_ids == d.world_component_ids

    assert sub.ndim == 3
    assert sub.shape == (3, 2, 4)

    assert sub.hub is d.hub


class TestCloneSubsetStates():

    def setup_method(self, method):
        self.data = Data(a=[-3, 2, 4, 1],
                         b=['a', 'b', 'a', 'c'],
                         c=[1.2, 1.3, 1.5, 1.9],
                         d=['x', 'y', 'z', 'y'])

    def test_element_subset_state(self):

        subset = self.data.new_subset()
        subset.subset_state = ElementSubsetState(indices=[1, 2])
        assert_equal(self.data.subsets[0].to_mask(), [0, 1, 1, 0])

        data_clone = clone(self.data)

        assert_equal(data_clone.subsets[0].to_mask(), [0, 1, 1, 0])

    def test_categorical_roi_subset_state(self):

        roi = CategoricalROI(['a', 'c'])

        subset = self.data.new_subset()
        subset.subset_state = CategoricalROISubsetState(att=self.data.id['b'], roi=roi)
        assert_equal(self.data.subsets[0].to_mask(), [1, 0, 1, 1])

        data_clone = clone(self.data)

        assert_equal(data_clone.subsets[0].to_mask(), [1, 0, 1, 1])

    def test_categorical_roi_2d_subset_state(self):

        selection = {'a': ['x'], 'b': ['x'], 'c': ['y']}

        subset = self.data.new_subset()
        subset.subset_state = CategoricalROISubsetState2D(selection, self.data.id['b'], self.data.id['d'])
        assert_equal(self.data.subsets[0].to_mask(), [1, 0, 0, 1])

        data_clone = clone(self.data)

        assert_equal(data_clone.subsets[0].to_mask(), [1, 0, 0, 1])

    def test_category_subset_state(self):

        subset = self.data.new_subset()
        subset.subset_state = CategorySubsetState(self.data.id['b'], [0, 2])
        assert_equal(self.data.subsets[0].to_mask(), [1, 0, 1, 1])

        data_clone = clone(self.data)

        assert_equal(data_clone.subsets[0].to_mask(), [1, 0, 1, 1])

    def test_category_multi_range_subset_state(self):

        ranges = {'a': [(1.0, 1.1), (1.3, 1.6)], 'b': [(1.1, 1.4), (1.7, 1.8)], 'c': [(1.1, 1.2)]}

        subset = self.data.new_subset()
        subset.subset_state = CategoricalMultiRangeSubsetState(ranges, self.data.id['b'], self.data.id['c'])
        assert_equal(self.data.subsets[0].to_mask(), [0, 1, 1, 0])

        data_clone = clone(self.data)

        assert_equal(data_clone.subsets[0].to_mask(), [0, 1, 1, 0])

    def test_inequality_roi_subset_state(self):

        subset = self.data.new_subset()
        subset.subset_state = self.data.id['a'] > 1.5
        assert isinstance(subset.subset_state, InequalitySubsetState)

        assert_equal(self.data.subsets[0].to_mask(), [0, 1, 1, 0])

        data_clone = clone(self.data)

        assert_equal(data_clone.subsets[0].to_mask(), [0, 1, 1, 0])

    def test_mask_subset_state(self):

        subset = self.data.new_subset()
        subset.subset_state = MaskSubsetState([0, 1, 0, 1], self.data.pixel_component_ids)

        assert_equal(self.data.subsets[0].to_mask(), [0, 1, 0, 1])

        data_clone = clone(self.data)

        assert_equal(data_clone.subsets[0].to_mask(), [0, 1, 0, 1])

    def test_range_subset_state(self):

        subset = self.data.new_subset()
        subset.subset_state = RangeSubsetState(1.1, 1.4, self.data.id['c'])
        assert_equal(self.data.subsets[0].to_mask(), [1, 1, 0, 0])

        data_clone = clone(self.data)

        assert_equal(data_clone.subsets[0].to_mask(), [1, 1, 0, 0])

    def test_and_subset_state(self):

        subset = self.data.new_subset()
        subset.subset_state = (self.data.id['a'] > 1) & (self.data.id['c'] < 1.5)
        assert isinstance(subset.subset_state, AndState)
        assert_equal(self.data.subsets[0].to_mask(), [0, 1, 0, 0])

        data_clone = clone(self.data)

        assert_equal(data_clone.subsets[0].to_mask(), [0, 1, 0, 0])

    def test_or_subset_state(self):

        subset = self.data.new_subset()
        subset.subset_state = (self.data.id['a'] > 1) | (self.data.id['c'] < 1.5)
        assert isinstance(subset.subset_state, OrState)
        assert_equal(self.data.subsets[0].to_mask(), [1, 1, 1, 0])

        data_clone = clone(self.data)

        assert_equal(data_clone.subsets[0].to_mask(), [1, 1, 1, 0])

    def test_not_subset_state(self):

        subset = self.data.new_subset()
        subset.subset_state = ~(self.data.id['a'] > 1)
        assert isinstance(subset.subset_state, InvertState)
        assert_equal(self.data.subsets[0].to_mask(), [1, 0, 0, 1])

        data_clone = clone(self.data)

        assert_equal(data_clone.subsets[0].to_mask(), [1, 0, 0, 1])

    def test_xor_subset_state(self):

        subset = self.data.new_subset()
        subset.subset_state = (self.data.id['a'] > 1) ^ (self.data.id['c'] > 1.3)
        assert isinstance(subset.subset_state, XorState)
        assert_equal(self.data.subsets[0].to_mask(), [0, 1, 0, 1])

        data_clone = clone(self.data)

        assert_equal(data_clone.subsets[0].to_mask(), [0, 1, 0, 1])

    def test_roi_subset_state(self):

        roi = RectangularROI(xmin=0, xmax=3, ymin=1.1, ymax=1.4)

        subset = self.data.new_subset()
        subset.subset_state = RoiSubsetState(xatt=self.data.id['a'], yatt=self.data.id['c'], roi=roi)
        assert_equal(self.data.subsets[0].to_mask(), [0, 1, 0, 0])

        data_clone = clone(self.data)

        assert_equal(data_clone.subsets[0].to_mask(), [0, 1, 0, 0])


@requires_scipy
def test_floodfill_subset_state():

    data = np.array([[9, 6, 2, 3],
                     [4, 5, 2, 5],
                     [2, 4, 1, 0],
                     [5, 6, 0, -1]])

    expected = np.array([[0, 0, 0, 0],
                         [1, 1, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 0]])

    data1 = Data(x=data)
    data2 = Data(x=4 - data1['x'])

    dc = DataCollection([data1, data2])

    subset_state = FloodFillSubsetState(data1, data1.id['x'], (1, 0), 1.3)

    dc.new_subset_group(label='subset', subset_state=subset_state)

    result = data1.subsets[0].to_mask()
    assert_equal(result, expected)

    with pytest.raises(IncompatibleAttribute):
        data2.subsets[0].to_mask()

    # Check that setting up pixel links works

    dc.add_link(LinkSame(data1.pixel_component_ids[0], data2.pixel_component_ids[0]))
    dc.add_link(LinkSame(data1.pixel_component_ids[1], data2.pixel_component_ids[1]))

    result = data2.subsets[0].to_mask()
    assert_equal(result, expected)

    dc._link_manager.clear_links()

    dc.add_link(LinkSame(data1.pixel_component_ids[1], data2.pixel_component_ids[0]))
    dc.add_link(LinkSame(data1.pixel_component_ids[0], data2.pixel_component_ids[1]))

    result = data2.subsets[0].to_mask()
    assert_equal(result, expected.T)

    # Check that (de)serialization works

    dc_new = clone(dc)

    result = dc_new[0].subsets[0].to_mask()
    assert_equal(result, expected)

    # Check that changing parameters invalidates the cache

    dc[0].subsets[0].subset_state.threshold = 10
    result = data1.subsets[0].to_mask()
    assert_equal(result, 1)


def test_projected_3d_clone():

    d = Data(x=[1, 2, 3], y=[2, 3, 4], z=[4, 3, 2])
    roi_2d = CircularROI(2, 3, 4)
    projection_matrix = np.random.uniform(-1, 1, 16).reshape((4, 4))
    roi_3d = Projected3dROI(roi_2d=roi_2d, projection_matrix=projection_matrix)

    subset_state = RoiSubsetState3d(d.id['y'], d.id['x'], d.id['z'], roi_3d)

    subset_state_new = clone(subset_state)

    assert subset_state_new.xatt.label == 'y'
    assert subset_state_new.yatt.label == 'x'
    assert subset_state_new.zatt.label == 'z'
    assert isinstance(subset_state_new.roi, Projected3dROI)
    assert_allclose(subset_state_new.roi.projection_matrix, projection_matrix)


def test_slice_subset_state():

    data1 = Data(x=np.arange(24).reshape((2, 3, 4)))

    slices = [slice(None), slice(1, 3), slice(None, None, 2)]
    subset_state = SliceSubsetState(data1, slices)

    expected_mask = np.zeros((2, 3, 4))
    expected_mask[slices] = 1
    assert_equal(subset_state.to_mask(data1), expected_mask)

    view = [slice(0, 1), slice(None), slice(None)]
    assert_equal(subset_state.to_mask(data1, view=view), expected_mask[view])

    data2 = Data(x=np.arange(24).reshape((3, 4, 2)))
    data_collection = DataCollection([data1, data2])

    assert_equal(subset_state.to_mask(data2), 0)

    data_collection.add_link(LinkSame(data1.pixel_component_ids[0], data2.pixel_component_ids[2]))

    assert_equal(subset_state.to_mask(data2), 0)

    data_collection.add_link(LinkSame(data1.pixel_component_ids[1], data2.pixel_component_ids[0]))
    data_collection.add_link(LinkSame(data1.pixel_component_ids[2], data2.pixel_component_ids[1]))

    assert_equal(subset_state.to_mask(data2), expected_mask.transpose().swapaxes(0, 1))

    view = [slice(None), slice(1, 3), slice(None)]
    assert_equal(subset_state.to_mask(data2, view=view), expected_mask.transpose().swapaxes(0, 1)[view])


def test_slice_subset_state_clone():

    data1 = Data(x=np.arange(24).reshape((2, 3, 4)))

    slices = [slice(None), slice(1, 3), slice(None, None, 2)]
    subset_state = SliceSubsetState(data1, slices)
    subset = data1.new_subset()
    subset.subset_state = subset_state

    expected_mask = np.zeros((2, 3, 4))
    expected_mask[slices] = 1
    assert_equal(data1.subsets[0].to_mask(), expected_mask)

    data2 = clone(data1)
    assert_equal(data2.subsets[0].to_mask(), expected_mask)
