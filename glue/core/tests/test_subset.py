#pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103
import tempfile
import operator as op

import pytest
import numpy as np
from mock import MagicMock

from ..data import Data, ComponentID, Component
from ..subset import Subset, SubsetState, ElementSubsetState
from ..subset import OrState
from ..subset import AndState
from ..subset import XorState
from ..subset import InvertState
from ..message import SubsetDeleteMessage
from ..registry import Registry


class TestSubset(object):

    def setup_method(self, method):
        self.data = MagicMock()
        self.data.label = "data"
        Registry().clear()

    def test_subset_mask_wraps_state(self):
        s = Subset(self.data)
        state = MagicMock(spec=SubsetState)
        s.subset_state = state
        s.to_mask()
        state.to_mask.assert_called_once_with(None)

    def test_subset_index_wraps_state(self):
        s = Subset(self.data)
        state = MagicMock(spec=SubsetState)
        s.subset_state = state
        s.to_index_list()
        state.to_index_list.assert_called_once_with()

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

    def test_subset_state_reparented_on_assignment(self):
        s = Subset(self.data)
        state = SubsetState()
        s.subset_state = state
        assert state.parent is s

    def test_paste_returns_copy_of_state(self):
        s = Subset(self.data)
        state1 = MagicMock(spec=SubsetState)
        state1_copy = MagicMock()
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
        s.broadcast()
        assert s.data.hub.broadcast.call_count == 0

    def test_broadcast_processed(self):
        """subset broadcasts after do_broadcast(True)"""
        s = Subset(self.data)
        s.do_broadcast(True)
        s.broadcast()
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


def test_subset_combinations_reparent():
    """ parent property of nested subset states assigned correctly """
    s = (Subset(None) & Subset(None)) | Subset(None)
    assert s.subset_state.parent is s
    assert s.subset_state.state1.parent is s
    assert s.subset_state.state1.state1.parent is s
    assert s.subset_state.state1.state2.parent is s
    assert s.subset_state.state2.parent is s


def test_inequality_subsets_reparent():
    """ hierarchical subsets using InequalityStates reparent properly """
    s = Subset(None) & ((ComponentID("") > 5) | (ComponentID("") < 2))
    assert s.subset_state.parent is s
    assert s.subset_state.state1.parent is s
    assert s.subset_state.state2.parent is s
    assert s.subset_state.state2.state1.parent is s
    assert s.subset_state.state2.state2.parent is s


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

        def to_mask(self, view):
            return self._mask

        def copy(self):
            return TestCompositeSubsetStates.DummyState(self._mask)

    def setup_method(self, method):
        self.sub1 = self.DummyState(np.array([1, 1, 0, 0], dtype='bool'))
        self.sub2 = self.DummyState(np.array([1, 0, 1, 0], dtype='bool'))

    def test_or(self):
        s3 = OrState(self.sub1, self.sub2)
        answer = s3.to_mask()
        expected = np.array([True, True, True, False])
        np.testing.assert_array_equal(answer, expected)

    def test_and(self):
        s3 = AndState(self.sub1, self.sub2)
        answer = s3.to_mask()
        expected = np.array([True, False, False, False])
        np.testing.assert_array_equal(answer, expected)

    def test_xor(self):
        s3 = XorState(self.sub1, self.sub2)
        answer = s3.to_mask()
        expected = np.array([False, True, True, False])
        np.testing.assert_array_equal(answer, expected)

    def test_invert(self):
        s3 = InvertState(self.sub1)
        answer = s3.to_mask()
        expected = np.array([False, False, True, True])
        np.testing.assert_array_equal(answer, expected)

    def test_multicomposite(self):
        s3 = AndState(self.sub1, self.sub2)
        s4 = XorState(s3, self.sub1)
        answer = s4.to_mask()
        expected = np.array([False, True, False, False])
        np.testing.assert_array_equal(answer, expected)


class TestElementSubsetState(object):

    def setup_method(self, method):
        self.state = ElementSubsetState()
        self.state.parent = MagicMock()
        self.state.parent.data.shape = (2, 1)

    def test_empty_mask(self):
        mask = self.state.to_mask()
        np.testing.assert_array_equal(mask, np.array([[False], [False]]))

    def test_empty_index_list(self):
        ilist = self.state.to_index_list()
        np.testing.assert_array_equal(ilist, np.array([]))

    def test_nonempty_index_list(self):
        self.state._indices = [0]
        ilist = self.state.to_index_list()
        np.testing.assert_array_equal(ilist, np.array([0]))

    def test_nonempty_mask(self):
        self.state._indices = [0]
        mask = self.state.to_mask()
        np.testing.assert_array_equal(mask, np.array([[True], [False]]))

    def test_define_on_init(self):
        ind = np.array([0, 1])
        state = ElementSubsetState(indices=ind)
        np.testing.assert_array_equal(ind, state._indices)


class TestSubsetIo(object):

    def setup_method(self, method):
        self.data = MagicMock()
        self.data.shape = (4, 4)
        self.subset = Subset(self.data)
        inds = np.array([1, 2, 3])
        self.subset.subset_state = ElementSubsetState(indices=inds)

    def test_write(self):
        with tempfile.NamedTemporaryFile() as tmp:
            self.subset.write_mask(tmp.name)
            from astropy.io import fits
            data = fits.open(tmp.name)[0].data
            expected = np.array([[0, 1, 1, 1],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0]], dtype=np.int16)
            np.testing.assert_array_equal(data, expected)

    def test_read(self):
        with tempfile.NamedTemporaryFile() as tmp:
            self.subset.write_mask(tmp.name)
            sub2 = Subset(self.data)
            sub2.read_mask(tmp.name)
            mask1 = self.subset.to_mask()
            mask2 = sub2.to_mask()
            np.testing.assert_array_equal(mask1, mask2)

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
        self.state.parent = MagicMock()

    def mask_check(self, mask, answer):
        self.state.to_mask = MagicMock()
        self.state.to_mask.return_value = mask
        np.testing.assert_array_equal(self.state.to_index_list(), answer)

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
    def to_mask(self, view=None):
        result = np.ones(self.parent.data.shape, dtype=bool)
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


#Test Fancy Indexing into the various subset states


def roifac(comp, cid):
    from ..roi import RectangularROI
    from ..subset import RoiSubsetState

    result = RoiSubsetState()
    result.xatt = cid
    result.yatt = cid
    roi = RectangularROI()
    roi.update_limits(0.5, 0.5, 1.5, 1.5)
    result.roi = roi
    return result


def rangefac(comp, cid):
    from ..subset import RangeSubsetState
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


views = (np.s_[:],
         np.s_[::-1, 0],
         np.s_[0, :],
         np.s_[:, 0],
         np.array([[True, False], [False, True]]),
         np.where(np.array([[True, False], [False, True]])),
         np.zeros((2, 2), dtype=bool),
         )
facs = [roifac, rangefac, orfac, andfac, xorfac, invertfac,
        elementfac, inequalityfac, basefac]


@pytest.mark.parametrize(('statefac', 'view'), [(f, v) for f in facs
                                                for v in views])
def test_mask_of_view_is_view_of_mask(statefac, view):
    print statefac, view
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
    m0 = np.zeros_like(c.data, dtype=bool)
    m0[view] = True
    v2 = c.data[s.to_mask() & m0]
    np.testing.assert_array_equal(v1, v2)
