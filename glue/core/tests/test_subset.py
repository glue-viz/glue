import tempfile

import pytest
import numpy as np
from mock import MagicMock
import pyfits

from ..subset import Subset, SubsetState, ElementSubsetState
from ..subset import OrState
from ..subset import AndState
from ..subset import XorState
from ..subset import InvertState


class TestSubset(object):

    def setup_method(self, method):
        self.data = MagicMock()

    def test_subset_mask_wraps_state(self):
        s = Subset(self.data)
        state = MagicMock(spec=SubsetState)
        s.subset_state = state
        s.to_mask()
        state.to_mask.assert_called_once_with()

    def test_subset_index_wraps_state(self):
        s = Subset(self.data)
        state = MagicMock(spec=SubsetState)
        s.subset_state = state
        s.to_index_list()
        state.to_index_list.assert_called_once_with()

    def test_set_label(self):
        s = Subset(self.data, label = 'hi')
        assert s.label == 'hi'

    def test_set_color(self):
        s = Subset(self.data, color = 'blue')
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

    def test_unregister_without_hub(self):
        self.data.hub = None
        s = Subset(self.data)
        s.register()
        s.unregister()
        assert not s._broadcasting

    def test_unregister_disables_broadcasting(self):
        s = Subset(self.data)
        s.register()
        s.unregister()
        assert not s._broadcasting

    def test_unregister_sends_message_if_hub_present(self):
        s = Subset(self.data)
        s.register()
        s.unregister()
        assert s.data.hub.broadcast.call_count == 1

    def test_broadcast_ignore(self):
        s = Subset(self.data)
        s.broadcast()
        assert s.data.hub.broadcast.call_count == 0

    def test_broadcast_processed(self):
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

    def setup_method(self, method):
        self.sub1 = MagicMock(spec=SubsetState)
        self.sub1.to_mask.return_value = np.array([1, 1, 0, 0], dtype='bool')
        self.sub2 = MagicMock(spec=SubsetState)
        self.sub2.to_mask.return_value = np.array([1, 0, 1, 0], dtype='bool')

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

class TestElementSubsetState(object):

    def setup_method(self, method):
        self.state = ElementSubsetState()
        self.state.parent = MagicMock()
        self.state.parent.data.shape = (2,1)

    def test_empty_mask(self):
        mask = self.state.to_mask()
        np.testing.assert_array_equal(mask, np.array([[False], [False]]))

    def test_empty_index_list(self):
        list = self.state.to_index_list()
        np.testing.assert_array_equal(list, np.array([]))

    def test_nonempty_index_list(self):
        self.state._indices = [0]
        list = self.state.to_index_list()
        np.testing.assert_array_equal(list, np.array([0]))

    def test_nonempty_mask(self):
        self.state._indices = [0]
        mask = self.state.to_mask()
        np.testing.assert_array_equal(mask, np.array([[True], [False]]))

    def test_define_on_init(self):
        ind = np.array([0,1])
        state = ElementSubsetState(indices = ind)
        np.testing.assert_array_equal(ind, state._indices)


class TestSubsetIo(object):

    def setup_method(self, method):
        self.data = MagicMock()
        self.data.shape = (4,4)
        self.subset = Subset(self.data)
        inds = np.array([1,2,3])
        self.subset.subset_state = ElementSubsetState(indices = inds)

    def test_write(self):
        with tempfile.NamedTemporaryFile() as tmp:
            self.subset.write_mask(tmp.name)
            data = pyfits.open(tmp.name)[0].data
            expected = np.array([ [0, 1, 1, 1],
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
        with pytest.raises(IOError):
            self.subset.read_mask('file_does_not_exist')

    def test_write_unsupported_format(self):
        with pytest.raises(AttributeError):
            self.subset.write_mask('file_will_fail', format='.hd5')
