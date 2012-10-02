#pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103
import numpy as np

from ..edit_subset_mode import (EditSubsetMode, ReplaceMode, OrMode, AndMode,
                                XorMode, AndNotMode)
from ..subset import ElementSubsetState
from ..data import Component, Data
from ..data_collection import DataCollection

class TestEditSubsetMode(object):
    def setup_method(self, method):
        data = Data()
        comp = Component(np.array([1, 2, 3]))

        ind1 = np.array([0, 1])
        ind2 = np.array([1, 2])

        cid = data.add_component(comp, 'test')
        state1 = ElementSubsetState(ind1)
        state2 = ElementSubsetState(ind2)

        data.edit_subset = data.new_subset()

        data.edit_subset.subset_state = state1
        state2.parent = state1.parent
        self.data = data
        self.cid = cid
        self.state1 = state1
        self.state2 = state2

    def check_mode(self, mode, expected):
        edit_mode = EditSubsetMode()
        edit_mode.mode = mode
        edit_mode.combine(self.data, self.state2)
        np.testing.assert_array_equal(self.data.edit_subset.to_mask(),
                                      expected)

    def test_replace(self):
        self.check_mode(ReplaceMode, [False, True, True])

    def test_or(self):
        self.check_mode(OrMode, [True, True, True])

    def test_and(self):
        self.check_mode(AndMode, [False, True, False])

    def test_xor(self):
        self.check_mode(XorMode, [True, False, True])

    def test_and_not(self):
        self.check_mode(AndNotMode, [True, False, False])

    def test_combine_adds_subset_if_empty(self):
        """If data has no subsets, one is created"""
        mode = EditSubsetMode()
        mode.mode = ReplaceMode
        self.data.edit_subset = None
        self.data.subsets = []
        mode.combine(self.data, self.state2)
        assert len(self.data.subsets) == 1
        assert self.data.edit_subset is not None

    def test_combine_ignores_nonselection(self):
        """If data has subsets but no edit subset, ignore"""
        mode = EditSubsetMode()
        mode.mode = ReplaceMode
        sub = self.data.new_subset()
        self.data.subsets = [sub]
        state = sub.subset_state
        self.data.edit_subset = None

        mode.combine(self.data, self.state2)
        assert sub.subset_state is state

    def test_combine_maps_over_multiselection(self):
        """If data has many edit subsets, act on all of them"""
        mode = EditSubsetMode()
        mode.mode = ReplaceMode
        for i in range(5):
            self.data.new_subset()

        self.data.edit_subset = list(self.data.subsets)

        mode.combine(self.data, self.state2)
        expected = np.array([False, True, True])
        for s in self.data.subsets:
            np.testing.assert_array_equal(s.to_mask(), expected)

    def test_combine_with_collection(self):
        """A data collection input works on each data object"""
        mode = EditSubsetMode()
        mode.mode = ReplaceMode

        for i in range(5):
            self.data.new_subset()
        self.data.edit_subset = list(self.data.subsets)

        dc = DataCollection([self.data])


        mode.combine(dc, self.state2)
        expected = np.array([False, True, True])
        for s in self.data.subsets:
            np.testing.assert_array_equal(s.to_mask(), expected)
