# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

import numpy as np

from ..data import Component, Data
from ..data_collection import DataCollection
from ..edit_subset_mode import (EditSubsetMode, ReplaceMode, OrMode, AndMode,
                                XorMode, AndNotMode)
from ..subset import ElementSubsetState


class TestEditSubsetMode(object):

    def setup_method(self, method):
        data = Data()
        comp = Component(np.array([1, 2, 3]))

        ind1 = np.array([0, 1])
        ind2 = np.array([1, 2])

        cid = data.add_component(comp, 'test')
        state1 = ElementSubsetState(ind1)
        state2 = ElementSubsetState(ind2)

        self.edit_mode = EditSubsetMode()
        self.edit_mode.edit_subset = data.new_subset()
        self.edit_mode.edit_subset.subset_state = state1

        self.data = data
        self.cid = cid
        self.state1 = state1
        self.state2 = state2

    def check_mode(self, mode, expected):
        self.edit_mode.mode = mode
        self.edit_mode.update(self.data, self.state2)
        np.testing.assert_array_equal(self.edit_mode.edit_subset.to_mask(),
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

    def test_combine_maps_over_multiselection(self):
        """If data has many edit subsets, act on all of them"""

        self.edit_mode.mode = ReplaceMode

        for i in range(5):
            self.data.new_subset()
        self.edit_mode.edit_subset = list(self.data.subsets)

        self.edit_mode.update(self.data, self.state2)
        expected = np.array([False, True, True])
        for s in self.data.subsets:
            np.testing.assert_array_equal(s.to_mask(), expected)

    def test_combine_with_collection(self):
        """A data collection input works on each data object"""

        self.edit_mode.mode = ReplaceMode

        for i in range(5):
            self.data.new_subset()
        self.edit_mode.edit_subset = list(self.data.subsets)

        dc = DataCollection([self.data])

        self.edit_mode.update(dc, self.state2)
        expected = np.array([False, True, True])
        for s in self.data.subsets:
            np.testing.assert_array_equal(s.to_mask(), expected)

    def test_combines_make_copy(self):
        self.edit_mode.mode = ReplaceMode
        self.edit_mode.edit_subset = self.data.new_subset()
        self.edit_mode.update(self.data, self.state2)
        assert self.edit_mode.edit_subset.subset_state is not self.state2
