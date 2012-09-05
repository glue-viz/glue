#pylint: disable=W0613,W0201,W0212,E1101,E1103
import numpy as np

from ..edit_subset_mode import *
from ..subset import ElementSubsetState
from ..data import Component, Data


class TestEditSubsetMode(object):
    def setup_method(self, method):
        data = Data()
        comp = Component(np.array([1, 2, 3]))

        ind1 = np.array([0, 1])
        ind2 = np.array([1, 2])

        cid = data.add_component(comp, 'test')
        state1 = ElementSubsetState(ind1)
        state2 = ElementSubsetState(ind2)

        data.edit_subset.subset_state = state1
        state2.parent = state1.parent
        self.data = data
        self.cid = cid
        self.state1 = state1
        self.state2 = state2

    def check_mode(self, mode, expected):
        edit_mode = EditSubsetMode()
        edit_mode.mode = mode
        edit_mode.combine(self.data.edit_subset, self.state2)
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

    def test_spawn(self):
        """ spawn subset replaces edit subset, and sends current edit to new"""
        n0 = len(self.data.subsets)
        self.check_mode(SpawnMode, [False, True, True])
        assert len(self.data.subsets) == n0 + 1
