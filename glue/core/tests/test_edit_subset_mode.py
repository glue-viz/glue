# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function

import itertools

import pytest
import numpy as np

from ..data import Component, Data
from ..data_collection import DataCollection
from ..edit_subset_mode import (EditSubsetMode, ReplaceMode, OrMode, AndMode,
                                XorMode, AndNotMode)
from ..subset import ElementSubsetState, SubsetState


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
        self.data = data
        self.cid = cid
        self.state1 = state1
        self.state2 = state2

    def check_mode(self, mode, expected):
        edit_mode = EditSubsetMode()
        edit_mode.mode = mode
        edit_mode.update(self.data, self.state2)
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

    def test_combine_maps_over_multiselection(self):
        """If data has many edit subsets, act on all of them"""
        mode = EditSubsetMode()
        mode.mode = ReplaceMode
        for i in range(5):
            self.data.new_subset()

        self.data.edit_subset = list(self.data.subsets)

        mode.update(self.data, self.state2)
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

        mode.update(dc, self.state2)
        expected = np.array([False, True, True])
        for s in self.data.subsets:
            np.testing.assert_array_equal(s.to_mask(), expected)

    def test_combines_make_copy(self):
        mode = EditSubsetMode()
        mode.mode = ReplaceMode
        self.data.edit_subset = self.data.new_subset()
        mode.update(self.data, self.state2)
        assert self.data.edit_subset.subset_state is not self.state2


# Tests for multiselection logic
combs = list(itertools.product([True, False], [True, False],
                               [True, False], [True, False]))


@pytest.mark.parametrize(("emp", "loc", "glob", "foc"), combs)
def test_multiselect(emp, loc, glob, foc):
    """Test logic of when subsets should be updated/added, given
    the state of all editable subsets in a data collection.

    We consider four variables. The first data set in the collection
    is tested, and considired the 'local' data
    :param emp: Is the local set empty (i.e. no subsets)?
    :param loc: Are any of the local subsets editable?
    :param glob: Are any non-local subsets editable?
    :param foc: Does the local dataset have focus?
    """
    if emp and loc:  # can't be empty with selections
        return
    dc, state = setup_multi(emp, loc, glob, foc)
    did_add, did_apply = apply(dc, state, foc)
    assert did_add == should_add(emp, loc, glob, foc)
    assert did_apply == should_apply(emp, loc, glob, foc)


def setup_multi(empty, local_select, global_select, focus):
    d1 = Data()
    d2 = Data()
    dc = DataCollection([d1, d2])
    EditSubsetMode().data_collection = dc

    d2.new_subset()
    if not empty:
        d1.new_subset()
    if (not empty) and local_select:
        d1.edit_subset = d1.subsets[0]

    if global_select:
        d2.edit_subset = d2.subsets[0]

    state = SubsetState()
    return dc, state


def should_add(emp, loc, glob, foc):
    return foc and not (loc or glob)


def should_apply(emp, loc, glob, foc):
    return loc and not emp


def apply(dc, state, focus=False):
    """Update data collection, return did_add, did_apply for
    first data object"""

    ct = len(dc[0].subsets)

    sub = dc[0].edit_subset
    if isinstance(sub, list):
        sub = None if len(sub) == 0 else sub[0]

    old_state = None
    if sub is not None:
        old_state = sub.subset_state

    mode = EditSubsetMode()
    mode.mode = ReplaceMode
    mode.update(dc, state, dc[0] if focus else None)

    print(len(dc[0].subsets))
    did_add = len(dc[0].subsets) > ct
    did_apply = sub is not None and sub.subset_state is not old_state
    return did_add, did_apply
