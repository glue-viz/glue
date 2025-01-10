# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

import operator
from unittest.mock import MagicMock

import numpy as np

from ..application_base import Application
from ..hub import HubListener
from ..message import SubsetCreateMessage, SubsetUpdateMessage, SubsetDeleteMessage
from ..command import ApplySubsetState
from ..data import Component, Data
from ..data_collection import DataCollection
from ..edit_subset_mode import (EditSubsetMode, ReplaceMode, OrMode, AndMode,
                                XorMode, AndNotMode)
from ..subset import ElementSubsetState, InequalitySubsetState


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


def test_no_double_messaging():

    # Make sure that when we create a new subset via EditSubsetMode, we don't
    # get two separate messages for creation and updating, but instead just a
    # single create message with the right subset state.

    handler = MagicMock()

    app = Application()

    class Listener(HubListener):
        pass

    listener = Listener()

    app.session.hub.subscribe(listener, SubsetCreateMessage, handler=handler)
    app.session.hub.subscribe(listener, SubsetUpdateMessage, handler=handler)

    data = Data(x=[1, 2, 3, 4, 5])

    app.data_collection.append(data)

    cmd = ApplySubsetState(data_collection=app.data_collection,
                           subset_state=data.id['x'] >= 3,
                           override_mode=ReplaceMode)

    app.session.command_stack.do(cmd)

    assert handler.call_count == 1
    message = handler.call_args[0][0]
    assert isinstance(message, SubsetCreateMessage)
    assert isinstance(message.subset.subset_state, InequalitySubsetState)
    assert message.subset.subset_state.left is data.id['x']
    assert message.subset.subset_state.operator is operator.ge
    assert message.subset.subset_state.right == 3


def test_broadcast_to_collection():
    """A data collection input works on each data component"""

    handler = MagicMock()

    app = Application()

    class Listener(HubListener):
        pass

    listener = Listener()

    app.session.hub.subscribe(listener, SubsetCreateMessage, handler=handler)
    app.session.hub.subscribe(listener, SubsetUpdateMessage, handler=handler)

    data = Data(x=[1, 2, 3, 4, 5])
    data2 = Data(x=[2, 3, 4, 5, 6])

    app.data_collection.append(data)
    app.data_collection.append(data2)

    cmd = ApplySubsetState(data_collection=app.data_collection,
                           subset_state=data.id['x'] >= 3,
                           override_mode=ReplaceMode)

    app.session.command_stack.do(cmd)

    assert len(data2.subsets) == 1
    assert data2.subsets[0].label == 'Subset 1'
    # fails with `IncompatibleAttribute` exception
    # assert data2.get_subset_object(subset_id='Subset 1', cls=NDDataArray)

    assert handler.call_count == 2

    assert data2.subsets[0].subset_state.left is data.id['x']
    assert data2.subsets[0].subset_state.operator is operator.ge
    assert data2.subsets[0].subset_state.right == 3


def test_message_once_all_subsets_exist():

    # Make sure that when we create a new subset via EditSubsetMode, we don't
    # get two separate messages for creation and updating, but instead just a
    # single create message with the right subset state.

    app = Application()

    class Listener(HubListener):
        pass

    listener = Listener()

    data1 = Data(x=[1, 2, 3, 4, 5])
    app.data_collection.append(data1)

    data2 = Data(x=[1, 2, 3, 4, 5])
    app.data_collection.append(data2)

    count = [0]

    def handler(msg):
        assert len(data1.subsets) == len(data2.subsets)
        for i in range(len(data1.subsets)):
            assert data2.subsets[i].subset_state is data1.subsets[i].subset_state
        count[0] += 1

    app.session.hub.subscribe(listener, SubsetCreateMessage, handler=handler)
    app.session.hub.subscribe(listener, SubsetDeleteMessage, handler=handler)

    cmd = ApplySubsetState(data_collection=app.data_collection,
                           subset_state=data1.id['x'] >= 3,
                           override_mode=ReplaceMode)

    app.session.command_stack.do(cmd)

    assert count[0] == 2

    app.data_collection.remove_subset_group(app.data_collection.subset_groups[0])

    assert count[0] == 4
