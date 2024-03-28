"""
Test that data.update_components() sends a NumericalDataChangedMessage
that conveys which components have been changed.
"""
from glue.core.data import Data
from glue.core.hub import HubListener
from glue.core.data_collection import DataCollection
from glue.core.message import NumericalDataChangedMessage

import numpy as np
from numpy.testing import assert_array_equal


def test_message_carries_components():

    test_data = Data(x=np.array([1, 2, 3, 4, 5]), y=np.array([1, 2, 3, 4, 5]), label='test_data')
    data_collection = DataCollection([test_data])

    class CustomListener(HubListener):

        def __init__(self, hub):
            self.received = 0
            self.components_changed = None
            hub.subscribe(self, NumericalDataChangedMessage,
                            handler=self.receive_message)

        def receive_message(self, message):
            self.received += 1
            try:
                self.components_changed = message.components_changed
            except AttributeError:
                self.components_changed = None

    listener = CustomListener(data_collection.hub)
    assert listener.received == 0
    assert listener.components_changed is None

    cid_to_change = test_data.id['x']
    new_data = [5, 2, 6, 7, 10]
    test_data.update_components({cid_to_change: new_data})

    assert listener.received == 1
    assert cid_to_change in listener.components_changed

    assert_array_equal(test_data['x'], new_data)
