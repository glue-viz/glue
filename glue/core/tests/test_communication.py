# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function

import pytest

from ..client import Client
from ..data import Data
from ..data_collection import DataCollection
from ..hub import Hub
from ..message import (SubsetCreateMessage, SubsetDeleteMessage,
                       SubsetUpdateMessage, Message, DataUpdateMessage)
from ..subset import Subset


"""
Client communication protocol

subsets added to data on creation
subsets subscribe to hub when data does

data are not added to clients automatically
subsets added to client only if data is in client

All create, update, delete events should emit signals
Processed (or ignored!) by clients
"""


class _TestClient(Client):

    def __init__(self, data):
        Client.__init__(self, data)
        self.last_message = None
        self.call = None

    def _add_subset(self, message):
        self.last_message = message
        self.call = self._add_subset

    def _remove_subset(self, message):
        self.last_message = message
        self.call = self._remove_subset

    def _update_subset(self, message):
        self.last_message = message
        self.call = self._update_subset

    def _update_data(self, message):
        self.last_message = message
        self.call = self._update_data


class TestCommunication(object):

    def setup_method(self, method):
        self.hub = Hub()
        self.d1 = Data()
        self.d2 = Data()
        self.d3 = Data()
        dc = DataCollection([self.d1])
        self.c1 = _TestClient(dc)
        self.c2 = _TestClient(DataCollection([self.d2]))
        self.c3 = _TestClient(dc)
        self.s1 = Subset(self.d1)
        self.s2 = Subset(self.d2)
        self.m1 = SubsetCreateMessage(self.s1)
        self.m2 = SubsetDeleteMessage(self.s1)
        self.m3 = SubsetUpdateMessage(self.s1)
        self.m4 = DataUpdateMessage(self.d1, 'dummy_attribute')

    def test_basic_register(self):
        # create and register a client. Make sure it's
        # added to subscription table

        h = Hub()
        d = Data()
        c = _TestClient(DataCollection([d]))
        assert not c in h._subscriptions
        c.register_to_hub(h)
        assert c in h._subscriptions

    def test_basic_broadcast(self):
        # broadcast a subsetCreateMessage.
        # make sure the registered client catches it.
        # make sure an unregistered one doesn't

        self.c1.register_to_hub(self.hub)
        self.hub.broadcast(self.m1)

        assert self.c1.last_message is self.m1
        assert self.c1.call == self.c1._add_subset
        assert self.c2.last_message is None

    def test_proper_handlers(self):
        # broadcast the 4 basic methods. make sure the proper handlers
        # catch them
        self.c1.register_to_hub(self.hub)
        assert self.c1.call is None

        self.hub.broadcast(self.m1)
        assert self.c1.call == self.c1._add_subset

        self.hub.broadcast(self.m2)
        assert self.c1.call == self.c1._remove_subset

        self.hub.broadcast(self.m3)
        assert self.c1.call == self.c1._update_subset

        self.hub.broadcast(self.m4)
        assert self.c1.call == self.c1._update_data

    def test_ignore_message(self):
        # send a message that should be ignored
        class IgnoredMessage(Message):
            pass
        self.c1.register_to_hub(self.hub)
        self.hub.broadcast(IgnoredMessage(None))
        assert self.c1.last_message is None
        assert self.c1.call is None

    @pytest.mark.skipif(True, reason="Relaxed requirement. Hub now ignores exceptions")
    def test_uncaught_message(self):
        # broadcast a message without a message handler
        self.hub.subscribe(self.c1, Message)
        with pytest.raises(NotImplementedError) as exc:
            self.hub.broadcast(Message(None))
        assert exc.value.args[0].startswith("Message has no handler:")

    def test_multi_client(self):
        # register 2 clients with same data to hub
        # make sure events get to both

        self.c1.register_to_hub(self.hub)
        self.c3.register_to_hub(self.hub)
        self.hub.broadcast(self.m1)
        assert self.c1.last_message is self.m1
        assert self.c3.last_message is self.m1

    def test_standard_filter(self):
        # register 2 clients with 2 different data sets
        # make sure events are filtered properly
        self.c1.register_to_hub(self.hub)
        self.c2.register_to_hub(self.hub)

        msg = DataUpdateMessage(self.d2, 'test_attribute')
        self.hub.broadcast(msg)

        assert self.c1.last_message is None
        assert self.c2.last_message is msg

    def test_unsubscribe(self):
        # subscribe and unsubscribe an object.
        # make sure message passing stays correct

        self.c1.register_to_hub(self.hub)
        self.hub.broadcast(self.m1)
        assert self.c1.last_message is self.m1

        self.hub.unsubscribe(self.c1, type(self.m2))
        self.hub.broadcast(self.m2)
        assert self.c1.last_message is self.m1

    def test_remove_client(self):

        self.c1.register_to_hub(self.hub)
        self.hub.broadcast(self.m1)
        assert self.c1.last_message is self.m1

        self.hub.unsubscribe_all(self.c1)
        self.hub.broadcast(self.m2)
        assert self.c1.last_message is self.m1

    def test_subset_relay(self):
        # make sure subset modification
        # sends messages
        d = Data()
        dc = DataCollection(d)
        c = _TestClient(dc)

        c.register_to_hub(dc.hub)
        sub = d.new_subset()
        assert c.last_message.sender is sub
        assert c.call == c._add_subset

        sub.modified = "modify"
        assert c.call == c._update_subset
        assert c.last_message.attribute == 'modified'
