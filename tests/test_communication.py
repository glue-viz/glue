import unittest

from glue.core.hub import Hub
from glue.core.client import Client
from glue.core.data import Data
from glue.core.subset import Subset
from glue.core.data_collection import DataCollection
from glue.core.message import SubsetCreateMessage, SubsetDeleteMessage, SubsetUpdateMessage, Message, DataUpdateMessage

"""
Client communication protocol

subsets added to data on creation
subsets subscribe to hub when data does

data are not added to clients automatically
subsets added to client only if data is in client

All create, update, delete events should emit signals
Processed (or ignored!) by clients
"""

class C(Client):
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

class TestCommunication(unittest.TestCase):

    def setUp(self):
        self.hub = Hub()
        self.d1 = Data()
        self.d2 = Data()
        self.c1 = C(DataCollection([self.d1]))
        self.c2 = C(DataCollection([self.d2]))
        self.c3 = C(DataCollection([self.d1]))
        self.s1 = Subset(self.d1)
        self.s2 = Subset(self.d2)
        self.m1 = SubsetCreateMessage(self.s1)
        self.m2 = SubsetDeleteMessage(self.s1)
        self.m3 = SubsetUpdateMessage(self.s1)
        self.m4 = DataUpdateMessage(self.d1, 'dummy_attribute')

    def test_basic_register(self):
        #create and register a client. Make sure it's
        #added to subscription table

        h = Hub()
        d = Data()
        c = C(DataCollection([d]))
        self.assertNotIn(c, h._subscriptions)
        c.register_to_hub(h)
        self.assertIn(c, h._subscriptions)

    def test_basic_broadcast(self):
        #broadcast a subsetCreateMessage.
        #make sure the registered client catches it.
        #make sure an unregistered one doesn't

        self.c1.register_to_hub(self.hub)
        self.hub.broadcast(self.m1)

        self.assertIs(self.c1.last_message, self.m1)
        self.assertEquals(self.c1.call, self.c1._add_subset)
        self.assertIsNone(self.c2.last_message)

    def test_proper_handlers(self):
        #broadcast the 4 basic methods. make sure the proper handlers
        #catch them
        self.c1.register_to_hub(self.hub)
        self.assertIsNone(self.c1.call)

        self.hub.broadcast(self.m1)
        self.assertEquals(self.c1.call, self.c1._add_subset)

        self.hub.broadcast(self.m2)
        self.assertEquals(self.c1.call, self.c1._remove_subset)

        self.hub.broadcast(self.m3)
        self.assertEquals(self.c1.call, self.c1._update_subset)

        self.hub.broadcast(self.m4)
        self.assertEquals(self.c1.call, self.c1._update_data)

    def test_ignore_message(self):
        #send a message that should be ignored
        class IgnoredMessage(Message):
            pass
        self.c1.register_to_hub(self.hub)
        self.hub.broadcast(IgnoredMessage(None))
        self.assertIsNone(self.c1.last_message)
        self.assertIsNone(self.c1.call)

    @unittest.skip("Relaxed requirement. Hub now ignores exceptions")
    def test_uncaught_message(self):
        #broadcast a message without a message handler
        self.hub.subscribe(self.c1, Message)
        self.assertRaises(NotImplementedError,
                          self.hub.broadcast, Message(None))

    def test_multi_client(self):
        #register 2 clients with same data to hub
        #make sure events get to both

        self.c1.register_to_hub(self.hub)
        self.c3.register_to_hub(self.hub)
        self.hub.broadcast(self.m1)
        self.assertIs(self.c1.last_message, self.m1)
        self.assertIs(self.c3.last_message, self.m1)

    def test_standard_filter(self):
        #register 2 clients with 2 different data sets
        #make sure events are filtered properly
        self.c1.register_to_hub(self.hub)
        self.c2.register_to_hub(self.hub)

        msg = DataUpdateMessage(self.d2, 'test_attribute')
        self.hub.broadcast(msg)

        self.assertIsNone(self.c1.last_message)
        self.assertIs(self.c2.last_message, msg)

    def test_unsubscribe(self):
        # subscribe and unsubscribe an object.
        # make sure message passing stays correct

        self.c1.register_to_hub(self.hub)
        self.hub.broadcast(self.m1)
        self.assertIs(self.c1.last_message, self.m1)

        self.hub.unsubscribe(self.c1, type(self.m2))
        self.hub.broadcast(self.m2)
        self.assertIs(self.c1.last_message, self.m1)

    def test_remove_client(self):

        self.c1.register_to_hub(self.hub)
        self.hub.broadcast(self.m1)
        self.assertIs(self.c1.last_message, self.m1)

        self.hub.unsubscribe_all(self.c1)
        self.hub.broadcast(self.m2)
        self.assertIs(self.c1.last_message, self.m1)

    def test_subset_relay(self):
        #make sure subset modification
        #sends messages
        self.c1.register_to_hub(self.hub)
        self.d1.register_to_hub(self.hub)
        self.s1.no_echo_before_registration = 1

        self.assertIsNone(self.c1.last_message)
        self.s1.register()
        self.assertIsNotNone(self.c1.last_message)
        self.assertIs(self.c1.last_message.sender, self.s1)
        self.assertEquals(self.c1.call, self.c1._add_subset)

        self.s1.echo_after_registration = "1"
        self.assertEquals(self.c1.call, self.c1._update_subset)
        self.assertEquals(self.c1.last_message.attribute, 'echo_after_registration')

if __name__ == "__main__":
    unittest.main()
