import unittest

from mock import MagicMock

import glue
from glue.exceptions import InvalidSubscriber, InvalidMessage
from glue.message import ErrorMessage

class TestHub(unittest.TestCase):

    def setUp(self):
        self.hub = glue.Hub()

    def get_subscription(self):
        msg = glue.message.Message
        handler = MagicMock()
        subscriber  = MagicMock(spec_set=glue.hub.HubListener)
        return msg, handler, subscriber

    def test_subscribe(self):
        msg, handler, subscriber = self.get_subscription()
        self.hub.subscribe(subscriber, msg, handler)
        self.assertTrue(self.hub.is_subscribed(subscriber, msg))
        self.assertEquals(self.hub.get_handler(subscriber, msg), handler)

    def test_get_handler(self):
        msg, handler, subscriber = self.get_subscription()
        self.hub.subscribe(subscriber, msg, handler)
        self.assertEquals(self.hub.get_handler(subscriber, msg), handler)
        self.assertEquals(self.hub.get_handler(subscriber, None), None)
        self.assertEquals(self.hub.get_handler(None, msg), None)

    def test_unsubscribe(self):
        msg, handler, subscriber = self.get_subscription()
        self.hub.subscribe(subscriber, msg, handler)
        self.hub.unsubscribe(subscriber, msg)

        self.assertFalse(self.hub.is_subscribed(subscriber, msg))
        self.assertEquals(self.hub.get_handler(subscriber, msg), None)

    def test_unsubscribe_all(self):
        msg, handler, subscriber = self.get_subscription()
        msg2 = glue.message.SubsetMessage
        self.hub.subscribe(subscriber, msg, handler)
        self.hub.subscribe(subscriber, msg2, handler)
        self.hub.unsubscribe_all(subscriber)
        self.assertFalse(self.hub.is_subscribed(subscriber, msg))
        self.assertFalse(self.hub.is_subscribed(subscriber, msg2))

    def test_unsubscribe_specific_to_message(self):
        msg, handler, subscriber = self.get_subscription()
        msg2 = glue.message.SubsetMessage
        self.hub.subscribe(subscriber, msg, handler)
        self.hub.subscribe(subscriber, msg2, handler)
        self.hub.unsubscribe(subscriber, msg)
        self.assertFalse(self.hub.is_subscribed(subscriber, msg))
        self.assertTrue(self.hub.is_subscribed(subscriber, msg2))

    def test_broadcast(self):
        msg, handler, subscriber = self.get_subscription()
        self.hub.subscribe(subscriber, msg, handler)
        msg_instance = msg("Test")
        self.hub.broadcast(msg_instance)
        handler.assert_called_once_with(msg_instance)

    def test_unsubscribe_halts_broadcast(self):
        msg, handler, subscriber = self.get_subscription()
        self.hub.subscribe(subscriber, msg, handler)
        self.hub.unsubscribe(subscriber, msg)
        msg_instance = msg("Test")
        self.hub.broadcast(msg_instance)
        self.assertEquals(handler.call_count, 0)

    def test_unsubscribe_spec_setific_to_message(self):
        msg, handler, subscriber = self.get_subscription()
        msg2 = glue.message.SubsetMessage
        self.hub.subscribe(subscriber, msg2, handler)
        msg_instance = msg("Test")
        self.hub.broadcast(msg_instance)
        self.assertEquals(handler.call_count, 0)

    def test_subscription_catches_message_subclasses(self):
        msg, handler, subscriber = self.get_subscription()
        msg2 = glue.message.SubsetMessage
        self.hub.subscribe(subscriber, msg, handler)
        msg_instance = msg2(MagicMock(spec_set=glue.Subset))
        self.hub.broadcast(msg_instance)
        handler.assert_called_once_with(msg_instance)

    def test_handler_ignored_if_subset_handler_present(self):
        msg, handler, subscriber = self.get_subscription()
        handler2 = MagicMock()
        msg2 = glue.message.SubsetMessage
        self.hub.subscribe(subscriber, msg, handler)
        self.hub.subscribe(subscriber, msg2, handler2)
        msg_instance = msg2(MagicMock(spec_set=glue.Subset))
        self.hub.broadcast(msg_instance)
        handler2.assert_called_once_with(msg_instance)
        self.assertEquals(handler.call_count, 0)

    def test_filter(self):
        msg, handler, subscriber = self.get_subscription()
        filter = lambda x: False
        self.hub.subscribe(subscriber, msg, handler)
        msg_instance = msg("Test")
        self.hub.broadcast(msg)
        self.assertEquals(handler.call_count, 0)

    def test_broadcast_sends_to_all_subsribers(self):
        msg, handler, subscriber = self.get_subscription()
        msg, handler2, subscriber2 = self.get_subscription()

        self.hub.subscribe(subscriber, msg, handler)
        self.hub.subscribe(subscriber2, msg, handler2)
        msg_instance = msg("Test")
        self.hub.broadcast(msg_instance)
        handler.assert_called_once_with(msg_instance)
        handler2.assert_called_once_with(msg_instance)

    def test_invalid_unsubscribe_ignored(self):
        msg, handler, subscriber = self.get_subscription()
        self.hub.unsubscribe(handler, subscriber)

    def test_invalid_subscribe(self):
        msg, handler, subscriber = self.get_subscription()

        self.assertRaises(InvalidSubscriber,
                          self.hub.subscribe,
                          None, msg, handler)

        self.assertRaises(InvalidMessage,
                          self.hub.subscribe,
                          subscriber, None, handler)

    def test_default_handler(self):
        msg, handler, subscriber = self.get_subscription()
        self.hub.subscribe(subscriber, msg)
        msg_instance = msg("Test")

        self.hub.broadcast(msg_instance)
        subscriber.notify.assert_called_once_with(msg_instance)

    def test_exception_on_broadcast(self):
        msg, handler, subscriber = self.get_subscription()
        error_handler = MagicMock()
        self.hub.subscribe(subscriber, msg, handler)
        self.hub.subscribe(subscriber, ErrorMessage, error_handler)

        test_exception = Exception("Test")
        handler.side_effect = test_exception

        msg_instance = msg("Test")
        self.hub.broadcast(msg_instance)

        error_handler.assert_called_once()
        err_msg = error_handler.call_args[0][0]
        self.assertEquals(err_msg.tag, "%s" % test_exception)

    def test_excpetions_dont_recurse_on_broadcast(self):
        msg, handler, subscriber = self.get_subscription()
        error_handler = MagicMock()
        handler.side_effect = Exception("First Exception")
        error_handler.side_effect = Exception("Don't recurse forever!")

        self.hub.subscribe(subscriber, msg, handler)
        self.hub.subscribe(subscriber, ErrorMessage, error_handler)


        msg_instance = msg("test")
        self.hub.broadcast(msg_instance)

    def test_autosubscribe(self):
        l = MagicMock(spec_set=glue.hub.HubListener)
        d = MagicMock(spec_set=glue.Data)
        s = MagicMock(spec_set=glue.Subset)
        dc = MagicMock(spec_set=glue.DataCollection)
        hub = glue.Hub(l, d, s, dc)

        l.register_to_hub.assert_called_once_with(hub)
        d.register_to_hub.assert_called_once_with(hub)
        dc.register_to_hub.assert_called_once_with(hub)
        s.register.assert_called_once_with()

    def test_invalid_init(self):
        self.assertRaises(TypeError, glue.Hub, None)

class TestHubListener(unittest.TestCase):
    """This is a dumb test, I know. Fixated on code coverage"""
    def test_unimplemented(self):
        hl = glue.HubListener()
        self.assertRaises(NotImplementedError, hl.register_to_hub, None)
        self.assertRaises(NotImplementedError, hl.notify, None)

if __name__ == "__main__":
    unittest.main()