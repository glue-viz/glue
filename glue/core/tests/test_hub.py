# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function

import pytest
from mock import MagicMock

from ..data import Data
from ..data_collection import DataCollection
from ..exceptions import InvalidSubscriber, InvalidMessage
from ..hub import Hub, HubListener
from ..message import SubsetMessage, Message
from ..subset import Subset


class TestHub(object):

    def setup_method(self, method):
        self.hub = Hub()

    def get_subscription(self):
        msg = Message
        handler = MagicMock()
        subscriber = MagicMock(spec_set=HubListener)
        return msg, handler, subscriber

    def test_subscribe(self):
        msg, handler, subscriber = self.get_subscription()
        self.hub.subscribe(subscriber, msg, handler)
        assert self.hub.is_subscribed(subscriber, msg)
        assert self.hub.get_handler(subscriber, msg) == handler

    def test_get_handler(self):
        msg, handler, subscriber = self.get_subscription()
        self.hub.subscribe(subscriber, msg, handler)
        assert self.hub.get_handler(subscriber, msg) == handler
        assert self.hub.get_handler(subscriber, None) is None
        assert self.hub.get_handler(None, msg) is None

    def test_unsubscribe(self):
        msg, handler, subscriber = self.get_subscription()
        self.hub.subscribe(subscriber, msg, handler)
        self.hub.unsubscribe(subscriber, msg)

        assert not self.hub.is_subscribed(subscriber, msg)
        assert self.hub.get_handler(subscriber, msg) is None

    def test_unsubscribe_all(self):
        msg, handler, subscriber = self.get_subscription()
        msg2 = SubsetMessage
        self.hub.subscribe(subscriber, msg, handler)
        self.hub.subscribe(subscriber, msg2, handler)
        self.hub.unsubscribe_all(subscriber)
        assert not self.hub.is_subscribed(subscriber, msg)
        assert not self.hub.is_subscribed(subscriber, msg2)

    def test_unsubscribe_specific_to_message(self):
        msg, handler, subscriber = self.get_subscription()
        msg2 = SubsetMessage
        self.hub.subscribe(subscriber, msg, handler)
        self.hub.subscribe(subscriber, msg2, handler)
        self.hub.unsubscribe(subscriber, msg)
        assert not self.hub.is_subscribed(subscriber, msg)
        assert self.hub.is_subscribed(subscriber, msg2)

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
        assert handler.call_count == 0

    def test_unsubscribe_spec_setific_to_message(self):
        msg, handler, subscriber = self.get_subscription()
        msg2 = SubsetMessage
        self.hub.subscribe(subscriber, msg2, handler)
        msg_instance = msg("Test")
        self.hub.broadcast(msg_instance)
        assert handler.call_count == 0

    def test_subscription_catches_message_subclasses(self):
        msg, handler, subscriber = self.get_subscription()
        msg2 = SubsetMessage
        self.hub.subscribe(subscriber, msg, handler)
        msg_instance = msg2(MagicMock(spec_set=Subset))
        self.hub.broadcast(msg_instance)
        handler.assert_called_once_with(msg_instance)

    def test_handler_ignored_if_subset_handler_present(self):
        msg, handler, subscriber = self.get_subscription()
        handler2 = MagicMock()
        msg2 = SubsetMessage
        self.hub.subscribe(subscriber, msg, handler)
        self.hub.subscribe(subscriber, msg2, handler2)
        msg_instance = SubsetMessage(Subset(None))
        self.hub.broadcast(msg_instance)
        handler2.assert_called_once_with(msg_instance)
        assert handler.call_count == 0

    def test_filter(self):
        msg, handler, subscriber = self.get_subscription()
        filter = lambda x: False
        self.hub.subscribe(subscriber, msg, handler)
        msg_instance = msg("Test")
        self.hub.broadcast(msg)
        assert handler.call_count == 0

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

        with pytest.raises(InvalidSubscriber) as exc:
            self.hub.subscribe(None, msg, handler)
        assert exc.value.args[0].startswith("Subscriber must be a HubListener")

        with pytest.raises(InvalidMessage) as exc:
            self.hub.subscribe(subscriber, None, handler)
        assert exc.value.args[0].startswith("message class must be "
                                            "a subclass of glue.Message")

    def test_default_handler(self):
        msg, handler, subscriber = self.get_subscription()
        self.hub.subscribe(subscriber, msg)
        msg_instance = msg("Test")

        self.hub.broadcast(msg_instance)
        subscriber.notify.assert_called_once_with(msg_instance)

    def test_autosubscribe(self):
        l = MagicMock(spec_set=HubListener)
        d = MagicMock(spec_set=Data)
        s = MagicMock(spec_set=Subset)
        dc = MagicMock(spec_set=DataCollection)
        hub = Hub(l, d, s, dc)

        l.register_to_hub.assert_called_once_with(hub)
        d.register_to_hub.assert_called_once_with(hub)
        dc.register_to_hub.assert_called_once_with(hub)
        s.register.assert_called_once_with()

    def test_invalid_init(self):
        with pytest.raises(TypeError) as exc:
            Hub(None)
        assert exc.value.args[0] == ("Inputs must be HubListener, data, "
                                     "subset, or data collection objects")


class TestHubListener(object):
    """This is a dumb test, I know. Fixated on code coverage"""

    def test_unimplemented(self):
        hl = HubListener()
        with pytest.raises(NotImplementedError):
            hl.register_to_hub(None)
        with pytest.raises(NotImplementedError):
            hl.notify(None)
