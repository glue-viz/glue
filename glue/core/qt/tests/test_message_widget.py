from glue.core.hub import Hub
from glue.core.message import Message

from ..message_widget import MessageWidget


def test_message_widget_runs():

    hub = Hub()

    widget = MessageWidget()
    widget.register_to_hub(hub)
    widget.show()

    message = Message('test_message_widget_runs', tag='1234')

    hub.broadcast(message)

    # TODO: check content of widget window
