"""
.. module::cloudviz.message

"""
import cloudviz


class Message(object):
    """
    Base class for messages that the hub handles.

    Each message represents a specific kind of event. After clients
    register to a hub, the subscribe to specific message classes, and
    will only receive those kinds of messages.

    The message class family is hierarchical, and a client subscribing
    to a message class implicitly subscribes to all of its subclasses.

    Attributes
    ----------
    sender: The object which sent the message
    tag: An optional string describing the message

    """
    def __init__(self, sender, tag=None, _level=0):
        """
        Create a new message

        Parameters
        ----------
        sender: The object sending the message
        tag: An optional string describing the message

        """
        self.sender = sender
        self.tag = tag
        self._level = _level  # setting this manually is fragile

    def __lt__(self, other):
        """
        Compares 2 message objects. message 1 < message 2 if it is
        further up the hierarchy (i.e. closer to the Message superclass).
        """

        return self._level < other._level

    def __str__(self):
        return 'Message: "%s"\n\t Sent from: %s' % (self.tag, self.sender)


class SubsetMessage(Message):
    """
    A general message issued by a subset.

    """

    def __init__(self, sender, tag=None, _level=0):
        if (not isinstance(sender, cloudviz.Subset)):
            raise TypeError("Sender must be a subset: %s"
                            % type(sender))
        Message.__init__(self, sender, tag=tag, _level=_level + 1)
        self.subset = self.sender


class SubsetCreateMessage(SubsetMessage):
    """
    A message that a subset issues when its state changes

    """
    def __init__(self, sender, tag=None,
                 _level=0):
        SubsetMessage.__init__(self, sender, tag=tag,
                               _level=_level + 1)


class SubsetUpdateMessage(SubsetMessage):
    """
    A message that a subset issues when its state changes

    Attributes
    ----------
    attribute: string
             An optional label of what attribute has changed
    """
    def __init__(self, sender, attribute=None, tag=None,
                 _level=0):
        SubsetMessage.__init__(self, sender, tag=tag,
                               _level=_level + 1)
        self.attribute = attribute


class SubsetDeleteMessage(SubsetMessage):
    """
    A message that a subset issues when it is deleted
    """
    def __init__(self, sender, tag=None, _level=0):
        SubsetMessage.__init__(self, sender, tag=tag,
                               _level=_level + 1)


class DataMessage(Message):
    """
    The base class for messages that data objects issue
    """
    def __init__(self, sender, tag=None, _level=0):
        if (not isinstance(sender, cloudviz.Data)):
            raise TypeError("Sender must be a data instance: %s"
                            % type(sender))
        Message.__init__(self, sender, tag=tag,
                         _level=_level + 1)
        self.data = self.sender
