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
    def __init__(self, sender, tag=None):
        """
        Create a new message

        Parameters
        ----------
        sender: The object sending the message
        tag: An optional string describing the message

        """
        self.sender = sender
        self.tag = tag

    def __str__(self):
        return 'Message: "%s"\n\t Sent from: %s', self.tag, self.sender


class SubsetMessage(Message):
    """
    A general message issued by a subset.

    """

    def __init__(self, sender, tag=None):
        if (not isinstance(sender, cloudviz.Subset)):
            raise TypeError("Sender must be a subset: %s"
                            % type(sender))
        Message.__init__(self, sender, tag=tag)
    pass


class SubsetUpdateMessage(SubsetMessage):
    """
    A message that a subset issues when its state changes

    Attributes
    ----------
    attribute: string
             An optional label of what attribute has changed
    """
    def __init__(self, sender, attribute=None, tag=None):
        SubsetMessage.__init__(self, sender, tag=tag)
        self.attribute = attribute


class SubsetDeleteMessage(SubsetMessage):
    """
    A message that a subset issues when it is deleted
    """
    pass


class DataMessage(Message):
    """
    The base class for messages that data objects issue
    """
    def __init__(self, sender, tag=None):
        if (not isinstance(sender, cloudviz.Data)):
            raise TypeError("Sender must be a data instance: %s"
                            % type(sender))
        Message.__init__(self, sender, tag=tag)
