"""
.. module::glue.message

"""
from inspect import getmro

__all__ = ['Message', 'ErrorMessage', 'SubsetMessage', 'SubsetCreateMessage',
           'SubsetUpdateMessage', 'SubsetDeleteMessage', 'DataMessage',
           'DataAddComponentMessage', 'DataUpdateMessage',
           'DataCollectionMessage', 'DataCollectionActiveChange',
           'DataCollectionActiveDataChange', 'DataCollectionAddMessage',
           'DataCollectionDeleteMessage']


class Message(object):
    """
    Base class for messages that the hub handles.

    Each message represents a specific kind of event. After clients
    register to a hub, the subscribe to specific message classes, and
    will only receive those kinds of messages.

    The message class family is hierarchical, and a client subscribing
    to a message class implicitly subscribes to all of its subclasses.

    Attributes :

    ----------
    sender : The object which sent the message
    tag : An optional string describing the message
    """
    def __init__(self, sender, tag=None, _level=0):
        """Create a new message

        Parameters
        ----------
        sender : The object sending the message

        tag : An optional string describing the message
        """
        self.sender = sender
        self.tag = tag
        self._level = len(getmro(type(self)))  # number of super-classes.

    def __lt__(self, other):
        """Compares 2 message objects. message 1 < message 2 if it is
        further up the hierarchy (i.e. closer to the Message superclass).

        Note:

        This comparison breaks down slightly if messages are multiply
        inherited.

        """

        return self._level < other._level

    def __str__(self):
        return 'Message: "%s"\n\t Sent from: %s' % (self.tag, self.sender)


class ErrorMessage(Message):
    """ Used to send general purpose error messages """
    pass


class SubsetMessage(Message):
    """
    A general message issued by a subset.

    """

    def __init__(self, sender, tag=None):
        from .subset import Subset
        if (not isinstance(sender, Subset)):
            raise TypeError("Sender must be a subset: %s"
                            % type(sender))
        Message.__init__(self, sender, tag=tag)
        self.subset = self.sender


class SubsetCreateMessage(SubsetMessage):
    """
    A message that a subset issues when its state changes
    """
    pass


class SubsetUpdateMessage(SubsetMessage):
    """
    A message that a subset issues when its state changes

    Attributes
    -----------
    attribute : string
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
        from .data import Data
        if (not isinstance(sender, Data)):
            raise TypeError("Sender must be a data instance: %s"
                            % type(sender))
        Message.__init__(self, sender, tag=tag)
        self.data = self.sender


class DataAddComponentMessage(DataMessage):
    def __init__(self, sender, component_id, tag=None):
        super(DataAddComponentMessage, self).__init__(sender, tag=tag)
        self.component_id = component_id


class DataUpdateMessage(DataMessage):
    def __init__(self, sender, attribute, tag=None):
        super(DataUpdateMessage, self).__init__(sender, tag=tag)
        self.attribute = attribute


class DataCollectionMessage(Message):
    def __init__(self, sender, tag=None):
        from .data_collection import DataCollection
        if (not isinstance(sender, DataCollection)):
            raise TypeError("Sender must be a DataCollection instance: %s"
                            % type(sender))
        Message.__init__(self, sender, tag=tag)


class DataCollectionActiveChange(DataCollectionMessage):
    pass


class DataCollectionActiveDataChange(DataCollectionMessage):
    pass


class DataCollectionAddMessage(DataCollectionMessage):
    def __init__(self, sender, data, tag=None):
        DataCollectionMessage.__init__(self, sender, tag=tag)
        self.data = data


class DataCollectionDeleteMessage(DataCollectionMessage):
    def __init__(self, sender, data, tag=None):
        DataCollectionMessage.__init__(self, sender, tag=tag)
        self.data = data

class LiveLinkMessage(Message):
    def __init__(self, sender, link, tag=None):
        super(LiveLinkMessage, self).__init__(sender, tag=tag)
        self.link = link

class LiveLinkAddMessage(LiveLinkMessage):
    pass

class LiveLinkDeleteMessage(LiveLinkMessage):
    pass
