# The classes in this file define toolbar modes. Mouse modes specifically
# are defined in mouse_modes.py

from glue.utils import nonpartial

__all__ = ['Mode', 'NonCheckableMode', 'CheckableMode']


class Mode(object):
    """
    The base class for all toolbar modes.

    All modes have the following attributes:

    * icon : QIcon object
    * mode_id : a short name for the mode
    * action_text : the action title
    * tool_tip : a tip that is shown when the user hovers over the icon
    * shortcut : keyboard shortcut to toggle the mode
    """

    icon = None
    mode_id = None
    action_text = None
    tool_tip = None
    shortcut = None

    def __init__(self, viewer=None):
        self.viewer = viewer
        self.viewer.window_closed.connect(nonpartial(self.close))

    def menu_actions(self):
        """
        List of QtWidgets.QActions to be attached to this mode
        as a context menu.
        """
        return []

    def close(self):
        pass


class NonCheckableMode(Mode):
    """
    A mode that is not checkable.

    When clicked, the ``activate`` method is executed.
    """

    def activate(self):
        """
        Fired when the toolbar button is activated
        """
        pass


class CheckableMode(Mode):
    """
    A mode that is checkable.

    When checked, the ``activate`` method is executed, and when unchecked, the
    ``deactivate`` method is executed.
    """

    def activate(self):
        """
        Fired when the toolbar button is activated
        """
        pass

    def deactivate(self):
        """
        Fired when the toolbar button is deactivated
        """
        pass
