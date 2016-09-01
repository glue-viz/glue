# The classes in this file define toolbar tools. Mouse modes specifically
# are defined in mouse_modes.py

from glue.utils import nonpartial
from glue.core.callback_property import CallbackProperty

__all__ = ['Tool', 'CheckableTool']


class Tool(object):
    """
    The base class for all toolbar tools.

    All tools have the following attributes:

    * icon : QIcon object
    * tool_id : a short name for the tool
    * action_text : the action title (used if the tool is made available in a menu)
    * tool_tip : a tip that is shown when the user hovers over the icon
    * status_tip : a tip that is shown in the status bar when the tool is active
    * shortcut : keyboard shortcut to toggle the tool
    """

    enabled = CallbackProperty(True)

    icon = None
    tool_id = None
    action_text = None
    tool_tip = None
    status_tip = None
    shortcut = None

    def __init__(self, viewer=None):
        self.viewer = viewer
        self.viewer.window_closed.connect(nonpartial(self.close))

    def activate(self):
        """
        Fired when the toolbar button is activated
        """
        pass

    def menu_actions(self):
        """
        List of QtWidgets.QActions to be attached to this tool
        as a context menu.
        """
        return []

    def close(self):
        pass



class CheckableTool(Tool):
    """
    A tool that is checkable.

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
