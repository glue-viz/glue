# The classes in this file define toolbar modes. Mouse modes specifically
# are defined in mouse_modes.py

__all__ = ['Mode', 'NonCheckableMode', 'CheckableMode']


class Mode(object):

    icon = None
    mode_id = None
    action_text = None
    tool_tip = None
    shortcut = None

    def menu_actions(self):
        return []


class NonCheckableMode(Mode):
    """
    A mode that is not checkable.

    When clicked, the ``activate`` method is executed.
    """

    def activate(self):
        pass


class CheckableMode(Mode):
    """
    A mode that is checkable.

    When checked, the ``activate`` method is executed, and when unchecked, the
    ``deactivate`` method is executed.
    """

    def activate(self):
        pass

    def defactivate(self):
        pass
