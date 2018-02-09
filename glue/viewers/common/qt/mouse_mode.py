"""
MouseModes define various mouse gestures.

MouseModes are generally activated and deactivated by toolbar buttons, although
not necessarily so. The toolbar maintains a list of MouseModes from the
visualization it is assigned to, and sees to it that only one MouseMode is
active at a time.

Each MouseMode appears as an Icon in the toolbar. Classes can assign methods to
the press_callback, move_callback, and release_callback methods of each Mouse
Mode, to implement custom functionality

The basic usage pattern is thus:
 * visualization object instantiates the MouseModes it wants
 * each of these is passed to the add_tool method of the toolbar
 * visualization object optionally attaches methods to the 3 _callback
   methods in a MouseMode, for additional behavior
"""

from __future__ import absolute_import, division, print_function


__all__ = ['MouseMode']


class MouseMode(object):
    """
    The base class for all MouseModes.

    MouseModes have the following attributes:

    * press_callback : Callback method that will be called
      whenever a MouseMode processes a mouse press event
    * move_callback : Same as above, for move events
    * release_callback : Same as above, for release events
    * key_callback : Same as above, for release events

    The _callback hooks are called with the MouseMode as its only
    argument
    """

    def __init__(self, viewer,
                 press_callback=None,
                 move_callback=None,
                 release_callback=None,
                 key_callback=None):

        self._axes = viewer.axes
        self._canvas = viewer.central_widget.canvas
        self._press_callback = press_callback
        self._move_callback = move_callback
        self._release_callback = release_callback
        self._key_callback = key_callback
        self._event_x = None
        self._event_y = None
        self._event_xdata = None
        self._event_ydata = None
        self._connections = []

    def _log_position(self, event):
        if event is None:
            return
        self._event_x, self._event_y = event.x, event.y
        self._event_xdata, self._event_ydata = event.xdata, event.ydata

    def press(self, event):
        """
        Handles mouse presses.

        Logs mouse position and calls press_callback method.

        Parameters
        ----------
        event : :class:`~matplotlib.backend_bases.MouseEvent`
            The event that was triggered
        """
        self._log_position(event)
        if self._press_callback is not None:
            self._press_callback(self)

    def move(self, event):
        """
        Handles mouse move events.

        Logs mouse position and calls move_callback method.

        Parameters
        ----------
        event : :class:`~matplotlib.backend_bases.MouseEvent`
            The event that was triggered
        """
        self._log_position(event)
        if self._move_callback is not None:
            self._move_callback(self)

    def release(self, event):
        """
        Handles mouse release events.

        Logs mouse position and calls release_callback method.

        Parameters
        ----------
        event : :class:`~matplotlib.backend_bases.MouseEvent`
            The event that was triggered
        """
        self._log_position(event)
        if self._release_callback is not None:
            self._release_callback(self)

    def key(self, event):
        """
        Handles key press events.

        Calls key_callback method.

        Parameters
        ----------
        event : :class:`~matplotlib.backend_bases.KeyEvent`
            The event that was triggered
        """
        if self._key_callback is not None:
            self._key_callback(self)

    def activate(self):
        """
        Activates all MPL event handlers associated with this mouse mode.
        """
        self._connections.append(self._canvas.mpl_connect('button_press_event', self.press))
        self._connections.append(self._canvas.mpl_connect('motion_notify_event', self.move))
        self._connections.append(self._canvas.mpl_connect('button_release_event', self.release))
        self._connections.append(self._canvas.mpl_connect('key_press_event', self.key))

    def deactivate(self):
        """
        Deactivates all MPL event handlers associated with this mouse mode.
        """
        for connection in self._connections:
            self._canvas.mpl_disconnect(connection)
        self._connections = []
