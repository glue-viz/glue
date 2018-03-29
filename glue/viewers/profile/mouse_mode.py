from glue.external.echo import CallbackProperty
from glue.core.state_objects import State
from glue.viewers.common.qt.mouse_mode import MouseMode


class NavigationModeState(State):
    x = CallbackProperty(None)


class NavigateMouseMode(MouseMode):

    def __init__(self, viewer):
        super(NavigateMouseMode, self).__init__(viewer)
        self.state = NavigationModeState()
        self.state.add_callback('x', self._update_line)
        self.pressed = False
        self._viewer = viewer

    def press(self, event):
        self.pressed = True
        if not event.inaxes:
            return
        self.state.x = event.xdata

    def move(self, event):
        if not self.pressed or not event.inaxes:
            return
        self.state.x = event.xdata

    def release(self, event):
        self.pressed = False

    def _update_line(self, *args):
        if hasattr(self, '_line'):
            self._line.set_data([self.state.x, self.state.x], [0, 1])
        else:
            self._line = self._axes.axvline(0)
        self._canvas.draw()
