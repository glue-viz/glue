from __future__ import absolute_import, division, print_function

from glue.external.echo import CallbackProperty, delay_callback
from glue.core.state_objects import State
from glue.viewers.common.qt.mouse_mode import MouseMode

__all__ = ['NavigateMouseMode', 'RangeMouseMode']


COLOR = (0.0, 0.25, 0.7)


class NavigationModeState(State):
    x = CallbackProperty(None)


class NavigateMouseMode(MouseMode):

    def __init__(self, viewer, press_callback=None):
        super(NavigateMouseMode, self).__init__(viewer)
        self.state = NavigationModeState()
        self.state.add_callback('x', self._update_artist)
        self.pressed = False
        self.active = False
        self._press_callback = press_callback

    def press(self, event):
        if not self.active or not event.inaxes:
            return
        if self._press_callback is not None:
            self._press_callback()
        self.pressed = True
        self.state.x = event.xdata

    def move(self, event):
        if not self.active or not self.pressed or not event.inaxes:
            return
        self.state.x = event.xdata

    def release(self, event):
        if not self.active:
            return
        self.pressed = False

    def _update_artist(self, *args):
        if hasattr(self, '_line'):
            if self.state.x is None:
                self._line.set_visible(False)
            else:
                self._line.set_data([self.state.x, self.state.x], [0, 1])
        else:
            if self.state.x is not None:
                self._line = self._axes.axvline(self.state.x, color=COLOR)
        self._canvas.draw()

    def deactivate(self):
        if hasattr(self, '_line'):
            self._line.set_visible(False)
        self._canvas.draw()
        super(NavigateMouseMode, self).deactivate()
        self.active = False

    def activate(self):
        if hasattr(self, '_line'):
            self._line.set_visible(True)
        self._canvas.draw()
        super(NavigateMouseMode, self).activate()
        self.active = True

    def clear(self):
        self.state.x = None


class RangeModeState(State):

    x_min = CallbackProperty(None)
    x_max = CallbackProperty(None)

    @property
    def x_range(self):
        return self.x_min, self.x_max


PICK_THRESH = 0.02


class RangeMouseMode(MouseMode):

    def __init__(self, viewer):
        super(RangeMouseMode, self).__init__(viewer)
        self.state = RangeModeState()
        self.state.add_callback('x_min', self._update_artist)
        self.state.add_callback('x_max', self._update_artist)
        self.pressed = False

        self.mode = None
        self.move_params = None

        self.active = False

    def press(self, event):

        if not self.active or not event.inaxes:
            return

        self.pressed = True

        x_min, x_max = self._axes.get_xlim()
        x_range = abs(x_max - x_min)

        if self.state.x_min is None or self.state.x_max is None:
            self.mode = 'move-x-max'
            with delay_callback(self.state, 'x_min', 'x_max'):
                self.state.x_min = event.xdata
                self.state.x_max = event.xdata
        elif abs(event.xdata - self.state.x_min) / x_range < PICK_THRESH:
            self.mode = 'move-x-min'
        elif abs(event.xdata - self.state.x_max) / x_range < PICK_THRESH:
            self.mode = 'move-x-max'
        elif (event.xdata > self.state.x_min) is (event.xdata < self.state.x_max):
            self.mode = 'move'
            self.move_params = (event.xdata, self.state.x_min, self.state.x_max)
        else:
            self.mode = 'move-x-max'
            self.state.x_min = event.xdata

    def move(self, event):

        if not self.active or not self.pressed or not event.inaxes:
            return

        if self.mode == 'move-x-min':
            self.state.x_min = event.xdata
        elif self.mode == 'move-x-max':
            self.state.x_max = event.xdata
        elif self.mode == 'move':
            orig_click, orig_x_min, orig_x_max = self.move_params
            with delay_callback(self.state, 'x_min', 'x_max'):
                self.state.x_min = orig_x_min + (event.xdata - orig_click)
                self.state.x_max = orig_x_max + (event.xdata - orig_click)

    def release(self, event):
        if not self.active:
            return
        self.pressed = False
        self.mode = None
        self.move_params

    def _update_artist(self, *args):
        y_min, y_max = self._axes.get_ylim()
        if hasattr(self, '_lines'):
            if self.state.x_min is None or self.state.x_max is None:
                self._lines[0].set_visible(False)
                self._lines[1].set_visible(False)
                self._interval.set_visible(False)
            else:
                self._lines[0].set_data([self.state.x_min, self.state.x_min], [0, 1])
                self._lines[1].set_data([self.state.x_max, self.state.x_max], [0, 1])
                self._interval.set_xy([[self.state.x_min, 0],
                                       [self.state.x_min, 1],
                                       [self.state.x_max, 1],
                                       [self.state.x_max, 0],
                                       [self.state.x_min, 0]])
        else:
            if self.state.x_min is not None and self.state.x_max is not None:
                self._lines = (self._axes.axvline(self.state.x_min, color=COLOR),
                               self._axes.axvline(self.state.x_max, color=COLOR))
                self._interval = self._axes.axvspan(self.state.x_min,
                                                    self.state.x_max,
                                                    color=COLOR, alpha=0.05)
        self._canvas.draw()

    def deactivate(self):
        if hasattr(self, '_lines'):
            self._lines[0].set_visible(False)
            self._lines[1].set_visible(False)
            self._interval.set_visible(False)

        self._canvas.draw()
        super(RangeMouseMode, self).deactivate()
        self.active = False

    def activate(self):
        if hasattr(self, '_lines'):
            self._lines[0].set_visible(True)
            self._lines[1].set_visible(True)
            self._interval.set_visible(True)
        self._canvas.draw()
        super(RangeMouseMode, self).activate()
        self.active = True

    def clear(self):
        with delay_callback(self.state, 'x_min', 'x_max'):
            self.state.x_min = None
            self.state.x_max = None
