import os

import numpy as np

from qtpy import QtCore, QtWidgets
from glue.utils.qt import load_ui
from glue.utils import nonpartial, format_minimal
from glue.icons.qt import get_icon
from glue.core.state_objects import State, CallbackProperty
from echo.qt import autoconnect_callbacks_to_qt


class SliceState(State):

    label = CallbackProperty()
    slider_label = CallbackProperty()
    slider_unit = CallbackProperty()
    slice_center = CallbackProperty()
    use_world = CallbackProperty()


class SliceWidget(QtWidgets.QWidget):

    slice_changed = QtCore.Signal(int)

    def __init__(self, label='', world=None, lo=0, hi=10,
                 parent=None, world_unit=None,
                 world_warning=False):

        super(SliceWidget, self).__init__(parent)

        self.state = SliceState()
        self.state.label = label
        self.state.slice_center = (lo + hi) // 2

        self._world = np.asarray(world)
        self._world_warning = world_warning
        self._world_unit = world_unit

        self.ui = load_ui('data_slice_widget.ui', self,
                          directory=os.path.dirname(__file__))

        self._connections = autoconnect_callbacks_to_qt(self.state, self.ui)

        font = self.text_warning.font()
        font.setPointSize(int(font.pointSize() * 0.75))
        self.text_warning.setFont(font)

        self.button_first.setStyleSheet('border: 0px')
        self.button_first.setIcon(get_icon('playback_first'))
        self.button_prev.setStyleSheet('border: 0px')
        self.button_prev.setIcon(get_icon('playback_prev'))
        self.button_back.setStyleSheet('border: 0px')
        self.button_back.setIcon(get_icon('playback_back'))
        self.button_stop.setStyleSheet('border: 0px')
        self.button_stop.setIcon(get_icon('playback_stop'))
        self.button_forw.setStyleSheet('border: 0px')
        self.button_forw.setIcon(get_icon('playback_forw'))
        self.button_next.setStyleSheet('border: 0px')
        self.button_next.setIcon(get_icon('playback_next'))
        self.button_last.setStyleSheet('border: 0px')
        self.button_last.setIcon(get_icon('playback_last'))

        self.value_slice_center.setMinimum(lo)
        self.value_slice_center.setMaximum(hi)
        self.value_slice_center.valueChanged.connect(nonpartial(self.set_label_from_slider))

        # Figure out the optimal format to use to show the world values. We do
        # this by figuring out the precision needed so that when converted to
        # a string, every string value is different.

        if world is not None and len(world) > 1:
            self.label_fmt = format_minimal(world)[0]
        else:
            self.label_fmt = "{:g}"

        self.text_slider_label.setMinimumWidth(80)
        self.state.slider_label = self.label_fmt.format(self.value_slice_center.value())
        self.text_slider_label.editingFinished.connect(nonpartial(self.set_slider_from_label))

        self._play_timer = QtCore.QTimer()
        self._play_timer.setInterval(500)
        self._play_timer.timeout.connect(nonpartial(self._play_slice))

        self.button_first.clicked.connect(nonpartial(self._browse_slice, 'first'))
        self.button_prev.clicked.connect(nonpartial(self._browse_slice, 'prev'))
        self.button_back.clicked.connect(nonpartial(self._adjust_play, 'back'))
        self.button_stop.clicked.connect(nonpartial(self._adjust_play, 'stop'))
        self.button_forw.clicked.connect(nonpartial(self._adjust_play, 'forw'))
        self.button_next.clicked.connect(nonpartial(self._browse_slice, 'next'))
        self.button_last.clicked.connect(nonpartial(self._browse_slice, 'last'))

        self.bool_use_world.toggled.connect(nonpartial(self.set_label_from_slider))

        if world is None:
            self.state.use_world = False
            self.bool_use_world.hide()
        else:
            self.state.use_world = not world_warning

        if world_unit:
            self.state.slider_unit = world_unit
        else:
            self.state.slider_unit = ''

        self._play_speed = 0

        self.set_label_from_slider()

    def set_label_from_slider(self):
        value = self.state.slice_center
        if self.state.use_world:
            value = self._world[value]
            if self._world_warning:
                self.text_warning.show()
            else:
                self.text_warning.hide()
            self.state.slider_unit = self._world_unit
            self.state.slider_label = self.label_fmt.format(value)
        else:
            self.text_warning.hide()
            self.state.slider_unit = ''
            self.state.slider_label = str(value)

    def set_slider_from_label(self):

        # Ignore recursive calls - we do this rather than ignore_callback
        # below when setting slider_label, otherwise we might be stopping other
        # subscribers to that event from being correctly updated
        if getattr(self, '_in_set_slider_from_label', False):
            return
        else:
            self._in_set_slider_from_label = True

        text = self.text_slider_label.text()
        if self.state.use_world:
            # Don't want to assume world is sorted, pick closest value
            value = np.argmin(np.abs(self._world - float(text)))
            self.state.slider_label = self.label_fmt.format(self._world[value])
        else:
            value = int(text)
        self.value_slice_center.setValue(value)

        self._in_set_slider_from_label = False

    def _adjust_play(self, action):
        if action == 'stop':
            self._play_speed = 0
        elif action == 'back':
            if self._play_speed > 0:
                self._play_speed = -1
            else:
                self._play_speed -= 1
        elif action == 'forw':
            if self._play_speed < 0:
                self._play_speed = +1
            else:
                self._play_speed += 1
        if self._play_speed == 0:
            self._play_timer.stop()
        else:
            self._play_timer.start()
            self._play_timer.setInterval(500 / abs(self._play_speed))

    def _play_slice(self):
        if self._play_speed > 0:
            self._browse_slice('next', play=True)
        elif self._play_speed < 0:
            self._browse_slice('prev', play=True)

    def _browse_slice(self, action, play=False):

        imin = self.value_slice_center.minimum()
        imax = self.value_slice_center.maximum()
        value = self.value_slice_center.value()

        # If this was not called from _play_slice, we should stop the
        # animation.
        if not play:
            self._adjust_play('stop')

        if action == 'first':
            value = imin
        elif action == 'last':
            value = imax
        elif action == 'prev':
            value = value - 1
            if value < imin:
                value = imax
        elif action == 'next':
            value = value + 1
            if value > imax:
                value = imin
        else:
            raise ValueError("Action should be one of first/prev/next/last")

        self.value_slice_center.setValue(value)


if __name__ == "__main__":

    from glue.utils.qt import get_qapp

    app = get_qapp()

    widget = SliceWidget(label='BANANA')
    widget.show()

    widget = SliceWidget(world=[1, 2, 3, 4, 5, 6, 7], lo=1, hi=7)
    widget.show()

    app.exec_()
