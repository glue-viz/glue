from __future__ import absolute_import, division, print_function

import os

from functools import partial
from collections import Counter

import numpy as np

from glue.core import Coordinates
from qtpy import QtCore, QtWidgets
from glue.utils.qt import load_ui
from glue.utils.qt.widget_properties import (TextProperty,
                                             ButtonProperty,
                                             ValueProperty,
                                             CurrentComboProperty)
from glue.utils import nonpartial
from glue.icons.qt import get_icon


class SliceWidget(QtWidgets.QWidget):

    label = TextProperty('_ui_label')
    slider_label = TextProperty('_ui_slider.label')
    slider_unit = TextProperty('_ui_slider.text_unit')
    slice_center = ValueProperty('_ui_slider.slider')
    mode = CurrentComboProperty('_ui_mode')
    use_world = ButtonProperty('_ui_slider.checkbox_world')

    slice_changed = QtCore.Signal(int)
    mode_changed = QtCore.Signal(str)

    def __init__(self, label='', world=None, lo=0, hi=10,
                 parent=None, aggregation=None, world_unit=None,
                 world_warning=False):

        super(SliceWidget, self).__init__(parent)

        if aggregation is not None:
            raise NotImplemented("Aggregation option not implemented")

        self._world = np.asarray(world)
        self._world_warning = world_warning
        self._world_unit = world_unit

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(3, 1, 3, 1)
        layout.setSpacing(0)

        top = QtWidgets.QHBoxLayout()
        top.setContentsMargins(3, 3, 3, 3)
        label = QtWidgets.QLabel(label)
        top.addWidget(label)

        mode = QtWidgets.QComboBox()
        mode.addItem('x', 'x')
        mode.addItem('y', 'y')
        mode.addItem('slice', 'slice')
        mode.currentIndexChanged.connect(lambda x:
                                         self.mode_changed.emit(self.mode))
        mode.currentIndexChanged.connect(self._update_mode)
        top.addWidget(mode)

        layout.addLayout(top)

        slider = load_ui('data_slice_widget.ui', None,
                         directory=os.path.dirname(__file__))
        self._ui_slider = slider

        slider.button_first.setStyleSheet('border: 0px')
        slider.button_first.setIcon(get_icon('playback_first'))
        slider.button_prev.setStyleSheet('border: 0px')
        slider.button_prev.setIcon(get_icon('playback_prev'))
        slider.button_back.setStyleSheet('border: 0px')
        slider.button_back.setIcon(get_icon('playback_back'))
        slider.button_stop.setStyleSheet('border: 0px')
        slider.button_stop.setIcon(get_icon('playback_stop'))
        slider.button_forw.setStyleSheet('border: 0px')
        slider.button_forw.setIcon(get_icon('playback_forw'))
        slider.button_next.setStyleSheet('border: 0px')
        slider.button_next.setIcon(get_icon('playback_next'))
        slider.button_last.setStyleSheet('border: 0px')
        slider.button_last.setIcon(get_icon('playback_last'))

        slider.slider.setMinimum(lo)
        slider.slider.setMaximum(hi)
        slider.slider.setValue((lo + hi) / 2)
        slider.slider.valueChanged.connect(lambda x:
                                           self.slice_changed.emit(self.mode))
        slider.slider.valueChanged.connect(nonpartial(self.set_label_from_slider))

        slider.label.setMinimumWidth(80)
        slider.label.setText(str(slider.slider.value()))
        slider.label.editingFinished.connect(nonpartial(self.set_slider_from_label))

        self._play_timer = QtCore.QTimer()
        self._play_timer.setInterval(500)
        self._play_timer.timeout.connect(nonpartial(self._play_slice))

        slider.button_first.clicked.connect(nonpartial(self._browse_slice, 'first'))
        slider.button_prev.clicked.connect(nonpartial(self._browse_slice, 'prev'))
        slider.button_back.clicked.connect(nonpartial(self._adjust_play, 'back'))
        slider.button_stop.clicked.connect(nonpartial(self._adjust_play, 'stop'))
        slider.button_forw.clicked.connect(nonpartial(self._adjust_play, 'forw'))
        slider.button_next.clicked.connect(nonpartial(self._browse_slice, 'next'))
        slider.button_last.clicked.connect(nonpartial(self._browse_slice, 'last'))

        slider.checkbox_world.toggled.connect(nonpartial(self.set_label_from_slider))

        if world is None:
            self.use_world = False
            slider.checkbox_world.hide()
        else:
            self.use_world = not world_warning

        if world_unit:
            self.slider_unit = world_unit
        else:
            self.slider_unit = ''

        layout.addWidget(slider)

        self.setLayout(layout)

        self._ui_label = label
        self._ui_mode = mode
        self._update_mode()
        self._frozen = False

        self._play_speed = 0

        self.set_label_from_slider()

    def set_label_from_slider(self):
        value = self._ui_slider.slider.value()
        if self.use_world:
            text = str(self._world[value])
            if self._world_warning:
                self._ui_slider.label_warning.show()
            else:
                self._ui_slider.label_warning.hide()
            self.slider_unit = self._world_unit
        else:
            text = str(value)
            self._ui_slider.label_warning.hide()
            self.slider_unit = ''
        self._ui_slider.label.setText(text)

    def set_slider_from_label(self):
        text = self._ui_slider.label.text()
        if self.use_world:
            # Don't want to assume world is sorted, pick closest value
            value = np.argmin(np.abs(self._world - float(text)))
            self._ui_slider.label.setText(str(self._world[value]))
        else:
            value = int(text)
        self._ui_slider.slider.setValue(value)

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

        imin = self._ui_slider.slider.minimum()
        imax = self._ui_slider.slider.maximum()
        value = self._ui_slider.slider.value()

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

        self._ui_slider.slider.setValue(value)

    def _update_mode(self, *args):
        if self.mode != 'slice':
            self._ui_slider.hide()
            self._adjust_play('stop')
        else:
            self._ui_slider.show()

    def freeze(self):
        self.mode = 'slice'
        self._ui_mode.setEnabled(False)
        self._ui_slider.hide()
        self._frozen = True

    @property
    def frozen(self):
        return self._frozen


class DataSlice(QtWidgets.QWidget):

    """
    A DatSlice widget provides an inteface for selection
    slices through an N-dimensional dataset

    QtCore.Signals
    -------
    slice_changed : triggered when the slice through the data changes
    """
    slice_changed = QtCore.Signal()

    def __init__(self, data=None, parent=None):
        """
        :param data: :class:`~glue.core.data.Data` instance, or None
        """
        super(DataSlice, self).__init__(parent)
        self._slices = []
        self._data = None

        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(4)
        layout.setContentsMargins(0, 3, 0, 3)
        self.layout = layout
        self.setLayout(layout)
        self.set_data(data)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def shape(self):
        return tuple() if self._data is None else self._data.shape

    def _clear(self):
        for _ in range(self.layout.count()):
            self.layout.takeAt(0)

        for s in self._slices:
            s.close()

        self._slices = []

    def set_data(self, data):
        """
        Change datasets

        :parm data: :class:`~glue.core.data.Data` instance
        """

        # remove old widgets
        self._clear()

        self._data = data

        if data is None or data.ndim < 3:
            return

        # create slider widget for each dimension...
        for i, s in enumerate(data.shape):

            # TODO: For now we simply pass a single set of world coordinates,
            # but we will need to generalize this in future. We deliberately
            # check the type of data.coords here since we want to treat
            # subclasses differently.
            if type(data.coords) != Coordinates:
                world = data.coords.world_axis(data, i)
                world_unit = data.coords.world_axis_unit(i)
                world_warning = len(data.coords.dependent_axes(i)) > 1
            else:
                world = None
                world_unit = None
                world_warning = False

            slider = SliceWidget(data.get_world_component_id(i).label,
                                 hi=s - 1, world=world, world_unit=world_unit,
                                 world_warning=world_warning)

            if i == self.ndim - 1:
                slider.mode = 'x'
            elif i == self.ndim - 2:
                slider.mode = 'y'
            else:
                slider.mode = 'slice'
            self._slices.append(slider)

            # save ref to prevent PySide segfault
            self.__on_slice = partial(self._on_slice, i)
            self.__on_mode = partial(self._on_mode, i)

            slider.slice_changed.connect(self.__on_slice)
            slider.mode_changed.connect(self.__on_mode)
            if s == 1:
                slider.freeze()

        # ... and add to the layout
        for s in self._slices[::-1]:
            self.layout.addWidget(s)
            if s is not self._slices[0]:
                line = QtWidgets.QFrame()
                line.setFrameShape(QtWidgets.QFrame.HLine)
                line.setFrameShadow(QtWidgets.QFrame.Sunken)
                self.layout.addWidget(line)
            s.show()  # this somehow fixes #342

        self.layout.addStretch(5)

    def _on_slice(self, index, slice_val):
        self.slice_changed.emit()

    def _on_mode(self, index, mode_index):
        s = self.slice

        def isok(ss):
            # valid slice description: 'x' and 'y' both appear
            c = Counter(ss)
            return c['x'] == 1 and c['y'] == 1

        if isok(s):
            self.slice_changed.emit()
            return

        for i in range(len(s)):
            if i == index:
                continue
            if self._slices[i].frozen:
                continue

            for mode in 'x', 'y', 'slice':
                if self._slices[i].mode == mode:
                    continue

                ss = list(s)
                ss[i] = mode
                if isok(ss):
                    self._slices[i].mode = mode
                    return

        else:
            raise RuntimeError("Corrupted Data Slice")

    @property
    def slice(self):
        """
        A description of the slice through the dataset

        A tuple of lenght equal to the dimensionality of the data

        Each element is an integer, 'x', or 'y'
        'x' and 'y' indicate the horizontal and vertical orientation
        of the slice
        """
        if self.ndim < 3:
            return {0: tuple(), 1: ('x',), 2: ('y', 'x')}[self.ndim]

        return tuple(s.mode if s.mode != 'slice' else s.slice_center
                     for s in self._slices)

    @slice.setter
    def slice(self, value):
        for v, s in zip(value, self._slices):
            if v in ['x', 'y']:
                s.mode = v
            else:
                s.mode = 'slice'
                s.slice_center = v

if __name__ == "__main__":

    from glue.utils.qt import get_qapp

    app = get_qapp()

    widget = SliceWidget()
    widget.show()

    app.exec_()
