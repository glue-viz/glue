from functools import partial
from ...compat.collections import Counter

from ...external.qt.QtGui import (QWidget, QSlider, QLabel, QComboBox,
                                  QHBoxLayout, QVBoxLayout, QPushButton, QLineEdit)
from ...external.qt.QtCore import Qt, Signal

from ..widget_properties import (TextProperty,
                                 ValueProperty,
                                 CurrentComboProperty)
from ..qtutil import nonpartial


class SliceWidget(QWidget):
    label = TextProperty('_ui_label')
    slice_center = ValueProperty('_ui_slider')
    mode = CurrentComboProperty('_ui_mode')

    slice_changed = Signal(int)
    mode_changed = Signal(str)

    def __init__(self, label='', pix2world=None, lo=0, hi=10,
                 parent=None, aggregation=None):
        super(SliceWidget, self).__init__(parent)
        if aggregation is not None:
            raise NotImplemented("Aggregation option not implemented")
        if pix2world is not None:
            raise NotImplemented("Pix2world option not implemented")

        layout = QVBoxLayout()
        layout.setContentsMargins(3, 1, 3, 1)

        top = QHBoxLayout()
        top.setContentsMargins(3, 3, 3, 3)
        label = QLabel(label)
        top.addWidget(label)

        mode = QComboBox()
        mode.addItem('x', 'x')
        mode.addItem('y', 'y')
        mode.addItem('slice', 'slice')
        mode.currentIndexChanged.connect(lambda x:
                                         self.mode_changed.emit(self.mode))
        mode.currentIndexChanged.connect(self._update_mode)
        top.addWidget(mode)

        layout.addLayout(top)

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(lo)
        slider.setMaximum(hi)
        slider.setValue((lo + hi) / 2)
        slider.valueChanged.connect(lambda x:
                                    self.slice_changed.emit(self.mode))

        slider_lbl = QLineEdit()
        slider_lbl.setMinimumWidth(50)
        slider_lbl.setText(str(slider.value()))
        slider.valueChanged.connect(lambda x: slider_lbl.setText(str(x)))
        slider_lbl.textChanged.connect(lambda x: slider.setValue(int(x)))

        browser = {}
        browser['first'] = QPushButton("<<")
        browser['prev'] = QPushButton("<")
        browser['label'] = slider_lbl
        browser['next'] = QPushButton(">")
        browser['last'] = QPushButton(">>")

        slider_browser = QHBoxLayout()
        for key in ['first', 'prev', 'label', 'next', 'last']:
            slider_browser.addWidget(browser[key])
            if key == 'label':
                pass
            else:
                browser[key].clicked.connect(nonpartial(self._browse_slice, key))

        layout.addLayout(slider_browser)
        layout.addWidget(slider)

        self.setLayout(layout)

        self._ui_label = label
        self._ui_slider = slider
        self._ui_slider_browser = browser
        self._ui_mode = mode
        self._update_mode()
        self._frozen = False

    def _browse_slice(self, action):
        imin = self._ui_slider.minimum()
        imax = self._ui_slider.maximum()
        value = self._ui_slider.value()
        if action == 'first':
            value = imin
        elif action == 'last':
            value = imax
        elif action == 'prev':
            value = max(value - 1, imin)
        elif action == 'next':
            value = min(value + 1, imax)
        self._ui_slider.setValue(value)

    def _update_mode(self, *args):
        if self.mode != 'slice':
            self._ui_slider.hide()
            for key in self._ui_slider_browser:
                self._ui_slider_browser[key].hide()
        else:
            self._ui_slider.show()
            for key in self._ui_slider_browser:
                self._ui_slider_browser[key].show()

    def freeze(self):
        self.mode = 'slice'
        self._ui_mode.setEnabled(False)
        self._ui_slider.hide()
        self._frozen = True

    @property
    def frozen(self):
        return self._frozen


class DataSlice(QWidget):

    """
    A DatSlice widget provides an inteface for selection
    slices through an N-dimensional dataset

    Signals
    -------
    slice_changed : triggered when the slice through the data changes
    """
    slice_changed = Signal()

    def __init__(self, data=None, parent=None):
        """
        :param data: :class:`~glue.core.data.Data` instance, or None
        """
        super(DataSlice, self).__init__(parent)
        self._slices = []
        self._data = None

        layout = QVBoxLayout()
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
            slider = SliceWidget(data.get_world_component_id(i).label,
                                 hi=s - 1)

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
