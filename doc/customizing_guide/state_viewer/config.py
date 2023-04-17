import os

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from qtpy.QtWidgets import QWidget, QVBoxLayout, QCheckBox

from glue.config import qt_client
from glue.core.data_combo_helper import ComponentIDComboHelper

from echo import CallbackProperty, SelectionCallbackProperty
from echo.qt import connect_checkable_button, autoconnect_callbacks_to_qt

from glue.viewers.common.layer_artist import LayerArtist
from glue.viewers.common.state import ViewerState, LayerState
from glue.viewers.common.qt.data_viewer import DataViewer

from glue.utils.qt import load_ui


class TutorialViewerState(ViewerState):

    x_att = SelectionCallbackProperty(docstring='The attribute to use on the x-axis')
    y_att = SelectionCallbackProperty(docstring='The attribute to use on the y-axis')

    def __init__(self, *args, **kwargs):
        super(TutorialViewerState, self).__init__(*args, **kwargs)
        self._x_att_helper = ComponentIDComboHelper(self, 'x_att')
        self._y_att_helper = ComponentIDComboHelper(self, 'y_att')
        self.add_callback('layers', self._on_layers_change)

    def _on_layers_change(self, value):
        # self.layers_data is a shortcut for
        # [layer_state.layer for layer_state in self.layers]
        self._x_att_helper.set_multiple_data(self.layers_data)
        self._y_att_helper.set_multiple_data(self.layers_data)


class TutorialLayerState(LayerState):
    fill = CallbackProperty(False, docstring='Whether to show the markers as filled or not')


class TutorialLayerArtist(LayerArtist):

    _layer_state_cls = TutorialLayerState

    def __init__(self, axes, *args, **kwargs):

        super(TutorialLayerArtist, self).__init__(*args, **kwargs)

        self.axes = axes

        self.artist = self.axes.plot([], [], 'o', color=self.state.layer.style.color)[0]

        self.state.add_callback('fill', self._on_fill_change)
        self.state.add_callback('visible', self._on_visible_change)
        self.state.add_callback('zorder', self._on_zorder_change)

        self._viewer_state.add_callback('x_att', self._on_attribute_change)
        self._viewer_state.add_callback('y_att', self._on_attribute_change)

    def _on_fill_change(self, value=None):
        if self.state.fill:
            self.artist.set_markerfacecolor(self.state.layer.style.color)
        else:
            self.artist.set_markerfacecolor('none')
        self.redraw()

    def _on_visible_change(self, value=None):
        self.artist.set_visible(self.state.visible)
        self.redraw()

    def _on_zorder_change(self, value=None):
        self.artist.set_zorder(self.state.zorder)
        self.redraw()

    def _on_attribute_change(self, value=None):

        if self._viewer_state.x_att is None or self._viewer_state.y_att is None:
            return

        x = self.state.layer[self._viewer_state.x_att]
        y = self.state.layer[self._viewer_state.y_att]

        self.artist.set_data(x, y)

        self.axes.set_xlim(np.nanmin(x), np.nanmax(x))
        self.axes.set_ylim(np.nanmin(y), np.nanmax(y))

        self.redraw()

    def clear(self):
        self.artist.set_visible(False)

    def remove(self):
        self.artist.remove()

    def redraw(self):
        self.axes.figure.canvas.draw_idle()

    def update(self):
        self._on_fill_change()
        self._on_attribute_change()


class TutorialViewerStateWidget(QWidget):

    def __init__(self, viewer_state=None, session=None):

        super(TutorialViewerStateWidget, self).__init__()

        self.ui = load_ui('viewer_state.ui', self,
                          directory=os.path.dirname(__file__))

        self.viewer_state = viewer_state
        self._connections = autoconnect_callbacks_to_qt(self.viewer_state, self.ui)


class TutorialLayerStateWidget(QWidget):

    def __init__(self, layer_artist):

        super(TutorialLayerStateWidget, self).__init__()

        self.checkbox = QCheckBox('Fill markers')
        layout = QVBoxLayout()
        layout.addWidget(self.checkbox)
        self.setLayout(layout)

        self.layer_state = layer_artist.state
        self._connection = connect_checkable_button(self.layer_state, 'fill', self.checkbox)


class TutorialDataViewer(DataViewer):

    LABEL = 'Tutorial viewer'
    _state_cls = TutorialViewerState
    _options_cls = TutorialViewerStateWidget
    _layer_style_widget_cls = TutorialLayerStateWidget
    _data_artist_cls = TutorialLayerArtist
    _subset_artist_cls = TutorialLayerArtist

    def __init__(self, *args, **kwargs):
        super(TutorialDataViewer, self).__init__(*args, **kwargs)
        self.axes = plt.subplot(1, 1, 1)
        self.setCentralWidget(self.axes.figure.canvas)

    def get_layer_artist(self, cls, layer=None, layer_state=None):
        return cls(self.axes, self.state, layer=layer, layer_state=layer_state)


qt_client.add(TutorialDataViewer)
