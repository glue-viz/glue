from __future__ import absolute_import, division, print_function

import os
from collections import OrderedDict

from qtpy import QtWidgets

from glue.external.echo.qt import autoconnect_callbacks_to_qt
from glue.utils.qt import load_ui, update_combobox
from glue.utils.qt.widget_properties import CurrentComboDataProperty
from glue.core.qt.data_combo_helper import ComponentIDComboHelper
from glue.external.echo import add_callback
from glue.viewers.scatter_new.state import ScatterLayerState
from glue.utils import nonpartial


class ScatterLayerStyleEditor(QtWidgets.QWidget):

    def __init__(self, layer, parent=None):

        super(ScatterLayerStyleEditor, self).__init__(parent=parent)

        self.ui = load_ui('scatter_style_editor.ui', self,
                          directory=os.path.dirname(__file__))

        # TODO: layer_state should not have data_collection attribute
        self.layer_state = layer.layer_state

        self.size_cid_helper = ComponentIDComboHelper(self.ui.combodata_size_attribute,
                                                      self.layer_state.data_collection,
                                                      categorical=False)
        self.size_cid_helper.append_data(self.layer_state.layer)

        self.cmap_cid_helper = ComponentIDComboHelper(self.ui.combodata_cmap_attribute,
                                                      self.layer_state.data_collection,
                                                      categorical=False)
        self.cmap_cid_helper.append_data(self.layer_state.layer)

        add_callback(self.layer_state, 'size_mode', nonpartial(self._update_size_mode))
        add_callback(self.layer_state, 'color_mode', nonpartial(self._update_color_mode))

    def _update_size_mode(self):

        if self.layer_state.size_mode == "Fixed":
            self.ui.size_row_2.hide()
            self.ui.combodata_size_attribute.hide()
            self.ui.value_size.show()
        else:
            self.ui.value_size.hide()
            self.ui.combodata_size_attribute.show()
            self.ui.size_row_2.show()

    def _update_color_mode(self):

        if self.layer_state.color_mode == "Fixed":
            self.ui.color_row_2.hide()
            self.ui.color_row_3.hide()
            self.ui.combodata_cmap_attribute.hide()
            self.ui.spacer_color_label.show()
            self.ui.color_color.show()
        else:
            self.ui.color_color.hide()
            self.ui.combodata_cmap_attribute.show()
            self.ui.spacer_color_label.hide()
            self.ui.color_row_2.show()
            self.ui.color_row_3.show()


class FastScatterLayerStyleEditor(QtWidgets.QWidget):

    def __init__(self, layer, parent=None):

        super(FastScatterLayerStyleEditor, self).__init__(parent=parent)

        self.ui = load_ui('fast_scatter_style_editor.ui', self,
                          directory=os.path.dirname(__file__))


class LineLayerStyleEditor(QtWidgets.QWidget):

    def __init__(self, layer, parent=None):

        super(LineLayerStyleEditor, self).__init__(parent=parent)

        self.ui = load_ui('line_style_editor.ui', self,
                          directory=os.path.dirname(__file__))

        label_data = [('–––––––', 'solid'),
                      ('– – – – –', 'dashed'),
                      ('· · · · · · · ·', 'dotted'),
                      ('– · – · – ·', 'dashdot')]

        update_combobox(self.ui.combodata_linestyle, label_data)


class Histogram2DLayerStyleEditor(QtWidgets.QWidget):

    def __init__(self, layer, parent=None):

        super(Histogram2DLayerStyleEditor, self).__init__(parent=parent)

        self.ui = load_ui('histogram_2d_style_editor.ui', self,
                          directory=os.path.dirname(__file__))


class VectorLayerStyleEditor(QtWidgets.QWidget):

    def __init__(self, layer, parent=None):

        super(VectorLayerStyleEditor, self).__init__(parent=parent)

        self.ui = load_ui('vector_style_editor.ui', self,
                          directory=os.path.dirname(__file__))

        # TODO: layer_state should not have data_collection attribute
        self.layer_state = layer.layer_state

        self.vx_cid_helper = ComponentIDComboHelper(self.ui.combodata_vector_x_attribute,
                                                    self.layer_state.data_collection,
                                                    categorical=False, default_index=0)
        self.vx_cid_helper.append_data(self.layer_state.layer)

        self.vy_cid_helper = ComponentIDComboHelper(self.ui.combodata_vector_y_attribute,
                                                    self.layer_state.data_collection,
                                                    categorical=False, default_index=1)
        self.vy_cid_helper.append_data(self.layer_state.layer)


class Generic2DLayerStyleEditor(QtWidgets.QWidget):

    sub_editor = CurrentComboDataProperty('ui.combotext_style')

    def __init__(self, layer, parent=None):

        super(Generic2DLayerStyleEditor, self).__init__(parent=parent)

        # TODO: In future, should pass only state not layer?
        self.layer_state = layer.layer_state

        self.ui = load_ui('layer_style_editor.ui', self,
                          directory=os.path.dirname(__file__))

        self.sub_editors = QtWidgets.QStackedLayout()
        self.ui.placeholder.setLayout(self.sub_editors)

        self.editors = OrderedDict()
        self.editors['Scatter'] = ScatterLayerStyleEditor(layer)
        self.editors['Fast Scatter'] = FastScatterLayerStyleEditor(layer)
        self.editors['Line'] = LineLayerStyleEditor(layer)
        self.editors['2D Histogram'] = Histogram2DLayerStyleEditor(layer)
        self.editors['Vectors'] = VectorLayerStyleEditor(layer)

        for name, widget in self.editors.items():
            self.sub_editors.addWidget(widget)
            self.ui.combotext_style.addItem(name, widget)

        add_callback(self.layer_state, 'style', nonpartial(self._style_changed))

        connect_kwargs = {'size_scaling': dict(value_range=(0.1, 10), log=True),
                          'alpha': dict(value_range=(0, 1)),
                          'vector_scaling': dict(value_range=(1, 100), log=True)}

        autoconnect_callbacks_to_qt(self.layer_state, self.ui, connect_kwargs)

        # TEMP: should happen in ScatterLayerStyleEditor
        self.editors['Scatter']._update_size_mode()
        self.editors['Scatter']._update_color_mode()

    def _style_changed(self):
        self.sub_editors.setCurrentWidget(self.sub_editor)


if __name__ == "__main__":

    from glue.utils.qt import get_qapp
    app = get_qapp()

    class LayerArtist(object):
        layer_state = None

    from glue.core import Data, DataCollection

    d = Data(x=[1, 2, 3, 5, 6], y=[3, 4, 3, 2, 2])
    dc = DataCollection([d])

    layer_artist = LayerArtist()
    layer_artist.layer_state = ScatterLayerState(layer=d)
    layer_artist.layer_state.data_collection = dc

    editor = Generic2DLayerStyleEditor(layer=layer_artist)
    editor.show()
    editor.raise_()

    app.exec_()
