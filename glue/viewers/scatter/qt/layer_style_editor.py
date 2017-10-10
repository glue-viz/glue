import os
import platform
from collections import OrderedDict

import numpy as np

from qtpy import QtWidgets

from glue.external.echo.qt import autoconnect_callbacks_to_qt
from glue.utils.qt import load_ui, get_qapp


class ScatterLayerStyleEditor(QtWidgets.QWidget):

    def __init__(self, layer, parent=None):

        super(ScatterLayerStyleEditor, self).__init__(parent=parent)

        self.ui = load_ui('layer_style_editor_scatter.ui', self,
                          directory=os.path.dirname(__file__))

        # The following is needed because of a bug in Qt which means that
        # tab titles don't get scaled right.
        if platform.system() == 'Darwin':
            app = get_qapp()
            app_font = app.font()
            self.ui.tab_widget.setStyleSheet('font-size: {0}px'.format(app_font.pointSize()))

        self.layer_state = layer.state

        self.layer_state.add_callback('xerr_visible', self._update_xerr_att_combo)
        self.layer_state.add_callback('yerr_visible', self._update_yerr_att_combo)
        self.layer_state.add_callback('vector_visible', self._update_vector_att_combo)
        self.layer_state.add_callback('size_mode', self._update_size_mode)
        self.layer_state.add_callback('vector_mode', self._update_vector_mode)
        self.layer_state.add_callback('cmap_mode', self._update_cmap_mode)
        self.layer_state.add_callback('layer', self._update_warnings)

        self._update_xerr_att_combo()
        self._update_yerr_att_combo()
        self._update_vector_att_combo()
        self._update_size_mode()
        self._update_vector_mode()
        self._update_cmap_mode()
        self._update_warnings()

    def _update_xerr_att_combo(self, *args):
        self.ui.combosel_xerr_att.setEnabled(self.layer_state.xerr_visible)

    def _update_yerr_att_combo(self, *args):
        self.ui.combosel_yerr_att.setEnabled(self.layer_state.yerr_visible)

    def _update_vector_att_combo(self, *args):
        self.ui.combosel_vx_att.setEnabled(self.layer_state.vector_visible)
        self.ui.combosel_vy_att.setEnabled(self.layer_state.vector_visible)
        self.ui.bool_vector_hide_arrow.setEnabled(self.layer_state.vector_visible)

    def _update_warnings(self):

        if self.layer_state.layer is None:
            n_points = 0
        else:
            n_points = np.product(self.layer_state.layer.shape)

        if n_points > 10000:
            self.ui.label_warning_size.show()
        else:
            self.ui.label_warning_size.hide()

        if n_points > 50000:
            self.ui.label_warning_color.show()
        else:
            self.ui.label_warning_color.hide()

        if n_points > 10000:
            self.ui.label_warning_errorbar.show()
        else:
            self.ui.label_warning_errorbar.hide()

        if n_points > 10000:
            self.ui.label_warning_vector.show()
        else:
            self.ui.label_warning_vector.hide()

    def _update_size_mode(self, size_mode=None):

        if self.layer_state.size_mode == 'Fixed':
            self.ui.size_row_2.hide()
            self.ui.combosel_size_att.hide()
            self.ui.value_size.show()
        else:
            self.ui.value_size.hide()
            self.ui.combosel_size_att.show()
            self.ui.size_row_2.show()

    def _update_vector_mode(self, vector_mode=None):
        if self.layer_state.vector_mode == 'vx/vy':
            self.ui.label_vector_x.setText('vx')
            self.ui.label_vector_y.setText('vy')
        elif self.layer_state.vector_mode == 'ang/length':
            self.ui.label_vector_x.setText('ang(deg)')
            self.ui.label_vector_y.setText('length')


    def _update_cmap_mode(self, cmap_mode=None):

        if self.layer_state.cmap_mode == 'Fixed':
            self.ui.color_row_2.hide()
            self.ui.color_row_3.hide()
            self.ui.combosel_cmap_att.hide()
            self.ui.spacer_color_label.show()
            self.ui.color_color.show()
        else:
            self.ui.color_color.hide()
            self.ui.combosel_cmap_att.show()
            self.ui.spacer_color_label.hide()
            self.ui.color_row_2.show()
            self.ui.color_row_3.show()


class LineLayerStyleEditor(QtWidgets.QWidget):

    def __init__(self, layer, parent=None):

        super(LineLayerStyleEditor, self).__init__(parent=parent)

        self.ui = load_ui('layer_style_editor_line.ui', self,
                          directory=os.path.dirname(__file__))


class GenericLayerStyleEditor(QtWidgets.QWidget):

    def __init__(self, layer, parent=None):

        super(GenericLayerStyleEditor, self).__init__(parent=parent)

        self.ui = load_ui('layer_style_editor.ui', self,
                          directory=os.path.dirname(__file__))

        self.layer_state = layer.state

        self.sub_editors = QtWidgets.QStackedLayout()
        self.ui.placeholder.setLayout(self.sub_editors)

        self.editors = OrderedDict()
        self.editors['Scatter'] = ScatterLayerStyleEditor(layer)
        self.editors['Line'] = LineLayerStyleEditor(layer)

        for name, widget in self.editors.items():
            self.sub_editors.addWidget(widget)

        connect_kwargs = {'alpha': dict(value_range=(0, 1)),
                          'size_scaling': dict(value_range=(0.1, 10), log=True)}
        autoconnect_callbacks_to_qt(layer.state, self.ui, connect_kwargs)

        self.layer_state.add_callback('style', self._style_changed)

    def _style_changed(self, style):
        self.sub_editors.setCurrentWidget(self.editors[self.layer_state.style])
