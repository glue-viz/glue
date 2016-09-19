from __future__ import absolute_import, division, print_function

import os

from qtpy import QtWidgets

from glue.utils.qt import load_ui
from glue.viewers.common.qt.attribute_limits_helper import AttributeLimitsHelper
from glue.core.qt.data_combo_helper import ComponentIDComboHelper

from glue_new_viewers.common.qt_helpers import autoconnect_qt


class ScatterLayerStyleEditor(QtWidgets.QWidget):

    def __init__(self, layer, parent=None):

        super(ScatterLayerStyleEditor, self).__init__(parent=parent)

        self.ui = load_ui('layer_style_editor.ui', self,
                          directory=os.path.dirname(__file__))

        # TODO: In future, should pass only state not layer?
        self.layer_state = layer.layer_state

        connect_kwargs = {
            'size_scaling': dict(value_range=(0.1, 10), log=True),
            'alpha': dict(value_range=(0, 1))
        }

        autoconnect_qt(self.layer_state, self.ui, connect_kwargs)
        # autoconnect_qt(self.layer_state.layer.style, self.ui)

        self.ui.combotext_size_mode.currentIndexChanged.connect(self._update_size_mode)

        # TODO: layer_state should not have data_collection attribute

        self.size_limits_helper = AttributeLimitsHelper(self.ui.combo_size_attribute,
                                                        self.ui.value_size_vmin,
                                                        self.ui.value_size_vmax,
                                                        flip_button=self.ui.button_flip_size)

        self.size_cid_helper = ComponentIDComboHelper(self.ui.combo_size_attribute,
                                                      self.layer_state.data_collection,
                                                      categorical=False)
        self.size_cid_helper.append_data(self.layer_state.layer)

        self.ui.combotext_color_mode.currentIndexChanged.connect(self._update_color_mode)

        self.cmap_limits_helper = AttributeLimitsHelper(self.ui.combo_cmap_attribute,
                                                        self.ui.value_cmap_vmin,
                                                        self.ui.value_cmap_vmax,
                                                        flip_button=self.ui.button_flip_cmap)

        self.cmap_cid_helper = ComponentIDComboHelper(self.ui.combo_cmap_attribute,
                                                      self.layer_state.data_collection,
                                                      categorical=False)
        self.cmap_cid_helper.append_data(self.layer_state.layer)

        self._update_size_mode()
        self._update_color_mode()

    def _update_size_mode(self):

        if self.layer_state.size_mode == "Fixed":
            self.ui.size_row_2.hide()
            self.ui.combo_size_attribute.hide()
            self.ui.value_size.show()
        else:
            self.ui.value_size.hide()
            self.ui.combo_size_attribute.show()
            self.ui.size_row_2.show()

    def _update_color_mode(self):

        if self.layer_state.color_mode == "Fixed":
            self.ui.color_row_2.hide()
            self.ui.color_row_3.hide()
            self.ui.combo_cmap_attribute.hide()
            self.ui.spacer_color_label.show()
            self.ui.color_color.show()
        else:
            self.ui.color_color.hide()
            self.ui.combo_cmap_attribute.show()
            self.ui.spacer_color_label.hide()
            self.ui.color_row_2.show()
            self.ui.color_row_3.show()

        # Set up attribute list
        # label_data = [(comp.label, comp) for comp in self.visible_components]
        # update_combobox(self.ui.combo_size_attribute, label_data)

        # self.ui.combotext_size_mode.currentIndexChanged.connect(self._update_size_mode)
        # self.ui.combo_size_attribute.currentIndexChanged.connect(self._update_size_limits)
        # self.ui.button_flip_size.clicked.connect(self._flip_size)
