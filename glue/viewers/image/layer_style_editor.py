import os

from qtpy import QtWidgets

from glue.utils.qt import load_ui

from glue_new_viewers.common.qt_helpers import autoconnect_qt
from glue.core.qt.data_combo_helper import ComponentIDComboHelper
from glue.viewers.common.qt.attribute_limits_helper import AttributeLimitsHelper


class ImageLayerStyleEditor(QtWidgets.QWidget):

    def __init__(self, layer, parent=None):

        super(ImageLayerStyleEditor, self).__init__(parent=parent)

        self.ui = load_ui('layer_style_editor.ui', self,
                          directory=os.path.dirname(__file__))

        # TODO: In future, should pass only state not layer?
        self.layer_state = layer.layer_state

        connect_kwargs = {
            'alpha': dict(value_range=(0, 1))
        }

        autoconnect_qt(self.layer_state, self.ui, connect_kwargs)

        self.att_helper = ComponentIDComboHelper(self.ui.combo_att,
                                                 self.layer_state.data_collection,
                                                 categorical=False)
        self.att_helper.append_data(self.layer_state.layer)

        self.att_limits_helper = AttributeLimitsHelper(self.ui.combo_att,
                                                       self.ui.value_vmin,
                                                       self.ui.value_vmax,
                                                       flip_button=self.ui.button_flip_limits)
