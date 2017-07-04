import os

from qtpy import QtWidgets

from astropy.visualization import (LinearStretch, SqrtStretch,
                                   LogStretch, AsinhStretch)

from glue.external.echo.qt import autoconnect_callbacks_to_qt
from glue.utils.qt import load_ui, update_combobox
from glue.core.qt.data_combo_helper import ComponentIDComboHelper


class ImageLayerStyleEditor(QtWidgets.QWidget):

    def __init__(self, layer, parent=None):

        super(ImageLayerStyleEditor, self).__init__(parent=parent)

        self.ui = load_ui('layer_style_editor.ui', self,
                          directory=os.path.dirname(__file__))

        connect_kwargs = {'alpha': dict(value_range=(0, 1)),
                          'contrast': dict(value_range=(-2, 4)),
                          'bias': dict(value_range=(-2, 3))}

        percentiles = [('Min/Max', 100),
                       ('99.5%', 99.5),
                       ('99%', 99),
                       ('95%', 95),
                       ('90%', 90),
                       ('Custom', 'Custom')]

        update_combobox(self.ui.combodata_percentile, percentiles)

        stretches = [('Linear', 'linear'),
                     ('Square Root', 'sqrt'),
                     ('Arcsinh', 'arcsinh'),
                     ('Logarithmic', 'log')]

        update_combobox(self.ui.combodata_stretch, stretches)

        self.attribute_helper = ComponentIDComboHelper(self.ui.combodata_attribute,
                                                       layer.data_collection)

        self.attribute_helper.append_data(layer.layer)

        autoconnect_callbacks_to_qt(layer.state, self.ui, connect_kwargs)

        layer._viewer_state.add_callback('color_mode', self._update_color_mode)

        self._update_color_mode(layer._viewer_state.color_mode)

    def _update_color_mode(self, color_mode):
        if color_mode == 'Colormaps':
            self.ui.color_color.hide()
            self.ui.combodata_cmap.show()
        else:
            self.ui.color_color.show()
            self.ui.combodata_cmap.hide()
