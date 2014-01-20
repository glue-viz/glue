from functools import partial

from ...external.qt import QtGui
from ...external.qt.QtCore import Qt

from ... import core

from ...clients.beeswarm_client import BeeSwarmClient
from ..glue_toolbar import GlueToolbar
from ..mouse_mode import (RectangleMode, CircleMode,
                          PolyMode, HRangeMode, VRangeMode)
from ...core.callback_property import add_callback

from .data_viewer import DataViewer
from .mpl_widget import MplWidget
from ..widget_properties import (ButtonProperty, FloatLineProperty,
                                 CurrentComboProperty,
                                 connect_bool_button, connect_float_edit)

from ..qtutil import pretty_number, load_ui
from scatter_widget import ScatterWidget


class BeeSwarmWidget(ScatterWidget):
    LABEL = 'BeeSwarm Plot'

    def __init__(self, session, parent=None):
        super(BeeSwarmWidget, self).__init__(session, parent)
        self.central_widget = MplWidget()
        self.option_widget = QtGui.QWidget()

        self.setCentralWidget(self.central_widget)

        self.ui = load_ui('scatterwidget', self.option_widget)
        # self.ui.setupUi(self.option_widget)
        self._tweak_geometry()

        self.client = BeeSwarmClient(self._data,
                                     self.central_widget.canvas.fig,
                                     artist_container=self._container)

        self._connect()
        self.unique_fields = set()
        self.make_toolbar()
        self.statusBar().setSizeGripEnabled(False)
        self.setFocusPolicy(Qt.StrongFocus)
