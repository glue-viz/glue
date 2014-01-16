from functools import partial

from ...external.qt import QtGui
from ...external.qt.QtCore import Qt

from ... import core

from ...clients.scatter_client import ScatterClient
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
