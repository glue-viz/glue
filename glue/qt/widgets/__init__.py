from .custom_component_widget import CustomComponentWidget
from .layer_tree_widget import LayerTreeWidget
from .message_widget import MessageWidget
from .mpl_widget import MplWidget

from ...viewers.histogram.qt_widget import HistogramWidget
from ...viewers.image.qt_widget import ImageWidget
from ...viewers.scatter.qt_widget import ScatterWidget
from ...viewers.dendro.qt_widget import DendroWidget

default_widgets = [ScatterWidget, HistogramWidget, ImageWidget]

try:
    from ...viewers.ginga.qt_widget import GingaWidget
    default_widgets.append(GingaWidget)
except ImportError:
    pass

def enable_dendrograms():
    if DendroWidget not in default_widgets:
        default_widgets.append(DendroWidget)
