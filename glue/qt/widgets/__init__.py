from .custom_component_widget import CustomComponentWidget
from .histogram_widget import HistogramWidget
from .image_widget import ImageWidget
from .layer_tree_widget import LayerTreeWidget
from .message_widget import MessageWidget
from .mpl_widget import MplWidget
from .scatter_widget import ScatterWidget
from .dendro_widget import DendroWidget

default_widgets = [ScatterWidget, HistogramWidget, ImageWidget]

try:
    from .qt.widgets.ginga_widget import GingaWidget
    default_widgets.append(GingaWidget)
except ImportError:
    pass

def enable_dendrograms():
    if DendroWidget not in default_widgets:
        default_widgets.append(DendroWidget)
