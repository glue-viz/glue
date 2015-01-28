from .custom_component_widget import CustomComponentWidget
from .layer_tree_widget import LayerTreeWidget
from .message_widget import MessageWidget
from .mpl_widget import MplWidget

def enable_dendrograms():
    from ..config import qt_client
    from ...viewers.dendro.qt_widget import DendroWidget
    qt_client.add(DendroWidget)
