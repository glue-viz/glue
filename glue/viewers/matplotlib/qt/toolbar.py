from matplotlib.backends.backend_qt5 import NavigationToolbar2QT

from glue.config import viewer_tool
from glue.viewers.common.tool import CheckableTool, Tool


__all__ = ['MatplotlibTool', 'MatplotlibCheckableTool', 'HomeTool', 'SaveTool',
           'PanTool', 'ZoomTool']


def _ensure_mpl_nav(viewer):
    # Set up virtual Matplotlib navigation toolbar (don't show it)
    if not hasattr(viewer, '_mpl_nav'):
        viewer._mpl_nav = NavigationToolbar2QT(viewer.central_widget.canvas, viewer)
        viewer._mpl_nav.hide()


def _cleanup_mpl_nav(viewer):
    if getattr(viewer, '_mpl_nav', None) is not None:
        viewer._mpl_nav.setParent(None)
        try:
            viewer._mpl_nav.parent = None
        except AttributeError:
            pass
        viewer._mpl_nav = None


class MatplotlibTool(Tool):

    def __init__(self, viewer=None):
        super(MatplotlibTool, self).__init__(viewer=viewer)
        _ensure_mpl_nav(viewer)

    def close(self):
        _cleanup_mpl_nav(self.viewer)
        super(MatplotlibTool, self).close()


class MatplotlibCheckableTool(CheckableTool):

    def __init__(self, viewer=None):
        super(MatplotlibCheckableTool, self).__init__(viewer=viewer)
        _ensure_mpl_nav(viewer)

    def close(self):
        _cleanup_mpl_nav(self.viewer)
        super(MatplotlibCheckableTool, self).close()


@viewer_tool
class HomeTool(MatplotlibTool):

    tool_id = 'mpl:home'
    icon = 'glue_home'
    action_text = 'Home'
    tool_tip = 'Reset original zoom'
    shortcut = 'H'

    def activate(self):
        if hasattr(self.viewer, 'state') and hasattr(self.viewer.state, 'reset_limits'):
            self.viewer.state.reset_limits()
        else:
            self.viewer._mpl_nav.home()


@viewer_tool
class SaveTool(MatplotlibTool):

    tool_id = 'mpl:save'
    icon = 'glue_filesave'
    action_text = 'Save plot to file'
    tool_tip = 'Save the figure'

    def activate(self):
        self.viewer._mpl_nav.save_figure()


@viewer_tool
class PanTool(MatplotlibCheckableTool):

    tool_id = 'mpl:pan'
    icon = 'glue_move'
    action_text = 'Pan'
    tool_tip = 'Pan axes with left mouse, zoom with right'
    shortcut = 'M'

    def activate(self):
        self.viewer._mpl_nav.pan()

    def deactivate(self):
        if hasattr(self.viewer, '_mpl_nav'):
            self.viewer._mpl_nav.pan()


@viewer_tool
class ZoomTool(MatplotlibCheckableTool):

    tool_id = 'mpl:zoom'
    icon = 'glue_zoom_to_rect'
    action_text = 'Zoom'
    tool_tip = 'Zoom to rectangle'
    shortcut = 'Z'

    def activate(self):
        self.viewer._mpl_nav.zoom()

    def deactivate(self):
        if hasattr(self.viewer, '_mpl_nav'):
            self.viewer._mpl_nav.zoom()
