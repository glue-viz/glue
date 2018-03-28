from __future__ import absolute_import, division, print_function

from glue.config import viewer_tool
from glue.viewers.common.qt.tool import CheckableTool, Tool


__all__ = ['HomeTool', 'SaveTool', 'PanTool', 'ZoomTool']


class MatplotlibTool(Tool):
    pass


class MatplotlibCheckableTool(CheckableTool):
    pass


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
        self.viewer._mpl_nav.zoom()
