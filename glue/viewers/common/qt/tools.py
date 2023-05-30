from glue.config import viewer_tool
from glue.utils.qt import pick_item
from glue.viewers.common.tool import Tool, SimpleToolMenu

__all__ = ['MoveTabTool', 'WindowTool']


@viewer_tool
class WindowTool(SimpleToolMenu):
    """
    A generic "window operations" tool that the Qt app and plugins
    can register tools for windowing operations with.
    """

    tool_id = 'window'
    icon = 'windows'
    tool_tip = 'Modify the viewer window'


@viewer_tool
class MoveTabTool(Tool):

    icon = 'windows'
    tool_id = 'window:movetab'
    action_text = 'Move to another tab'
    tool_tip = 'Move viewer to another tab'

    def activate(self):
        app = self.viewer.session.application
        default = 1 if (app.tab_count > 1 and app.current_tab == app.tab(0)) else 0
        tab = pick_item(range(app.tab_count), app.tab_names, title="Move Viewer", label="Select a tab", default=default)
        if tab is not None:
            app.move_viewer_to_tab(self.viewer, tab)
