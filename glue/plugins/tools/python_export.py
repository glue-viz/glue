from __future__ import absolute_import, division, print_function

from qtpy import compat
from glue.config import viewer_tool
from glue.viewers.common.qt.tool import Tool


@viewer_tool
class PythonExportTool(Tool):

    icon = 'glue_pythonsave'
    tool_id = 'save:python'
    action_text = 'Save Python script to reproduce plot'
    tool_tip = 'Save Python script to reproduce plot'

    def activate(self):

        filename, _ = compat.getsavefilename(parent=self.viewer, basedir="make_plot.py")

        if not filename:
            return

        if not filename.endswith('.py'):
            filename += '.py'

        self.viewer.export_as_script(filename)
