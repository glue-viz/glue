# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

import pytest
from glue.config import viewer_tool
from glue.viewers.common.qt.data_viewer import DataViewer
from glue.viewers.common.tool import Tool
from glue.viewers.common.qt.toolbar import BasicToolbar
from glue.core.tests.util import simple_session


@viewer_tool
class ExampleTool1(Tool):

    tool_id = 'TEST1'
    tool_tip = 'tes1'
    icon = 'glue_square'
    shortcut = 'A'


@viewer_tool
class ExampleTool2(Tool):

    tool_id = 'TEST2'
    tool_tip = 'tes2'
    icon = 'glue_square'
    shortcut = 'A'


class ExampleViewer2(DataViewer):

    _toolbar_cls = BasicToolbar
    tools = ['TEST1', 'TEST2']

    def __init__(self, session, parent=None):
        super(ExampleViewer2, self).__init__(session, parent=parent)


def test_duplicate_shortcut():
    session = simple_session()
    expected_warning = ("Tools 'TEST1' and 'TEST2' have the same "
                        r"shortcut \('A'\). Ignoring shortcut for 'TEST2'")
    with pytest.warns(UserWarning, match=expected_warning):
        ExampleViewer2(session)
