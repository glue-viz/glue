# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function

import warnings

from glue.config import viewer_tool
from glue.viewers.common.qt.data_viewer import DataViewer
from glue.viewers.common.qt.tool import Tool
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
    with warnings.catch_warnings(record=True) as w:
        ExampleViewer2(session)
    assert len(w) == 1
    assert str(w[0].message) == ("Tools 'TEST1' and 'TEST2' have the same "
                                 "shortcut ('A'). Ignoring shortcut for 'TEST2'")
