# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function

import warnings

from qtpy import QtWidgets

from glue.config import viewer_tool
from glue.viewers.common.qt.mpl_widget import MplWidget
from glue.viewers.common.qt.data_viewer import DataViewer
from glue.viewers.common.qt.tool import Tool
from glue.viewers.common.qt.toolbar import BasicToolbar
from glue.viewers.common.qt.mouse_mode import MouseMode
from glue.icons.qt import get_icon
from glue.core.tests.util import simple_session

from ..mpl_toolbar import MatplotlibViewerToolbar


class MouseModeTest(MouseMode):

    tool_id = 'test'
    tool_tip = 'just testing'
    icon = 'glue_square'

    def __init__(self, axes, release_callback=None):
        super(MouseModeTest, self).__init__(axes, release_callback=release_callback)
        self.action_text = 'test text'
        self.last_mode = None

    def press(self, event):
        self.last_mode = 'PRESS'

    def move(self, event):
        self.last_mode = 'MOVE'


class ExampleViewer(DataViewer):

    _toolbar_cls = MatplotlibViewerToolbar

    def __init__(self, session, parent=None):
        super(ExampleViewer, self).__init__(session, parent=parent)
        self.central_widget = MplWidget(parent)
        self._axes = self.central_widget.canvas.fig.add_subplot(111)
        self._axes.plot([1, 2, 3])[0]
        self.setCentralWidget(self.central_widget)

    def initialize_toolbar(self):
        super(ExampleViewer, self).initialize_toolbar()
        self.tool = MouseModeTest(self, release_callback=self.callback)
        self.toolbar.add_tool(self.tool)

    def callback(self, mode):
        self._called_back = True

    @property
    def axes(self):
        return self._axes


class TestToolbar(object):

    def setup_method(self, method):
        self.session = simple_session()
        self.viewer = ExampleViewer(self.session)
        self._called_back = False

    def teardown_method(self, method):
        self.viewer.close()

    def assert_valid_mode_state(self, target_mode):
        for tool_id in self.viewer.toolbar.actions:
            if tool_id == target_mode and self.viewer.toolbar.actions[tool_id].isCheckable():
                assert self.viewer.toolbar.actions[tool_id].isChecked()
                self.viewer.toolbar._active == target_mode
            else:
                assert not self.viewer.toolbar.actions[tool_id].isChecked()

    def test_callback(self):
        self.viewer.toolbar.actions['mpl:home'].trigger()
        self.viewer.tool.release(None)
        assert self.viewer._called_back

    def test_change_mode(self):

        self.viewer.toolbar.actions['mpl:pan'].toggle()
        assert self.viewer.toolbar.active_tool.tool_id == 'mpl:pan'
        assert self.viewer.toolbar._mpl_nav.mode == 'pan/zoom'

        self.viewer.toolbar.actions['mpl:pan'].toggle()
        assert self.viewer.toolbar.active_tool is None
        assert self.viewer.toolbar._mpl_nav.mode == ''

        self.viewer.toolbar.actions['mpl:zoom'].trigger()
        assert self.viewer.toolbar.active_tool.tool_id == 'mpl:zoom'
        assert self.viewer.toolbar._mpl_nav.mode == 'zoom rect'

        self.viewer.toolbar.actions['mpl:back'].trigger()
        assert self.viewer.toolbar.active_tool is None
        assert self.viewer.toolbar._mpl_nav.mode == ''

        self.viewer.toolbar.actions['test'].trigger()
        assert self.viewer.toolbar.active_tool.tool_id == 'test'
        assert self.viewer.toolbar._mpl_nav.mode == ''


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
        viewer = ExampleViewer2(session)
    assert len(w) == 1
    assert str(w[0].message) == "Tools 'TEST1' and 'TEST2' have the same shortcut ('A'). Ignoring shortcut for 'TEST2'"
