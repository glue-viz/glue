# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from glue.viewers.matplotlib.qt.data_viewer import MatplotlibDataViewer
from glue.viewers.matplotlib.toolbar_mode import ToolbarModeBase
from glue.core.tests.util import simple_session


class ToolbarModeTest(ToolbarModeBase):

    tool_id = 'test'
    tool_tip = 'just testing'
    icon = 'glue_square'

    def __init__(self, axes, release_callback=None):
        super(ToolbarModeTest, self).__init__(axes, release_callback=release_callback)
        self.action_text = 'test text'
        self.last_mode = None

    def press(self, event):
        self.last_mode = 'PRESS'

    def move(self, event):
        self.last_mode = 'MOVE'


class ExampleViewer(MatplotlibDataViewer):

    def __init__(self, session, parent=None):
        super(ExampleViewer, self).__init__(session, parent=parent)
        self.axes.plot([1, 2, 3])[0]

    def initialize_toolbar(self):
        super(ExampleViewer, self).initialize_toolbar()
        self.tool = ToolbarModeTest(self, release_callback=self.callback)
        self.toolbar.add_tool(self.tool)

    def callback(self, mode):
        self._called_back = True


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
        assert self.viewer._mpl_nav.mode == 'pan/zoom'

        self.viewer.toolbar.actions['mpl:pan'].toggle()
        assert self.viewer.toolbar.active_tool is None
        assert self.viewer._mpl_nav.mode == ''

        self.viewer.toolbar.actions['mpl:zoom'].trigger()
        assert self.viewer.toolbar.active_tool.tool_id == 'mpl:zoom'
        assert self.viewer._mpl_nav.mode == 'zoom rect'

        self.viewer.toolbar.actions['test'].trigger()
        assert self.viewer.toolbar.active_tool.tool_id == 'test'
        assert self.viewer._mpl_nav.mode == ''
