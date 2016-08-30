# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function

from qtpy import QtWidgets

from glue.viewers.common.qt.mpl_widget import MplWidget
from glue.viewers.common.qt.data_viewer import DataViewer
from glue.viewers.common.qt.mouse_mode import MouseMode
from glue.icons.qt import get_icon
from glue.core.tests.util import simple_session

from ..mpl_toolbar import MatplotlibViewerToolbar


class MouseModeTest(MouseMode):

    tool_id = 'TEST'
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
        self._make_toolbar()

    def _make_toolbar(self):
        super(ExampleViewer, self)._make_toolbar()
        self.mode = MouseModeTest(self, release_callback=self.callback)
        self.toolbar.add_mode(self.mode)

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
        for mode in self.viewer.toolbar.buttons:
            if mode == target_mode and self.viewer.toolbar.buttons[mode].isCheckable():
                assert self.viewer.toolbar.buttons[mode].isChecked()
                self.viewer.toolbar._active == target_mode
            else:
                assert not self.viewer.toolbar.buttons[mode].isChecked()

    def test_callback(self):
        self.viewer.toolbar.buttons['HOME'].trigger()
        self.viewer.mode.release(None)
        assert self.viewer._called_back

    def test_change_mode(self):

        self.viewer.toolbar.buttons['PAN'].toggle()
        assert self.viewer.toolbar.mode.tool_id == 'PAN'
        assert self.viewer.toolbar._mpl_nav.mode == 'pan/zoom'

        self.viewer.toolbar.buttons['PAN'].toggle()
        assert self.viewer.toolbar.mode is None
        assert self.viewer.toolbar._mpl_nav.mode == ''

        self.viewer.toolbar.buttons['ZOOM'].trigger()
        assert self.viewer.toolbar.mode.tool_id == 'ZOOM'
        assert self.viewer.toolbar._mpl_nav.mode == 'zoom rect'

        self.viewer.toolbar.buttons['BACK'].trigger()
        assert self.viewer.toolbar.mode is None
        assert self.viewer.toolbar._mpl_nav.mode == ''

        self.viewer.toolbar.buttons['TEST'].trigger()
        assert self.viewer.toolbar.mode.tool_id == 'TEST'
        assert self.viewer.toolbar._mpl_nav.mode == ''
