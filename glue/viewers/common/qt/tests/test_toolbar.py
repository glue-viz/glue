# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function

from glue.viewers.common.qt.mpl_widget import MplWidget

from glue.viewers.common.qt.mouse_mode import MouseMode
from glue.icons.qt import get_icon

from ..mpl_toolbar import MatplotlibViewerToolbar


class MouseModeTest(MouseMode):

    def __init__(self, axes, release_callback=None):
        super(MouseModeTest, self).__init__(axes, release_callback=release_callback)
        self.icon = get_icon('glue_square')
        self.mode_id = 'TEST'
        self.action_text = 'test text'
        self.tool_tip = 'just testing'
        self.last_mode = None

    def press(self, event):
        self.last_mode = 'PRESS'

    def move(self, event):
        self.last_mode = 'MOVE'


class TestToolbar(object):

    def setup_method(self, method):
        from qtpy import QtWidgets
        assert QtWidgets.QApplication.instance() is not None
        self.win = QtWidgets.QMainWindow()
        widget, axes = self._make_plot_widget(self.win)
        self.canvas = widget.canvas
        self.axes = axes
        self.tb = MatplotlibViewerToolbar(self.canvas, self.win)
        self.mode = MouseModeTest(self.axes, release_callback=self.callback)
        self.tb.add_mode(self.mode)
        self.win.addToolBar(self.tb)
        self._called_back = False

    def teardown_method(self, method):
        self.win.close()

    def _make_plot_widget(self, parent):
        widget = MplWidget(parent)
        ax = widget.canvas.fig.add_subplot(111)
        p = ax.plot([1, 2, 3])[0]
        return widget, ax

    def callback(self, mode):
        self._called_back = True

    def assert_valid_mode_state(self, target_mode):
        for mode in self.tb.buttons:
            if mode == target_mode and self.tb.buttons[mode].isCheckable():
                assert self.tb.buttons[mode].isChecked()
                self.tb._active == target_mode
            else:
                assert not self.tb.buttons[mode].isChecked()

    def test_callback(self):
        self.tb.buttons['HOME'].trigger()
        self.mode.release(None)
        assert self._called_back
