#pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103
import matplotlib.pyplot as plt
from ...external.qt.QtGui import QMainWindow, QIcon
from ..widgets import MplWidget
from ..glue_toolbar import GlueToolbar
from ..mouse_mode import MouseMode


class TestMode(MouseMode):

    def __init__(self, axes, release_callback=None):
        super(TestMode, self).__init__(axes, release_callback=release_callback)
        self.icon = QIcon(':icons/square.png')
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
        from ...external.qt.QtGui import QApplication
        assert QApplication.instance() is not None
        self.win = QMainWindow()
        widget, axes = self._make_plot_widget(self.win)
        self.canvas = widget.canvas
        self.axes = axes
        self.tb = GlueToolbar(self.canvas, self.win)
        self.mode = TestMode(self.axes, release_callback=self.callback)
        self.tb.add_mode(self.mode)
        self.win.addToolBar(self.tb)
        self._called_back = False

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
        self.tb.buttons['TEST'].trigger()
        self.mode.release(None)
        assert self._called_back
