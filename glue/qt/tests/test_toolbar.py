import sys

import pytest
import matplotlib.pyplot as plt
from PyQt4.QtGui import QApplication, QMainWindow, QIcon

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
        self.app = QApplication(sys.argv)
        self.win = QMainWindow()
        p = plt.plot([1, 2, 3])[0]
        self.canvas = p.axes.figure.canvas
        self.axes = p.axes
        self.tb = GlueToolbar(self.canvas, self.win)
        self.mode = TestMode(self.axes, release_callback=self.callback)
        self.tb.add_mode(self.mode)
        self.win.addToolBar(self.tb)
        self._called_back = False

    def callback(self, mode):
        self._called_back = True

    def assert_valid_mode_state(self, target_mode):
        for mode in self.tb.buttons:
            if mode == target_mode:
                assert self.tb.buttons[mode].isChecked()
            else:
                assert not self.tb.buttons[mode].isChecked()

        assert self.tb._active == target_mode

    @pytest.mark.skip("Test running into issues with widget locks?")
    def test_mode_exclusive(self):
        for mode in self.tb.buttons:
            self.tb.buttons[mode].trigger()
            self.assert_valid_mode_state(mode)

    def test_callback(self):
        self.tb.buttons['TEST'].trigger()
        self.mode.release(None)
        assert self._called_back
