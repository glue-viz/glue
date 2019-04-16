from glue.viewers.matplotlib.tests.test_mouse_mode import TestMouseMode, Event

from ..toolbar_mode import ContrastMode


class TestContrastMode(TestMouseMode):

    def mode_factory(self):
        return ContrastMode

    def test_move_ignored_if_not_right_drag(self):
        e = Event(1, 2, button=1)
        self.mode.move(e)
        count = self.mode._axes.figure.canvas.get_width_height.call_count
        assert count == 0

    def test_clip_percentile(self):
        assert self.mode.get_clip_percentile() == (1, 99)
        self.mode.set_clip_percentile(2, 33)
        assert self.mode.get_clip_percentile() == (2, 33)

    def test_vmin_vmax(self):
        assert self.mode.get_vmin_vmax() == (None, None)
        self.mode.set_vmin_vmax(3, 4)
        assert self.mode.get_vmin_vmax() == (3, 4)
        assert self.mode.get_clip_percentile() == (None, None)

    # TODO: at the moment, this doesn't work because the dialog is non-modal
    # assert self.mode.get_vmin_vmax() == (5, 7)
    # def test_choose_vmin_vmax(self):
    #
    #     assert self.mode.get_vmin_vmax() == (None, None)
    #
    #     def fill_apply(dialog):
    #         dialog.vmin.setText('5')
    #         dialog.vmax.setText('7')
    #         dialog.accept()
    #
    #     with process_dialog(delay=500, function=fill_apply):
    #         self.mode.choose_vmin_vmax()
