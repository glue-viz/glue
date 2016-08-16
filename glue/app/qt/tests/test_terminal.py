from __future__ import absolute_import, division, print_function

from mock import MagicMock, patch

from glue.tests.helpers import requires_ipython, IPYTHON_INSTALLED


if IPYTHON_INSTALLED:
    from ..terminal import glue_terminal


@requires_ipython
class TestTerminal(object):
    def test_mpl_non_interactive(self):
        """IPython v0.12 sometimes turns on mpl interactive. Ensure
           we catch that"""

        import matplotlib
        assert not matplotlib.is_interactive()
        gt = glue_terminal()
        assert not matplotlib.is_interactive()

    def test_update_namespace(self):
        """Test that top level namespace API works without error"""
        gt = glue_terminal()
        gt.update_namespace({'x': 3})
        assert 'x' in gt.namespace

    def test_accepts_drops(self):
        gt = glue_terminal()
        assert gt.acceptDrops()

    def test_drops_update_namespace(self):
        """DnD adds variable name to namespace"""
        with patch('glue.app.qt.terminal.QtWidgets.QInputDialog') as dialog:
            dialog.getText.return_value = 'accept_var', True

            gt = glue_terminal()
            event = MagicMock()
            event.mimeData().data.return_value = [5]

            gt.dropEvent(event)
            assert gt.namespace.get('accept_var') == 5

    def test_cancel_drop(self):
        """Drop not added if user cancels dialog box"""

        with patch('glue.app.qt.terminal.QtWidgets.QInputDialog') as dialog:
            dialog.getText.return_value = 'cancel_var', False

            gt = glue_terminal()
            event = MagicMock()
            event.mimeData().data.return_value = [5]

            gt.dropEvent(event)
            assert 'cancel_var' not in gt.namespace

    def test_ignore_drag_enter(self):
        event = MagicMock()
        event.mimeData().hasFormat.return_value = False

        gt = glue_terminal()
        gt.dragEnterEvent(event)

        event.ignore.assert_called_once_with()

    def test_accept_drag_enter(self):
        event = MagicMock()
        event.mimeData().hasFormat.return_value = True

        gt = glue_terminal()
        gt.dragEnterEvent(event)

        event.accept.assert_called_once_with()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
