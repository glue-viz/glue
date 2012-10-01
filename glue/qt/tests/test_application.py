#pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103
from mock import patch, MagicMock

from PyQt4.QtGui import QApplication

from .. import get_qapp
from ..glue_application import GlueApplication


def tab_count(app):
    return app.tab_bar.count()


class TestGlueApplication(object):

    def setup_method(self, method):
        self.app = GlueApplication()

    def teardown_method(self, method):
        self.app.close()

    def test_new_tabs(self):
        t0 = tab_count(self.app)
        self.app._new_tab()
        assert tab_count(self.app) == t0 + 1

    def test_save_session_pickle_error(self):
        from pickle import PicklingError
        with patch('glue.core.glue_pickle.CloudPickler') as cp:
            with patch('glue.qt.glue_application.QFileDialog') as fd:
                fd.getSaveFileName.return_value = '/tmp/junk'
                with patch('glue.qt.decorators.QMessageBox') as mb:
                    cp().dump.side_effect = PicklingError
                    self.app._save_session()
                    assert mb.call_count == 1

    def test_save_session_no_file(self):
        """shouldnt try to save file if no file name provided"""
        with patch('glue.core.glue_pickle.CloudPickler') as cp:
            cp.return_value = ''
            with patch('glue.qt.glue_application.QFileDialog') as fd:
                fd.getSaveFileName.return_value = ''
                # crashes if open called on null string
                self.app._save_session()

    def test_save_session_ioerror(self):
        """should show box on ioerror"""
        with patch('glue.qt.glue_application.QFileDialog') as fd:
            # can't write, raises IOError. Dangerous hack!
            fd.getSaveFileName.return_value = '/_junk'
            with patch('glue.qt.decorators.QMessageBox') as mb:
                self.app._save_session()
                assert mb.call_count == 1

    def test_terminal_present(self):
        """For good setups, terminal is available"""
        assert self.app.has_terminal()

    def app_without_terminal(self):
        if not self.app.has_terminal():
            return self.app

        with patch('glue.qt.widgets.terminal.glue_terminal') as terminal:
            terminal.side_effect = Exception("disabled")
            app = GlueApplication()
            return app

    def test_functional_without_terminal(self):
        """Can still create app without terminal"""
        app = self.app_without_terminal()

    def test_messagebox_on_disabled_terminal(self):
        """Clicking on the terminal toggle button raises messagebox on error"""
        app = self.app_without_terminal()
        with patch('glue.qt.glue_application.QMessageBox') as qmb:
            app._terminal_button.click()
            assert qmb.critical.call_count == 1

    def is_terminal_importable(self):
        try:
            import glue.qt.widgets.glue_terminal
            return True
        except:
            return False
