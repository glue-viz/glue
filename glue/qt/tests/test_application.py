# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103
from distutils.version import LooseVersion
import tempfile
import os

import pytest
from mock import patch, MagicMock
import numpy as np

try:
    from IPython import __version__ as ipy_version
except:
    ipy_version = '0.0'

from ..glue_application import GlueApplication
from ..widgets.scatter_widget import ScatterWidget
from ..widgets.image_widget import ImageWidget
from ...core import Data


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
                fd.getSaveFileName.return_value = '/tmp/junk', 'jnk'
                with patch('glue.qt.decorators.QMessageBox') as mb:
                    cp().dump.side_effect = PicklingError
                    self.app._save_session()
                    assert mb.call_count == 1

    def test_save_session_no_file(self):
        """shouldnt try to save file if no file name provided"""
        with patch('glue.core.glue_pickle.CloudPickler') as cp:
            cp.return_value = ''
            with patch('glue.qt.glue_application.QFileDialog') as fd:
                fd.getSaveFileName.return_value = '', 'jnk'
                # crashes if open called on null string
                self.app._save_session()

    def test_save_session_ioerror(self):
        """should show box on ioerror"""
        with patch('glue.qt.glue_application.QFileDialog') as fd:
            with patch('__builtin__.open') as op:
                op.side_effect = IOError
                fd.getSaveFileName.return_value = '/tmp/junk', '/tmp/junk'
                with patch('glue.qt.decorators.QMessageBox') as mb:
                    self.app._save_session()
                    assert mb.call_count == 1

    def test_save_restore(self):
        self.app._data.append(Data(label='x', x=[1, 2, 3]))

        with patch('glue.qt.glue_application.QFileDialog') as fd:
            _, fname = tempfile.mkstemp(suffix='.glu')
            fd.getSaveFileName.return_value = fname, '*.*'

            self.app._save_session()

            fd.getOpenFileName.return_value = fname, '*.*'

            new = self.app._restore_session(show=False)
            assert new._data[0].label == 'x'
            np.testing.assert_array_equal(new._data[0]['x'], [1, 2, 3])

            os.unlink(fname)

    @pytest.mark.xfail("LooseVersion(ipy_version) <= LooseVersion('0.11')")
    def test_terminal_present(self):
        """For good setups, terminal is available"""
        if not self.app.has_terminal():
            import sys
            sys.stderr.write(self.app._terminal_exception)
            assert False

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
            assert qmb.call_count == 1

    def is_terminal_importable(self):
        try:
            import glue.qt.widgets.glue_terminal
            return True
        except:
            return False

    @pytest.mark.xfail("LooseVersion(ipy_version) <= LooseVersion('0.11')")
    def test_toggle_terminal(self):
        term = MagicMock()
        self.app._terminal = term

        term.isVisible.return_value = False
        self.app._terminal_button.click()
        assert term.show.call_count == 1

        term.isVisible.return_value = True
        self.app._terminal_button.click()
        assert term.hide.call_count == 1

    def test_close_tab(self):
        assert self.app.tab_widget.count() == 1
        self.app._new_tab()
        assert self.app.tab_widget.count() == 2
        self.app._close_tab(0)
        assert self.app.tab_widget.count() == 1
        # do not delete last tab
        self.app._close_tab(0)
        assert self.app.tab_widget.count() == 1

    def test_new_data_viewer_cancel(self):
        with patch('glue.qt.glue_application.pick_class') as pc:
            pc.return_value = None

            ct = len(self.app.current_tab.subWindowList())

            self.app.new_data_viewer()
            assert len(self.app.current_tab.subWindowList()) == ct

    def test_new_data_viewer(self):
        with patch('glue.qt.glue_application.pick_class') as pc:

            pc.return_value = ScatterWidget

            ct = len(self.app.current_tab.subWindowList())

            self.app.new_data_viewer()
            assert len(self.app.current_tab.subWindowList()) == ct + 1

    def test_new_data_defaults(self):
        from ...config import qt_client

        with patch('glue.qt.glue_application.pick_class') as pc:
            pc.return_value = None

            d2 = Data(x=np.array([[1, 2, 3], [4, 5, 6]]))
            d1 = Data(x=np.array([1, 2, 3]))

            self.app.new_data_viewer(data=d1)
            args, kwargs = pc.call_args
            assert qt_client.members[kwargs['default']] == ScatterWidget

            self.app.new_data_viewer(data=d2)
            args, kwargs = pc.call_args
            assert qt_client.members[kwargs['default']] == ImageWidget
