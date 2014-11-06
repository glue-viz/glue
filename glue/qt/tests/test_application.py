# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function

from distutils.version import LooseVersion
import tempfile
import io
import os
import sys

import pytest
from mock import patch, MagicMock
import numpy as np

try:
    from IPython import __version__ as ipy_version
except:
    ipy_version = '0.0'

from ..glue_application import GlueApplication
from ...external.qt.QtCore import QMimeData, QUrl
from ..widgets.scatter_widget import ScatterWidget
from ..widgets.image_widget import ImageWidget
from ...core import Data

from ...external.six import PY3

from ...tests.helpers import requires_ipython_ge_012


def tab_count(app):
    return app.tab_bar.count()


class TestGlueApplication(object):

    def setup_method(self, method):
        self.app = GlueApplication()
        self.app._create_terminal()

    def teardown_method(self, method):
        self.app.close()

    def test_new_tabs(self):
        t0 = tab_count(self.app)
        self.app.new_tab()
        assert tab_count(self.app) == t0 + 1

    def test_save_session(self):
        self.app.save_session = MagicMock()
        with patch('glue.qt.glue_application.QFileDialog') as fd:
            fd.getSaveFileName.return_value = '/tmp/junk', 'jnk'
            self.app._choose_save_session()
            self.app.save_session.assert_called_once_with('/tmp/junk')

    def test_save_session_cancel(self):
        """shouldnt try to save file if no file name provided"""
        self.app.save_session = MagicMock()
        with patch('glue.qt.glue_application.QFileDialog') as fd:
            fd.getSaveFileName.return_value = '', 'jnk'
            self.app._choose_save_session()
            assert self.app.save_session.call_count == 0

    def test_choose_save_session_ioerror(self):
        """should show box on ioerror"""
        with patch('glue.qt.glue_application.QFileDialog') as fd:
            if sys.version_info.major == 2:
                mock_open = '__builtin__.open'
            else:
                mock_open = 'builtins.open'
            with patch(mock_open) as op:
                op.side_effect = IOError
                fd.getSaveFileName.return_value = '/tmp/junk', '/tmp/junk'
                with patch('glue.qt.glue_application.QMessageBox') as mb:
                    self.app._choose_save_session()
                    assert mb.call_count == 1

    @requires_ipython_ge_012
    def test_terminal_present(self):
        """For good setups, terminal is available"""
        if not self.app.has_terminal():
            sys.stderr.write(self.app._terminal_exception)
            assert False

    def app_without_terminal(self):
        if not self.app.has_terminal():
            return self.app

        with patch('glue.qt.widgets.terminal.glue_terminal') as terminal:
            terminal.side_effect = Exception("disabled")
            app = GlueApplication()
            app._create_terminal()
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

    @requires_ipython_ge_012
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
        self.app.new_tab()
        assert self.app.tab_widget.count() == 2
        self.app.close_tab(0)
        assert self.app.tab_widget.count() == 1
        # do not delete last tab
        self.app.close_tab(0)
        assert self.app.tab_widget.count() == 1

    def test_new_data_viewer_cancel(self):
        with patch('glue.qt.glue_application.pick_class') as pc:
            pc.return_value = None

            ct = len(self.app.current_tab.subWindowList())

            self.app.choose_new_data_viewer()
            assert len(self.app.current_tab.subWindowList()) == ct

    def test_new_data_viewer(self):
        with patch('glue.qt.glue_application.pick_class') as pc:

            pc.return_value = ScatterWidget

            ct = len(self.app.current_tab.subWindowList())

            self.app.choose_new_data_viewer()
            assert len(self.app.current_tab.subWindowList()) == ct + 1

    def test_move(self):
        viewer = self.app.new_data_viewer(ScatterWidget)
        viewer.move(10, 20)
        assert viewer.position == (10, 20)

    def test_resize(self):
        viewer = self.app.new_data_viewer(ScatterWidget)
        viewer.viewer_size = (100, 200)
        assert viewer.viewer_size == (100, 200)

    def test_new_data_defaults(self):
        from ...config import qt_client

        with patch('glue.qt.glue_application.pick_class') as pc:
            pc.return_value = None

            d2 = Data(x=np.array([[1, 2, 3], [4, 5, 6]]))
            d1 = Data(x=np.array([1, 2, 3]))

            self.app.choose_new_data_viewer(data=d1)
            args, kwargs = pc.call_args
            assert qt_client.members[kwargs['default']] == ScatterWidget

            self.app.choose_new_data_viewer(data=d2)
            args, kwargs = pc.call_args
            assert qt_client.members[kwargs['default']] == ImageWidget

    def test_drop_load_data(self):
        m = QMimeData()
        m.setUrls([QUrl('test.fits')])
        e = MagicMock()
        e.mimeData.return_value = m
        load = MagicMock()
        self.app.load_data = load
        self.app.dropEvent(e)
        assert load.call_count == 1

    def test_subset_facet(self):
        # regression test for 335

        act = self.app._ui.layerWidget._actions['facet']
        self.app.data_collection.append(Data(x=[1, 2, 3]))
        with patch('glue.qt.widgets.subset_facet.SubsetFacet.exec_'):
            act._do_action()
