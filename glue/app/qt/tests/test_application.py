# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function

import sys

import pytest
import numpy as np
from mock import patch, MagicMock

from qtpy import QtCore
from glue.core.data import Data
from glue.core.component_link import ComponentLink
from glue.core.data_collection import DataCollection
from glue.core.tests.test_state import Cloner, containers_equal, doubler, clone
from glue.tests.helpers import requires_ipython, PYSIDE2_INSTALLED  # noqa
from glue.viewers.image.qt import ImageViewer
from glue.viewers.scatter.qt import ScatterViewer
from glue.viewers.histogram.qt import HistogramViewer
from glue.utils.qt import get_qapp


from ..application import GlueApplication, GlueLogger


def tab_count(app):
    return app.tab_bar.count()


class TestGlueApplication(object):

    def setup_method(self, method):
        self.app = GlueApplication()
        self.app._create_terminal()

    def teardown_method(self, method):
        self.app.close()
        self.app = None

    def test_new_tabs(self):
        t0 = tab_count(self.app)
        self.app.new_tab()
        assert tab_count(self.app) == t0 + 1

    def test_save_session_ok(self):
        with patch.object(self.app, 'save_session') as save:
            with patch('qtpy.compat.getsavefilename') as fd:
                fd.return_value = '/tmp/junk', 'jnk'
                self.app._choose_save_session()
                save.assert_called_once_with('/tmp/junk.glu', include_data=False, absolute_paths=False)
                fd.reset_mock()
            save.reset_mock()

    def test_save_session_cancel(self):
        """shouldnt try to save file if no file name provided"""
        with patch.object(self.app, 'save_session') as save:
            with patch('glue.app.qt.application.compat.getsavefilename') as fd:
                fd.return_value = '', 'jnk'
                self.app._choose_save_session()
                assert save.call_count == 0
                fd.reset_mock()
            save.reset_mock()

    def test_choose_save_session_ioerror(self):
        """should show box on ioerror"""
        with patch('qtpy.compat.getsavefilename') as fd:
            if sys.version_info[0] == 2:
                mock_open = '__builtin__.open'
            else:
                mock_open = 'builtins.open'
            with patch(mock_open) as op:
                op.side_effect = IOError
                fd.return_value = '/tmp/junk', '/tmp/junk'
                with patch('qtpy.QtWidgets.QMessageBox') as mb:
                    self.app._choose_save_session()
                    assert mb.call_count == 1
                    mb.reset_mock()
                op.reset_mock()
            fd.reset_mock()

    @requires_ipython
    def test_terminal_present(self):
        """For good setups, terminal is available"""
        if not self.app.has_terminal():
            sys.stderr.write(self.app._terminal_exception)
            assert False

    def is_terminal_importable(self):
        import glue.qt.widgets.glue_terminal  # noqa

    @requires_ipython
    def test_toggle_terminal(self):
        with patch.object(self.app, '_terminal') as term:

            self.app._terminal = term

            term.isVisible.return_value = False
            self.app._button_ipython.click()
            assert term.show.call_count == 1

            term.isVisible.return_value = True
            self.app._button_ipython.click()
            assert term.hide.call_count == 1

            term.reset_mock()

    def test_close_tab(self):

        assert self.app.tab_widget.count() == 1
        assert self.app.tab_bar.tabText(0) == 'Tab 1'

        self.app.new_tab()
        assert self.app.tab_widget.count() == 2
        assert self.app.tab_bar.tabText(0) == 'Tab 1'
        assert self.app.tab_bar.tabText(1) == 'Tab 2'

        self.app.close_tab(0)
        assert self.app.tab_widget.count() == 1
        assert self.app.tab_bar.tabText(0) == 'Tab 2'

        # do not delete last tab
        self.app.close_tab(0)
        assert self.app.tab_widget.count() == 1

        # check that counter always goes up
        self.app.new_tab()
        assert self.app.tab_bar.tabText(0) == 'Tab 2'
        assert self.app.tab_bar.tabText(1) == 'Tab 3'

    def test_new_data_viewer_cancel(self):
        with patch('glue.app.qt.application.pick_class') as pc:
            pc.return_value = None

            ct = len(self.app.current_tab.subWindowList())

            self.app.choose_new_data_viewer()
            assert len(self.app.current_tab.subWindowList()) == ct

            pc.reset_mock()

    @pytest.mark.skipif('PYSIDE2_INSTALLED')
    def test_new_data_viewer_ok(self):

        with patch('glue.app.qt.application.pick_class') as pc:

            pc.return_value = ScatterViewer

            ct = len(self.app.current_tab.subWindowList())

            viewer = self.app.choose_new_data_viewer()
            assert len(self.app.current_tab.subWindowList()) == ct + 1
            viewer.close()
            # TODO: figure out why this doesn't work as expected
            # assert len(self.app.current_tab.subWindowList()) == ct

            pc.reset_mock()

    @pytest.mark.skipif('PYSIDE2_INSTALLED')
    def test_move(self):
        viewer = self.app.new_data_viewer(ScatterViewer)
        viewer.move(10, 20)
        assert viewer.position == (10, 20)
        viewer.close()

    @pytest.mark.skipif('PYSIDE2_INSTALLED')
    def test_resize(self):
        viewer = self.app.new_data_viewer(ScatterViewer)
        viewer.viewer_size = (100, 200)
        assert viewer.viewer_size == (100, 200)
        viewer.close()

    def test_new_data_defaults(self):

        with patch('glue.app.qt.application.pick_class') as pc:
            pc.return_value = None

            d2 = Data(x=np.array([[1, 2, 3], [4, 5, 6]]))
            d1 = Data(x=np.array([1, 2, 3]))

            self.app.choose_new_data_viewer(data=d1)
            args, kwargs = pc.call_args
            assert kwargs['default'] is ScatterViewer

            self.app.choose_new_data_viewer(data=d2)
            args, kwargs = pc.call_args
            assert kwargs['default'] is ImageViewer

            pc.reset_mock()

    def test_drop_load_data(self):

        load_data = MagicMock()
        load_session = MagicMock()
        self.app.load_data = load_data
        self.app.restore_session_and_close = load_session

        e = MagicMock()

        m = QtCore.QMimeData()
        m.setUrls([QtCore.QUrl('test.fits')])
        e.mimeData.return_value = m

        self.app.dropEvent(e)
        assert load_data.called_once_with('test.fits')
        assert load_session.call_count == 0

        load_data.reset_mock()

        m = QtCore.QMimeData()
        m.setUrls([QtCore.QUrl('test1.fits'), QtCore.QUrl('test2.fits')])
        e.mimeData.return_value = m
        self.app.dropEvent(e)
        assert load_data.called_once_with(['test1.fits', 'test2.fits'])
        assert load_session.call_count == 0

        load_data.reset_mock()

        m = QtCore.QMimeData()
        m.setUrls([QtCore.QUrl('test.glu')])
        e.mimeData.return_value = m
        self.app.dropEvent(e)
        assert load_data.call_count == 0
        assert load_session.called_once_with(['test.glu'])

        load_data.reset_mock()

        m = QtCore.QMimeData()
        m.setUrls([QtCore.QUrl('test.glu'), QtCore.QUrl('test.fits')])
        e.mimeData.return_value = m
        with patch('qtpy.QtWidgets.QMessageBox') as mb:
            self.app.dropEvent(e)
            assert mb.call_count == 1
            assert "When dragging and dropping files" in mb.call_args[0][2]
            mb.reset_mock()

        load_data.reset_mock()

    def test_subset_facet(self):
        # regression test for 335

        act = self.app._layer_widget._actions['facet']
        self.app.data_collection.append(Data(x=[1, 2, 3]))
        with patch('glue.dialogs.subset_facet.qt.SubsetFacet.exec_'):
            act._do_action()

    # FIXME: The following test fails and causes subsequent issues if run with
    #
    # pytest -s -v -x glue
    #
    # Need to investigate this, but for now, no solution other than skipping
    # the test.
    #
    # def test_suggest_merge(self):
    #
    #     x = Data(x=[1, 2, 3], label='x')
    #     y = Data(y=[4, 5, 6, 7], label='y')
    #     z = Data(z=[8, 9, 10], label='z')
    #
    #     self.app.data_collection.append(x)
    #     self.app.data_collection.append(y)
    #
    #     with process_dialog(delay=500, accept=True):
    #         result = self.app.add_datasets(self.app.data_collection, z)
    #
    #     np.testing.assert_equal(self.app.data_collection[0]['x'], [1, 2, 3])
    #     np.testing.assert_equal(self.app.data_collection[0]['z'], [8, 9, 10])
    #     np.testing.assert_equal(self.app.data_collection[1]['y'], [4, 5, 6, 7])


def check_clone_app(app):
    c = Cloner(app)
    copy = c.us.object('__main__')

    hub1 = app.session.hub
    hub2 = copy.session.hub

    assert len(hub1._subscriptions) == len(hub2._subscriptions)

    # data collections are the same
    for d1, d2 in zip(app.session.data_collection,
                      copy.session.data_collection):
        assert d1.label == d2.label
        for cid1, cid2 in zip(d1.components, d2.components):
            assert cid1.label == cid2.label

            # order of components unspecified if label collisions
            cid2 = c.get(cid1)
            np.testing.assert_array_almost_equal(d1[cid1, 0:1],
                                                 d2[cid2, 0:1], 3)

    # same data viewers, in the same tabs
    for tab1, tab2 in zip(app.viewers, copy.viewers):
        assert len(tab1) == len(tab2)
        for v1, v2 in zip(tab1, tab2):
            assert type(v1) == type(v2)
            # same window properties
            assert v1.viewer_size == v2.viewer_size
            assert v1.position == v2.position

            # same viewer-level properties (axis label, scaling, etc)
            assert set(v1.properties.keys()) == set(v2.properties.keys())
            for k in v1.properties:
                if hasattr(v1.properties[k], 'label'):
                    assert v1.properties[k].label == v2.properties[k].label
                else:
                    assert v1.properties[k] == v2.properties[k] or \
                        containers_equal(v1.properties[k], v2.properties[k])

            assert len(v1.layers) == len(v2.layers)
            for l1, l2 in zip(v1.layers, v2.layers):
                assert l1.layer.label == l2.layer.label  # same data/subset
                assert l1.layer.style == l2.layer.style

    return copy


class TestApplicationSession(object):

    def check_clone(self, app):
        return check_clone_app(app)

    def test_bare_application(self):
        app = GlueApplication()
        self.check_clone(app)

    def test_tab_names(self):
        app = GlueApplication()
        app.tab_bar.setTabText(0, 'Banana')
        assert app.tab_names == ['Banana']
        app2 = self.check_clone(app)
        assert app2.tab_names == ['Banana']

    def test_data_application(self):
        dc = DataCollection([Data(label='test',
                                  x=[1, 2, 3], y=[2, 3, 4])])
        app = GlueApplication(dc)
        self.check_clone(app)

    def test_links(self):
        d1 = Data(label='x', x=[1, 2, 3])
        d2 = Data(label='y', y=[3, 4, 8])
        dc = DataCollection([d1, d2])
        link = ComponentLink([d1.id['x']], d2.id['y'], doubler)
        dc.add_link(link)

        np.testing.assert_array_equal(d1['y'], [2, 4, 6])

        app = GlueApplication(dc)
        self.check_clone(app)

    @pytest.mark.skipif('PYSIDE2_INSTALLED')
    def test_scatter_viewer(self):
        d = Data(label='x', x=[1, 2, 3, 4, 5], y=[2, 3, 4, 5, 6])
        dc = DataCollection([d])
        app = GlueApplication(dc)
        w = app.new_data_viewer(ScatterViewer, data=d)
        copy1 = self.check_clone(app)

        dc.new_subset_group()
        dc.new_subset_group()
        assert len(w.layers) == 3
        l1, l2, l3 = w.layers
        l1.zorder, l2.zorder = l2.zorder, l1.zorder
        l3.visible = False
        assert l3.visible is False
        copy2 = self.check_clone(app)
        assert copy2.viewers[0][0].layers[-1].visible is False

        app.close()
        copy1.close()
        copy2.close()

    @pytest.mark.skipif('PYSIDE2_INSTALLED')
    def test_multi_tab(self):
        d = Data(label='hist', x=[[1, 2], [2, 3]])
        dc = DataCollection([d])

        app = GlueApplication(dc)
        w1 = app.new_data_viewer(HistogramViewer, data=d)
        app.new_tab()
        w2 = app.new_data_viewer(HistogramViewer, data=d)
        assert app.viewers == ((w1,), (w2,))

        copy = self.check_clone(app)

        app.close()
        copy.close()

    @pytest.mark.skipif('PYSIDE2_INSTALLED')
    def test_histogram(self):
        d = Data(label='hist', x=[[1, 2], [2, 3]])
        dc = DataCollection([d])

        app = GlueApplication(dc)
        w = app.new_data_viewer(HistogramViewer, data=d)
        copy1 = self.check_clone(app)

        dc.new_subset_group()
        assert len(w.layers) == 2
        copy2 = self.check_clone(app)

        w.nbins = 7
        copy3 = self.check_clone(app)

        app.close()
        copy1.close()
        copy2.close()
        copy3.close()

    def test_subset_groups_remain_synced_after_restore(self):
        # regrssion test for 352
        d = Data(label='hist', x=[[1, 2], [2, 3]])
        dc = DataCollection([d])
        dc.new_subset_group()
        app = GlueApplication(dc)

        app2 = clone(app)
        sg = app2.data_collection.subset_groups[0]
        assert sg.style.parent is sg

        sg.style.color = '#112233'
        assert sg.subsets[0].style.color == '#112233'

    @pytest.mark.skipif('PYSIDE2_INSTALLED')
    def test_deselect_tool_on_viewer_change(self):

        d = Data(label='hist', x=[[1, 2], [2, 3]])
        dc = DataCollection([d])

        app = GlueApplication(dc)
        v1 = app.new_data_viewer(HistogramViewer, data=d)
        v2 = app.new_data_viewer(HistogramViewer, data=d)

        assert v1.toolbar.active_tool is None
        assert v2.toolbar.active_tool is None

        v2.toolbar.active_tool = 'select:xrange'

        assert v1.toolbar.active_tool is None
        assert v2.toolbar.active_tool.tool_id == 'select:xrange'

        app.current_tab.activateNextSubWindow()

        assert v1.toolbar.active_tool is None
        assert v2.toolbar.active_tool is None

        v1.toolbar.active_tool = 'select:xrange'

        # Emit a signal without changing the active subWindow to make sure that
        # the tool doesn't get reset.
        app.current_tab.subWindowActivated.emit(app.current_tab.activeSubWindow())

        assert v1.toolbar.active_tool.tool_id == 'select:xrange'
        assert v2.toolbar.active_tool is None

        app.current_tab.activateNextSubWindow()

        assert v1.toolbar.active_tool is None
        assert v2.toolbar.active_tool is None

        app.close()


def test_logger_close():

    # Regression test to make sure that when closing an application, sys.stderr
    # no longer points to GlueLogger.

    app = GlueApplication()
    app.close()

    qapp = get_qapp()
    qapp.processEvents()

    assert not isinstance(sys.stderr, GlueLogger)


def test_reset_session_terminal():

    # Regression test to make sure that the terminal still works when
    # resetting a session

    app = GlueApplication()
    app2 = app._reset_session(warn=False)

    assert app2.has_terminal()

    app.close()
    app2.close()


def test_open_session_terminal(tmpdir):

    # Regression test to make sure that the terminal still works when
    # opening a previous session

    session_file = tmpdir.join('test.glu').strpath

    app = GlueApplication()
    app.save_session(session_file)

    app2 = app.restore_session(session_file)

    assert app2.has_terminal()

    app.close()
    app2.close()
