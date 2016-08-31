# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function

import os
import time

import pytest
import numpy as np
from mock import MagicMock

from glue.viewers.common.qt.tool import Tool
from glue.utils.qt import get_qapp
from glue.app.qt.tests.test_application import TestApplicationSession
from glue import core
from glue.app.qt import GlueApplication

from glue.core.tests.util import simple_session
from ..viewer_widget import ImageWidget


os.environ['GLUE_TESTING'] = 'True'

CI = os.environ.get('CI', 'false').lower() == 'true'
TRAVIS_LINUX = os.environ.get('TRAVIS_OS_NAME', None) == 'linux'


class _TestImageWidgetBase(object):
    widget_cls = None

    def setup_method(self, method):
        self.session = simple_session()
        self.hub = self.session.hub
        self.collect = self.session.data_collection

        self.im = core.Data(label='im',
                            x=[[1, 2], [3, 4]],
                            y=[[2, 3], [4, 5]])
        self.cube = core.Data(label='cube',
                              x=[[[1, 2], [3, 4]], [[1, 2], [3, 4]]],
                              y=[[[1, 2], [3, 4]], [[1, 2], [3, 4]]])
        self.widget = self.widget_cls(self.session)
        self.connect_to_hub()
        self.collect.append(self.im)
        self.collect.append(self.cube)

    def assert_title_correct(self):
        expected = "%s - %s" % (self.widget.data.label,
                                self.widget.attribute.label)
        assert self.widget.windowTitle() == expected

    def connect_to_hub(self):
        self.widget.register_to_hub(self.hub)
        self.collect.register_to_hub(self.hub)

    def _test_widget_synced_with_collection(self):
        dc = self.widget.ui.displayDataCombo
        assert dc.count() == len(self.collect)
        for data in self.collect:
            label = data.label
            pos = dc.findText(label)
            assert pos >= 0
            assert dc.itemData(pos) is data

    def test_synced_on_init(self):
        self._test_widget_synced_with_collection()

    def test_multi_add_ignored(self):
        """calling add_data multiple times doesn't corrupt data combo"""
        self.widget.add_data(self.collect[0])
        self.widget.add_data(self.collect[0])
        self._test_widget_synced_with_collection()

    def test_synced_on_remove(self):
        self.collect.remove(self.cube)
        self._test_widget_synced_with_collection()

    def test_window_title_matches_data(self):
        self.widget.add_data(self.collect[0])
        self.assert_title_correct()

    def test_window_title_updates_on_label_change(self):
        self.connect_to_hub()
        self.widget.add_data(self.collect[0])
        self.collect[0].label = 'Changed'
        self.assert_title_correct()

    def test_window_title_updates_on_component_change(self):
        self.connect_to_hub()
        self.widget.add_data(self.collect[0])
        self.widget.ui.attributeComboBox.setCurrentIndex(1)
        self.assert_title_correct()

    def test_data_combo_updates_on_change(self):
        self.connect_to_hub()
        self.widget.add_data(self.collect[0])
        self.collect[0].label = 'changed'
        data_labels = self._data_combo_labels()
        assert self.collect[0].label in data_labels

    def test_data_not_added_on_init(self):
        w = ImageWidget(self.session)
        assert self.im not in w.client.artists

    def test_selection_switched_on_add(self):
        w = self.widget_cls(self.session)
        assert self.im not in w.client.artists
        w.add_data(self.im)
        assert self.im in w.client.artists
        w.add_data(self.cube)
        assert self.im not in w.client.artists
        assert self.cube in w.client.artists

    def test_component_add_updates_combo(self):
        self.widget.add_data(self.im)
        self.im.add_component(self.im[self.im.components[0]], 'testing')
        combo = self.widget.ui.attributeComboBox
        cids = [combo.itemText(i) for i in range(combo.count())]
        assert 'testing' in cids

    def test_image_correct_on_init_if_first_attribute_hidden(self):
        """Regression test for #127"""
        self.im.components[0]._hidden = True
        self.widget.add_data(self.im)
        combo = self.widget.ui.attributeComboBox
        index = combo.currentIndex()
        assert self.widget.client.display_attribute is combo.itemData(index)

    def _data_combo_labels(self):
        combo = self.widget.ui.displayDataCombo
        return [combo.itemText(i) for i in range(combo.count())]

    def test_plugins_closed_when_viewer_closed(self):

        # Regression test for #518
        self.widget.add_data(self.im)

        class TestTool(Tool):
            icon = 'glue_lasso'
            tool_id = 'test'
            def close(self):
                self.closed = True

        test_mode = TestTool(self.widget)
        self.widget.toolbar.add_tool(test_mode)
        self.widget.close()
        assert test_mode.closed

class TestImageWidget(_TestImageWidgetBase):
    widget_cls = ImageWidget

    def test_intensity_label(self):
        self.widget.add_data(self.im)
        att = self.widget.attribute
        intensity = self.im[att][1, 0]
        x, y = self.widget.client.axes.transData.transform([(0.5, 1.5)])[0]
        assert self.widget._intensity_label(x, y) == 'data: %s' % intensity

    def test_paint(self):
        # make sure paint Events don't trigger any errors
        self.widget.add_data(self.im)
        self.widget.show()
        self.widget.close()

    def test_enable_rgb_doesnt_close_viewer(self):
        # regression test for #446
        def fail():
            assert False

        self.widget.add_data(self.im)
        self.widget._layer_artist_container.on_empty(fail)
        self.widget.rgb_mode = True
        self.widget.rgb_mode = False

    def test_rgb_mode_toggle_aspect(self):

        # Regression test: make sure that aspect ratio is preserved when
        # toggling the RGB mode.

        self.widget.add_data(self.im)

        client = self.widget.client

        assert client.display_aspect == 'equal'
        for artist in client.artists:
            assert artist.aspect == 'equal'

        self.widget.rgb_mode = True

        assert client.display_aspect == 'equal'
        for artist in client.artists:
            assert artist.aspect == 'equal'

        self.widget.rgb_mode = False
        self.widget.aspect_ratio = 'auto'

        assert client.display_aspect == 'auto'
        for artist in client.artists:
            assert artist.aspect == 'auto'

        self.widget.rgb_mode = True

        assert client.display_aspect == 'auto'
        for artist in client.artists:
            assert artist.aspect == 'auto'

    @pytest.mark.skipif("CI and not TRAVIS_LINUX")
    def test_resize(self):

        # Regression test for a bug that caused images to not be shown at
        # full resolution after resizing a widget.

        # This test only runs correctly on Linux on Travis at the moment,
        # although it works fine locally on MacOS X. I have not yet tracked
        # down the cause of the failure, but essentially the first time that
        # self.widget.client._view_window is accessed below, it is still None.
        # The issue is made more complicated by the fact that whether the test
        # succeeds or not (after removing code in ImageWidget) depends on
        # whether another test is run first - in particular I tried with
        # test_resize from test_application.py. I was able to then get the
        # test here to pass if the other test_resize was *not* run first.
        # This should be investigated more in future, but for now, it's most
        # important that we get the fix in.

        # What appears to happen when the test fails is that the QTimer gets
        # started but basically never ends up triggering the timeout.

        large = core.Data(label='largeim', x=np.random.random((1024, 1024)))
        self.collect.append(large)

        app = get_qapp()
        self.widget.add_data(large)
        self.widget.show()

        self.widget.resize(300, 300)
        time.sleep(0.5)
        app.processEvents()

        extx0, exty0 = self.widget.client._view_window[4:]

        # While resizing, the view window should not change until we've
        # waited for a bit, to avoid resampling the data every time.
        for res in range(10):

            self.widget.resize(300 + res * 30, 300 + res * 30)
            app.processEvents()

            extx, exty = self.widget.client._view_window[4:]
            assert extx == extx0
            assert exty == exty0

        time.sleep(0.5)
        app.processEvents()

        extx, exty = self.widget.client._view_window[4:]
        assert extx != extx0
        assert exty != exty0

        self.widget.close()


class TestStateSave(TestApplicationSession):

    def setup_method(self, method):
        LinkSame = core.link_helpers.LinkSame

        d = core.Data(label='im', x=[[1, 2], [2, 3]], y=[[2, 3], [4, 5]])
        d2 = core.Data(label='cat',
                       x=[0, 1, 0, 1],
                       y=[0, 0, 1, 1],
                       z=[1, 2, 3, 4])

        dc = core.DataCollection([d, d2])
        dc.add_link(LinkSame(d.get_pixel_component_id(0), d2.id['x']))
        dc.add_link(LinkSame(d.get_pixel_component_id(1), d2.id['y']))

        app = GlueApplication(dc)
        w = app.new_data_viewer(ImageWidget, data=d)
        self.d = d
        self.app = app
        self.w = w
        self.d2 = d2
        self.dc = dc

    def test_image_viewer(self):
        self.check_clone(self.app)

    def test_subset(self):
        d, w, app = self.d, self.w, self.app
        self.dc.new_subset_group()
        assert len(w.layers) == 2
        self.check_clone(app)

    def test_scatter_layer(self):
        # add scatter layer
        d, w, app, d2 = self.d, self.w, self.app, self.d2
        w.add_data(d2)
        assert len(w.layers) == 2
        self.check_clone(app)

    def test_cube(self):
        d = core.Data(label='cube',
                      x=np.zeros((2, 2, 2)))
        dc = core.DataCollection([d])
        app = GlueApplication(dc)
        w = app.new_data_viewer(ImageWidget, d)
        w.slice = ('x', 'y', 1)
        assert w.slice == ('x', 'y', 1)

        c = self.check_clone(app)
        w2 = c.viewers[0][0]
        assert w2.ui.slice.slice == w.slice

    def test_rgb_layer(self):
        d, w, app = self.d, self.w, self.app

        x = d.id['x']
        y = d.id['y']
        w.client.display_data = d
        w.rgb_mode = True
        w.rgb_viz = (True, True, False)
        w.ratt = x
        w.gatt = y
        w.batt = x

        clone = self.check_clone(app)

        w = clone.viewers[0][0]

        assert w.rgb_viz == (True, True, False)
        assert w.rgb_mode
        assert w.ratt.label == 'x'
        assert w.gatt.label == 'y'
        assert w.batt.label == 'x'


def test_combo_box_updates():

    # Regression test for a bug that caused combo boxes to not be updated
    # correctly when switching between different datasets.

    session = simple_session()
    hub = session.hub
    dc = session.data_collection

    data1 = core.Data(label='im1',
                      x=[[1, 2], [3, 4]],
                      y=[[2, 3], [4, 5]])

    data2 = core.Data(label='im2',
                      a=[[1, 2], [3, 4]],
                      b=[[2, 3], [4, 5]])

    dc.append(data1)
    dc.append(data2)

    widget = ImageWidget(session)
    widget.register_to_hub(hub)

    widget.add_data(data1)

    assert widget.client.display_data is data1

    assert widget.data.label == 'im1'
    assert widget.attribute.label == 'x'

    widget.add_data(data2)

    assert widget.client.display_data is data2

    assert widget.data.label == 'im2'
    assert widget.attribute.label == 'a'

    widget.attribute = data2.find_component_id('b')

    with pytest.raises(ValueError) as exc:
        widget.attribute = data1.find_component_id('x')
    assert exc.value.args[0] == "Cannot find data 'x' in combo box"

    widget.data = data1
    assert widget.attribute.label == 'x'

    widget.attribute = data1.find_component_id('y')

    with pytest.raises(ValueError) as exc:
        widget.attribute = data2.find_component_id('a')
    assert exc.value.args[0] == "Cannot find data 'a' in combo box"

    assert widget.client.display_data is data1


del TestApplicationSession
