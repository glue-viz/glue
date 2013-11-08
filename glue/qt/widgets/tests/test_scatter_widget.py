#pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103
from distutils.version import LooseVersion  # pylint:disable=W0611

import pytest

from ..scatter_widget import ScatterWidget
from .... import core

from matplotlib import __version__ as mpl_version  # pylint:disable=W0611


class TestScatterWidget(object):

    def setup_method(self, method):
        self.hub = core.hub.Hub()
        self.d1 = core.Data(x=[1, 2, 3], y=[2, 3, 4],
                            z=[3, 4, 5], w=[4, 5, 6])
        self.d1.label = 'd1'
        self.d2 = core.Data(x=[1, 2, 3], y=[2, 3, 4],
                            z=[3, 4, 5], w=[4, 5, 6])
        self.d2.label = 'd2'
        self.data = [self.d1, self.d2]
        self.collect = core.data_collection.DataCollection(list(self.data))
        self.widget = ScatterWidget(self.collect)
        self.connect_to_hub()

    def teardown_method(self, method):
        self.assert_widget_synced()

    def assert_widget_synced(self):
        cl = self.widget.client
        w = self.widget
        assert abs(w.xmin - cl.xmin) < 1e-3
        assert abs(w.xmax - cl.xmax) < 1e-3
        assert w.xlog == cl.xlog
        assert w.ylog == cl.ylog
        assert w.xflip == cl.xflip
        assert w.yflip == cl.yflip
        assert abs(w.ymin - cl.ymin) < 1e-3
        assert abs(w.ymax - cl.ymax) < 1e-3

    def connect_to_hub(self):
        self.widget.register_to_hub(self.hub)
        self.collect.register_to_hub(self.hub)

    def add_layer_via_hub(self):
        layer = self.data[0]
        layer.label = 'Test Layer'
        self.collect.append(layer)
        return layer

    def add_layer_via_method(self, index=0):
        layer = self.data[index]
        self.widget.add_data(layer)
        return layer

    def plot_data(self, layer):
        """ Return the data bounds for a given layer (data or subset)
        Output format: [xmin, xmax], [ymin, ymax]
        """
        client = self.widget.client
        x, y = client.artists[layer][0].get_data()
        assert x.size > 0
        assert y.size > 0
        xmin = x.min()
        xmax = x.max()
        ymin = y.min()
        ymax = y.max()
        return [xmin, xmax], [ymin, ymax]

    def plot_limits(self):
        """ Return the plot limits
        Output format [xmin, xmax], [ymin, ymax]
        """
        ax = self.widget.client.axes
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        return xlim, ylim

    def assert_layer_inside_limits(self, layer):
        """Assert that points of a layer are within plot limits """
        xydata = self.plot_data(layer)
        xylimits = self.plot_limits()
        assert xydata[0][0] >= xylimits[0][0]
        assert xydata[1][0] >= xylimits[1][0]
        assert xydata[0][1] <= xylimits[0][1]
        assert xydata[1][1] <= xylimits[1][1]

    def is_layer_present(self, layer):
        return self.widget.client.is_layer_present(layer)

    def is_layer_visible(self, layer):
        return self.widget.client.is_visible(layer)

    def test_rescaled_on_init(self):
        layer = self.add_layer_via_method()
        self.assert_layer_inside_limits(layer)

    def test_hub_data_add_is_ignored(self):
        layer = self.add_layer_via_hub()
        assert not self.widget.client.is_layer_present(layer)

    def test_valid_add_data_via_method(self):
        layer = self.add_layer_via_method()
        assert self.is_layer_present(layer)

    def test_add_first_data_updates_combos(self):
        layer = self.add_layer_via_method()
        xatt = str(self.widget.ui.xAxisComboBox.currentText())
        yatt = str(self.widget.ui.yAxisComboBox.currentText())
        assert xatt is not None
        assert yatt is not None

    def test_flip_x(self):
        layer = self.add_layer_via_method()
        self.widget.xflip = True
        assert self.widget.client.xflip
        self.widget.xflip = False
        assert not self.widget.client.xflip

    def test_flip_y(self):
        layer = self.add_layer_via_method()
        self.widget.yflip = True
        assert self.widget.client.yflip
        self.widget.yflip = False
        assert not self.widget.client.yflip

    def test_log_x(self):
        layer = self.add_layer_via_method()
        self.widget.xlog = True
        assert self.widget.client.xlog
        self.widget.xlog = False
        assert not self.widget.client.xlog

    def test_log_y(self):
        self.widget.ylog = True
        assert self.widget.client.ylog
        self.widget.ylog = False
        assert not self.widget.client.ylog

    def test_double_add_ignored(self):
        layer = self.add_layer_via_method()
        nobj = self.widget.ui.xAxisComboBox.count()
        layer = self.add_layer_via_method()
        assert self.widget.ui.xAxisComboBox.count() == nobj

    def test_subsets_dont_duplicate_fields(self):
        layer = self.add_layer_via_method()
        nobj = self.widget.ui.xAxisComboBox.count()
        subset = layer.new_subset()
        subset.register()
        assert self.widget.ui.xAxisComboBox.count() == nobj

    def test_correct_title_single_data(self):
        ct = self.widget.client.layer_count
        assert ct == 0
        layer = self.add_layer_via_method()
        ct = self.widget.client.layer_count
        assert ct == 1
        assert len(layer.label) > 0
        assert self.widget.windowTitle() == layer.label

    def test_title_updates_with_label_change(self):
        layer = self.add_layer_via_method()
        assert layer.hub is self.hub
        layer.label = "changed label"
        assert self.widget.windowTitle() == layer.label

    def test_title_updates_with_second_data(self):
        l1 = self.add_layer_via_method(0)
        l2 = self.add_layer_via_method(1)
        expected = '%s | %s' % (l1.label, l2.label)
        self.widget.windowTitle() == expected

    def test_second_data_add_preserves_plot_variables(self):
        l1 = self.add_layer_via_method(0)
        self.widget.ui.xAxisComboBox.setCurrentIndex(3)
        self.widget.ui.yAxisComboBox.setCurrentIndex(2)
        l2 = self.add_layer_via_method(1)

        assert self.widget.ui.xAxisComboBox.currentIndex() == 3
        assert self.widget.ui.yAxisComboBox.currentIndex() == 2

    def test_set_limits(self):
        l1 = self.add_layer_via_method(0)
        w = self.widget
        c = self.widget.client
        ax = self.widget.client.axes

        print w.xmin, w.xmax, w.ymin, w.ymax
        print c.xmin, c.xmax, c.ymin, c.ymax
        print ax.get_xlim(), ax.get_ylim()

        self.widget.xmax = 20
        print w.xmin, w.xmax, w.ymin, w.ymax
        print c.xmin, c.xmax, c.ymin, c.ymax
        print ax.get_xlim(), ax.get_ylim()

        self.widget.xmin = 10
        print w.xmin, w.xmax, w.ymin, w.ymax
        print c.xmin, c.xmax, c.ymin, c.ymax
        print ax.get_xlim(), ax.get_ylim()

        self.widget.ymax = 40
        print w.xmin, w.xmax, w.ymin, w.ymax
        print c.xmin, c.xmax, c.ymin, c.ymax
        print ax.get_xlim(), ax.get_ylim()

        self.widget.ymin = 30
        print w.xmin, w.xmax, w.ymin, w.ymax
        print c.xmin, c.xmax, c.ymin, c.ymax
        print ax.get_xlim(), ax.get_ylim()

        assert self.widget.client.axes.get_xlim() == (10, 20)
        assert self.widget.client.axes.get_ylim() == (30, 40)
        assert float(self.widget.ui.xmin.text()) == 10
        assert float(self.widget.ui.xmax.text()) == 20
        assert float(self.widget.ui.ymin.text()) == 30
        assert float(self.widget.ui.ymax.text()) == 40

    def test_widget_props_synced_with_client(self):

        self.widget.client.xmax = 100
        assert self.widget.xmax == 100
        self.widget.client.ymax = 200
        assert self.widget.ymax == 200

        self.widget.client.xmin = 10
        assert self.widget.xmin == 10

        self.widget.client.ymin = 30
        assert self.widget.ymin == 30

    @pytest.mark.xfail("LooseVersion(mpl_version) <= LooseVersion('1.1.0')")
    def test_labels_sync_with_plot_limits(self):
        """For some reason, manually calling draw() doesnt trigger the
        draw_event in MPL 1.1.0. Ths functionality nevertheless seems
        to work when actually using Glue"""

        l1 = self.add_layer_via_method(0)
        self.widget.client.axes.set_xlim((3, 4))
        self.widget.client.axes.set_ylim((5, 6))
        self.widget.client.axes.figure.canvas.draw()

        assert float(self.widget.ui.xmin.text()) == 3
        assert float(self.widget.ui.xmax.text()) == 4
        assert float(self.widget.ui.ymin.text()) == 5
        assert float(self.widget.ui.ymax.text()) == 6

    def assert_component_present(self, label):
        ui = self.widget.ui
        for combo in [ui.xAxisComboBox, ui.yAxisComboBox]:
            atts = [combo.itemText(i) for i in range(combo.count())]
            assert label in atts

    def test_component_change_syncs_with_combo(self):
        l1 = self.add_layer_via_method()
        cid = l1.add_component(l1[l1.components[0]], 'testing')
        self.assert_component_present('testing')

    def test_swap_axes(self):
        l1 = self.add_layer_via_method()
        cl = self.widget.client
        cl.xlog, cl.xflip = True, True
        cl.ylog, cl.yflip = False, False

        x, y = cl.xatt, cl.yatt

        self.widget.swap_axes()
        assert (cl.xlog, cl.xflip) == (False, False)
        assert (cl.ylog, cl.yflip) == (True, True)
        assert (cl.xatt, cl.yatt) == (y, x)

    def test_hidden(self):
        l1 = self.add_layer_via_method()
        xcombo = self.widget.ui.xAxisComboBox

        self.widget.hidden = False
        assert xcombo.count() == 4
        self.widget.hidden = True
        assert xcombo.count() == 6
        self.widget.hidden = False
        assert xcombo.count() == 4

    def test_add_subset_preserves_plot_variables(self):
        l1 = self.add_layer_via_method(0)
        print self.widget.client.layer_count

        self.widget.ui.xAxisComboBox.setCurrentIndex(3)
        self.widget.ui.yAxisComboBox.setCurrentIndex(2)
        assert self.widget.ui.xAxisComboBox.currentIndex() == 3
        assert self.widget.ui.yAxisComboBox.currentIndex() == 2

        s = self.data[1].new_subset(label='new')
        self.widget.add_subset(s)

        assert self.widget.ui.xAxisComboBox.currentIndex() == 3
        assert self.widget.ui.yAxisComboBox.currentIndex() == 2
