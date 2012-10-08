#pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103
from PyQt4.QtGui import QApplication, QMainWindow
from PyQt4.QtTest import QTest
from PyQt4.QtCore import Qt

from ..scatter_widget import ScatterWidget

from ....tests import example_data
from .... import core


class TestScatterWidget(object):

    def setup_method(self, method):
        self.hub = core.hub.Hub()
        self.data = example_data.test_data()
        self.collect = core.data_collection.DataCollection(list(self.data))
        self.widget = ScatterWidget(self.collect)
        self.win = QMainWindow()
        self.win.setCentralWidget(self.widget)
        self.connect_to_hub()
        self.win.show()

    def teardown_method(self, method):
        self.win.close()

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
        x, y = client.managers[layer].get_data()
        xmin = x.min()
        xmax = x.max()
        ymin = y.min()
        ymax = y.max()
        return [xmin, xmax], [ymin, ymax]

    def plot_limits(self):
        """ Return the plot limits
        Output format [xmin, xmax], [ymin, ymax]
        """
        ax = self.widget.client.ax
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        return xlim, ylim
        return

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
        QTest.mouseClick(self.widget.ui.xFlipCheckBox, Qt.LeftButton)
        assert self.widget.client.is_xflip()
        QTest.mouseClick(self.widget.ui.xFlipCheckBox, Qt.LeftButton)
        assert not self.widget.client.is_xflip()

    def test_flip_y(self):
        layer = self.add_layer_via_method()
        QTest.mouseClick(self.widget.ui.yFlipCheckBox, Qt.LeftButton)
        assert self.widget.client.is_yflip()
        QTest.mouseClick(self.widget.ui.yFlipCheckBox, Qt.LeftButton)
        assert not self.widget.client.is_yflip()

    def test_log_x(self):
        layer = self.add_layer_via_method()
        QTest.mouseClick(self.widget.ui.xLogCheckBox, Qt.LeftButton)
        assert self.widget.client.is_xlog()
        QTest.mouseClick(self.widget.ui.xLogCheckBox, Qt.LeftButton)
        assert not self.widget.client.is_xlog()

    def test_log_y(self):
        QTest.mouseClick(self.widget.ui.yLogCheckBox, Qt.LeftButton)
        assert self.widget.client.is_ylog()
        QTest.mouseClick(self.widget.ui.yLogCheckBox, Qt.LeftButton)
        assert not self.widget.client.is_ylog()

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
