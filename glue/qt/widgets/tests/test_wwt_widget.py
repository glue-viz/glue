from mock import MagicMock

from ....core import Data, DataCollection, Hub, message
from ..wwt_widget import WWTWidget


class TestWWTWidget(object):

    def setup_method(self, method):
        self.d = Data(x=[1, 2, 3], y=[2, 3, 4])
        self.dc = DataCollection([self.d])
        self.widget = WWTWidget(self.dc, webdriver_class=MagicMock)

    def register(self):
        self.hub = Hub()
        self.dc.register_to_hub(self.hub)
        self.widget.register_to_hub(self.hub)

    def test_add_data(self):
        self.widget.ra = self.d.id['x']
        self.widget.dec = self.d.id['y']
        self.widget.add_data(self.d)
        assert self.d in self.widget

    def test_double_add_ignored(self):
        self.widget.ra = self.d.id['x']
        self.widget.dec = self.d.id['y']
        assert len(self.widget) == 0
        self.widget.add_data(self.d)
        assert len(self.widget) == 1
        self.widget.add_data(self.d)
        assert len(self.widget) == 1

    def test_updated_on_data_update_message(self):
        self.register()
        self.widget.ra = self.d.id['x']
        self.widget.dec = self.d.id['y']
        self.widget.add_data(self.d)
        layer = self.widget._container[self.d][0]
        layer.update = MagicMock()
        self.d.style.color = 'green'
        assert layer.update.call_count == 1

    def test_updated_on_subset_update_message(self):
        self.register()
        self.widget.ra = self.d.id['x']
        self.widget.dec = self.d.id['y']
        s = self.d.new_subset()
        self.widget.add_subset(s)
        layer = self.widget._container[s][0]
        layer.update = MagicMock()
        s.style.color = 'green'
        assert layer.update.call_count == 1

    def test_remove_data(self):
        self.register()
        self.widget.ra = self.d.id['x']
        self.widget.dec = self.d.id['y']
        self.widget.add_subset(self.d)
        layer = self.widget._container[self.d][0]

        layer.clear = MagicMock()
        self.hub.broadcast(message.DataCollectionDeleteMessage(self.dc,
                                                               data=self.d))
        assert layer.clear.call_count == 1
        assert self.d not in self.widget

    def test_remove_subset(self):
        self.register()
        self.widget.ra = self.d.id['x']
        self.widget.dec = self.d.id['y']
        s = self.d.new_subset()
        self.widget.add_subset(s)

        layer = self.widget._container[s][0]
        layer.clear = MagicMock()

        self.hub.broadcast(message.SubsetDeleteMessage(s))

        assert layer.clear.call_count == 1
        assert self.d not in self.widget

    def test_subsets_added_with_data(self):
        self.widget.ra = self.d.id['x']
        self.widget.dec = self.d.id['y']
        s = self.d.new_subset()
        self.widget.add_data(self.d)

        assert s in self.widget

    def test_subsets_live_added(self):
        self.register()
        self.widget.ra = self.d.id['x']
        self.widget.dec = self.d.id['y']
        self.widget.add_data(self.d)
        s = self.d.new_subset()

        assert s in self.widget

    def test_subsets_not_live_added_if_data_not_present(self):
        self.register()
        s = self.d.new_subset()
        assert s not in self.widget

    def test_updated_on_add(self):
        self.register()
        self.widget.ra = self.d.id['x']
        self.widget.dec = self.d.id['y']

        self.widget._update_layer = MagicMock()
        self.widget.add_data(self.d)
        self.widget._update_layer.assert_called_once_with(self.d)

    def test_updated_on_coordinate_change(self):
        self.register()
        self.widget.ra = self.d.id['x']
        self.widget.dec = self.d.id['y']

        self.widget.add_data(self.d)
        artist = self.widget._container[self.d][0]

        self.widget._update_layer = MagicMock()
        self.widget.ra = self.d.id['y']
        self.widget._update_layer.call_count > 0
