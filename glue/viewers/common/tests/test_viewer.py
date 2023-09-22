from glue.config import layer_artist_maker
from glue.core.application_base import Application
from glue.core import Data
from glue.viewers.common.viewer import Viewer
from glue.viewers.common.layer_artist import LayerArtist
from glue.core.message import NumericalDataChangedMessage


def test_custom_layer_artist_maker():

    # This is a test of the infrastructure that allows users/developers to
    # define custom layer artist makers - i.e. functions that given a dataset
    # and a viewer can decide whether to create a custom layer artist rather
    # than the default types available for a given viewer.

    class CustomLayerArtist(LayerArtist):
        pass

    class CustomApplication(Application):
        def add_widget(self, *args, **kwargs):
            pass

    @layer_artist_maker('custom_maker')
    def custom_maker(viewer, data):
        if hasattr(data, 'custom'):
            return CustomLayerArtist(viewer.state, layer=data)

    app = CustomApplication()

    data1 = Data(x=[1, 2, 3], label='test1')
    data2 = Data(x=[1, 2, 3], label='test2')
    data2.custom = True

    app.data_collection.append(data1)
    app.data_collection.append(data2)

    viewer = app.new_data_viewer(Viewer)

    assert len(viewer.layers) == 0

    # NOTE: Check exact type, not using isinstance
    viewer.add_data(data1)
    assert len(viewer.layers) == 1
    assert type(viewer.layers[0]) is LayerArtist

    viewer.add_data(data2)
    assert len(viewer.layers) == 2
    assert type(viewer.layers[1]) is CustomLayerArtist


def test_viewer_update_data():
    """
    Test that we can have a LayerArtist that can respond
    to a NumericalDataChangedMessage.
    """

    class CustomUpdateLayerArtist(LayerArtist):

        def _on_components_changed(self, components_changed):
            self.called_component_limits = True
            self.num_components_changed = len(components_changed)

    class CustomApplication(Application):
        def add_widget(self, *args, **kwargs):
            pass

    @layer_artist_maker('custom_maker_for_update')
    def custom_maker_for_update(viewer, data):
        if hasattr(data, 'custom_for_update'):
            return CustomUpdateLayerArtist(viewer.state, layer=data)
        if hasattr(data.data, 'custom_for_update'):
            return CustomUpdateLayerArtist(viewer.state, layer=data)

    app = CustomApplication()

    data1 = Data(x=[1, 2, 3], label='test1')
    data1.custom_for_update = True

    app.data_collection.append(data1)

    viewer = app.new_data_viewer(Viewer)

    assert len(viewer.layers) == 0

    # NOTE: Check exact type, not using isinstance
    viewer.add_data(data1)
    assert len(viewer.layers) == 1
    assert type(viewer.layers[0]) is CustomUpdateLayerArtist

    msg = NumericalDataChangedMessage(data1, components_changed=[data1.id['x']])

    viewer._update_data(msg)
    assert viewer.layers[0].called_component_limits
    assert viewer.layers[0].num_components_changed == 1

    subset = data1.new_subset()
    subset.subset_state = data1.id['x'] > 2

    assert len(viewer.layers) == 2
    assert type(viewer.layers[1]) is CustomUpdateLayerArtist

    msg = NumericalDataChangedMessage(data1, components_changed=[data1.id['x']])
    viewer._update_data(msg)
    assert viewer.layers[1].called_component_limits
    assert viewer.layers[1].num_components_changed == 1
