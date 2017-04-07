# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function

import pytest

from glue.core import Data
from glue.core.tests.util import simple_session
from glue.core.exceptions import IncompatibleDataException


class MatplotlibDrawCounter(object):
    def __init__(self, figure):
        self.draw_count = 0
        figure.canvas.mpl_connect('draw_event', self.on_draw)
    def on_draw(self, event):
        self.draw_count += 1


class BaseTestMatplotlibDataViewer(object):
    """
    Base class to test viewers based on MatplotlibDataViewer. This only runs
    a subset of tests that relate to functionality implemented in
    MatplotlibDataViewer and specific viewers are responsible for implementing
    a more complete test suite.

    Viewers based on this should inherit from this test class and define the
    following attributes:

    * ``data``: an instance of a data object that works by default in the viewer
    * ``viewer_cls``: the viewer class

    It is then safe to assume that ``data_collection``, ``viewer``, and ``hub``
    are defined when writing tests.
    """

    def setup_method(self, method):

        self.data = self.init_data()

        self.session = simple_session()
        self.hub = self.session.hub

        self.data_collection = self.session.data_collection
        self.data_collection.append(self.data)

        self.viewer = self.viewer_cls(self.session)

        self.data_collection.register_to_hub(self.hub)
        self.viewer.register_to_hub(self.hub)

    def init_subset(self):
        cid = self.data.visible_components[0]
        self.data_collection.new_subset_group('subset 1', cid > 0)

    def teardown_method(self, method):
        self.viewer.close()

    def test_add_data(self):

        # Add a dataset with no subsets and make sure the appropriate layer
        # state and layer artists are created

        self.viewer.add_data(self.data)

        assert len(self.viewer.layers) == 1
        assert self.viewer.layers[0].layer is self.data

        assert len(self.viewer.viewer_state.layers) == 1
        assert self.viewer.viewer_state.layers[0].layer is self.data

    def test_add_data_with_subset(self):

        # Make sure that if subsets are present in the data, they are added
        # automatically

        self.init_subset()
        self.viewer.add_data(self.data)

        assert len(self.viewer.layers) == 2
        assert self.viewer.layers[0].layer is self.data
        assert self.viewer.layers[1].layer is self.data.subsets[0]

        assert len(self.viewer.viewer_state.layers) == 2
        assert self.viewer.viewer_state.layers[0].layer is self.data
        assert self.viewer.viewer_state.layers[1].layer is self.data.subsets[0]

    def test_adding_subset_adds_data(self):

        # TODO: in future consider whether we want to deprecate this behavior

        self.init_subset()
        self.viewer.add_subset(self.data.subsets[0])

        assert len(self.viewer.layers) == 2
        assert self.viewer.layers[0].layer is self.data
        assert self.viewer.layers[1].layer is self.data.subsets[0]

        assert len(self.viewer.viewer_state.layers) == 2
        assert self.viewer.viewer_state.layers[0].layer is self.data
        assert self.viewer.viewer_state.layers[1].layer is self.data.subsets[0]

    def test_add_data_then_subset(self):

        # Make sure that if a subset is created in a dataset that has already
        # been added to a viewer, the subset gets added

        self.viewer.add_data(self.data)

        assert len(self.viewer.layers) == 1
        assert self.viewer.layers[0].layer is self.data

        assert len(self.viewer.viewer_state.layers) == 1
        assert self.viewer.viewer_state.layers[0].layer is self.data

        self.init_subset()

        assert len(self.viewer.layers) == 2
        assert self.viewer.layers[0].layer is self.data
        assert self.viewer.layers[1].layer is self.data.subsets[0]

        assert len(self.viewer.viewer_state.layers) == 2
        assert self.viewer.viewer_state.layers[0].layer is self.data
        assert self.viewer.viewer_state.layers[1].layer is self.data.subsets[0]

    def init_draw_count(self):
        self.mpl_counter = MatplotlibDrawCounter(self.viewer.axes.figure)

    @property
    def draw_count(self):
        return self.mpl_counter.draw_count

    def test_single_draw(self):
        # Make sure that the number of draws is kept to a minimum
        self.init_draw_count()
        self.init_subset()
        assert self.draw_count == 0
        self.viewer.add_data(self.data)
        assert self.draw_count == 1

    def test_update_subset(self):

        self.init_draw_count()

        # Check that updating a subset causes the plot to be updated

        self.init_subset()

        assert self.draw_count == 0

        self.viewer.add_data(self.data)

        count_before = self.draw_count

        # Change the subset
        cid = self.data.visible_components[0]
        self.data.subsets[0].subset_state = cid > 1

        # Make sure the figure has been redrawn
        assert self.draw_count - count_before > 0

    def test_double_add_ignored(self):
        self.viewer.add_data(self.data)
        assert len(self.viewer.viewer_state.layers) == 1
        self.viewer.add_data(self.data)
        assert len(self.viewer.viewer_state.layers) == 1

    def test_removing_data_removes_layer_state(self):
        # Removing data from data collection should remove data from viewer
        self.viewer.add_data(self.data)
        assert len(self.viewer.viewer_state.layers) == 1
        self.data_collection.remove(self.data)
        assert len(self.viewer.viewer_state.layers) == 0

    def test_removing_data_removes_subsets(self):
        # Removing data from data collection should remove subsets from viewer
        self.init_subset()
        self.viewer.add_data(self.data)
        assert len(self.viewer.viewer_state.layers) == 2
        self.data_collection.remove(self.data)
        assert len(self.viewer.viewer_state.layers) == 0

    def test_removing_subset_removes_layers(self):

        # Removing a layer artist removes the corresponding layer state. We need
        # to do this with a subset otherwise the viewer is closed

        self.init_subset()
        self.viewer.add_data(self.data)

        assert len(self.viewer.layers) == 2
        assert len(self.viewer.viewer_state.layers) == 2

        self.data_collection.remove_subset_group(self.data_collection.subset_groups[0])

        assert len(self.viewer.layers) == 1
        assert self.viewer.layers[0].layer is self.data

        assert len(self.viewer.viewer_state.layers) == 1
        assert self.viewer.viewer_state.layers[0].layer is self.data

    def test_removing_layer_artist_removes_layer_state(self):

        # Removing a layer artist removes the corresponding layer state. We need
        # to do this with a subset otherwise the viewer is closed

        self.init_subset()
        self.viewer.add_data(self.data)

        assert len(self.viewer.layers) == 2
        assert len(self.viewer.viewer_state.layers) == 2

        # self.layers is a copy so we need to remove from the original list
        self.viewer._layer_artist_container.remove(self.viewer.layers[1])

        assert len(self.viewer.layers) == 1
        assert self.viewer.layers[0].layer is self.data

        assert len(self.viewer.viewer_state.layers) == 1
        assert self.viewer.viewer_state.layers[0].layer is self.data

    def test_removing_layer_state_removes_layer_artist(self):

        # Removing a layer artist removes the corresponding layer state. We need
        # to do this with a subset otherwise the viewer is closed

        self.init_subset()
        self.viewer.add_data(self.data)

        assert len(self.viewer.layers) == 2
        assert len(self.viewer.viewer_state.layers) == 2

        # self.layers is a copy so we need to remove from the original list
        self.viewer.viewer_state.layers.pop(1)

        assert len(self.viewer.layers) == 1
        assert self.viewer.layers[0].layer is self.data

        assert len(self.viewer.viewer_state.layers) == 1
        assert self.viewer.viewer_state.layers[0].layer is self.data

    def test_new_subset_after_remove_data(self):

        # Once we remove a dataset, if we make a new subset, it will not be
        # added to the viewer

        self.init_subset()
        self.viewer.add_data(self.data)

        assert len(self.viewer.layers) == 2
        assert len(self.viewer.viewer_state.layers) == 2

        self.viewer.viewer_state.layers.pop(0)

        self.init_subset()  # makes a new subset

        assert len(self.data.subsets) == 2

        assert len(self.viewer.layers) == 1
        assert self.viewer.layers[0].layer is self.data.subsets[0]

        assert len(self.viewer.viewer_state.layers) == 1
        assert self.viewer.viewer_state.layers[0].layer is self.data.subsets[0]

    def test_remove_not_present_ignored(self):
        data = Data(label='not in viewer')
        self.viewer.remove_data(data)

    def test_limits_sync(self):

        viewer_state = self.viewer.viewer_state
        axes = self.viewer.axes

        # Make sure that the viewer state and matplotlib viewer limits and log
        # settings are in sync. We start by modifying the state and making sure
        # that the axes follow.

        viewer_state.x_min = 3
        viewer_state.x_max = 9

        viewer_state.y_min = -2
        viewer_state.y_max = 3

        assert axes.get_xlim() == (3, 9)
        assert axes.get_ylim() == (-2, 3)
        assert axes.get_xscale() == 'linear'
        assert axes.get_yscale() == 'linear'

        viewer_state.log_x = True

        assert axes.get_xlim() == (3, 9)
        assert axes.get_ylim() == (-2, 3)
        assert axes.get_xscale() == 'log'
        assert axes.get_yscale() == 'linear'

        viewer_state.log_y = True

        # FIXME: the limits for y don't seem right, should be adjusted because of log?
        assert axes.get_xlim() == (3, 9)
        assert axes.get_ylim() == (-2, 3)
        assert axes.get_xscale() == 'log'
        assert axes.get_yscale() == 'log'

        # Check that changing the axes changes the state

        # NOTE: at the moment this doesn't work because Matplotlib doesn't
        # emit events for changing xscale/yscale. This isn't crucial anyway for
        # glue, but leaving the tests below in case this is fixed some day. The
        # Matplotlib issue is https://github.com/matplotlib/matplotlib/issues/8439

        axes.set_xscale('linear')
        #
        # assert viewer_state.x_min == 3
        # assert viewer_state.x_max == 9
        # assert viewer_state.y_min == -2
        # assert viewer_state.y_max == 3
        # assert not viewer_state.log_x
        # assert viewer_state.log_y
        #
        axes.set_yscale('linear')
        #
        # assert viewer_state.x_min == 3
        # assert viewer_state.x_max == 9
        # assert viewer_state.y_min == -2
        # assert viewer_state.y_max == 3
        # assert not viewer_state.log_x
        # assert not viewer_state.log_y

        axes.set_xlim(-1, 4)

        assert viewer_state.x_min == -1
        assert viewer_state.x_max == 4
        assert viewer_state.y_min == -2
        assert viewer_state.y_max == 3
        # assert not viewer_state.log_x
        # assert not viewer_state.log_y

        axes.set_ylim(5, 6)

        assert viewer_state.x_min == -1
        assert viewer_state.x_max == 4
        assert viewer_state.y_min == 5
        assert viewer_state.y_max == 6
        # assert not viewer_state.log_x
        # assert not viewer_state.log_y

    def test_add_invalid_data(self):
        data2 = Data()
        with pytest.raises(IncompatibleDataException):
            self.viewer.add_data(data2)

    # Communication tests

    def test_ignore_data_add_message(self):
        self.data_collection.append(self.data)
        assert len(self.viewer.layers) == 0

    def test_update_data_ignored_if_data_not_present(self):
        self.init_draw_count()
        self.data_collection.append(self.data)
        ct0 = self.draw_count
        self.data.style.color = 'blue'
        assert self.draw_count == ct0

    def test_update_data_processed_if_data_present(self):
        self.init_draw_count()
        self.data_collection.append(self.data)
        self.viewer.add_data(self.data)
        ct0 = self.draw_count
        self.data.style.color = 'blue'
        assert self.draw_count > ct0

    def test_add_subset_ignored_if_data_not_present(self):
        self.data_collection.append(self.data)
        sub = self.data.new_subset()
        assert sub not in self.viewer._layer_artist_container

    def test_add_subset_processed_if_data_present(self):
        self.data_collection.append(self.data)
        self.viewer.add_data(self.data)
        sub = self.data.new_subset()
        assert sub in self.viewer._layer_artist_container

    @pytest.mark.xfail
    def test_update_subset_ignored_if_not_present(self):
        self.init_draw_count()
        self.data_collection.append(self.data)
        self.viewer.add_data(self.data)
        sub = self.data.new_subset()
        self.viewer.remove_subset(sub)
        ct0 = self.draw_count
        sub.style.color = 'blue'
        assert self.draw_count == ct0

    def test_update_subset_processed_if_present(self):
        self.init_draw_count()
        self.data_collection.append(self.data)
        self.viewer.add_data(self.data)
        sub = self.data.new_subset()
        ct0 = self.draw_count
        sub.style.color = 'blue'
        assert self.draw_count > ct0

    def test_data_remove_message(self):
        self.data_collection.append(self.data)
        self.viewer.add_data(self.data)
        self.data_collection.remove(self.data)
        assert self.data not in self.viewer._layer_artist_container

    def test_subset_remove_message(self):
        self.data_collection.append(self.data)
        self.viewer.add_data(self.data)
        sub = self.data.new_subset()
        assert sub in self.viewer._layer_artist_container
        sub.delete()
        assert sub not in self.viewer._layer_artist_container
