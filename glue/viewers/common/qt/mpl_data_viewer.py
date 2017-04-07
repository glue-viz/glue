from __future__ import absolute_import, division, print_function

from glue.viewers.common.qt.data_viewer import DataViewer
from glue.viewers.common.qt.mpl_widget import MplWidget
from glue.viewers.common.viz_client import init_mpl
from glue.external.echo import add_callback
from glue.utils import nonpartial
from glue.utils.decorators import avoid_circular
from glue.viewers.common.qt.mpl_toolbar import MatplotlibViewerToolbar
from glue.viewers.common.mpl_state import MatplotlibDataViewerState
from glue.core import message as msg
from glue.core import Data
from glue.core.exceptions import IncompatibleDataException

__all__ = ['MatplotlibDataViewer']


class MatplotlibDataViewer(DataViewer):

    _toolbar_cls = MatplotlibViewerToolbar
    _state_cls = MatplotlibDataViewerState

    def __init__(self, session, parent=None):

        super(MatplotlibDataViewer, self).__init__(session, parent)

        # Use MplWidget to set up a Matplotlib canvas inside the Qt window
        self.mpl_widget = MplWidget()
        self.setCentralWidget(self.mpl_widget)

        # TODO: shouldn't have to do this
        self.central_widget = self.mpl_widget

        self.figure, self._axes = init_mpl(self.mpl_widget.canvas.fig)

        # Set up the state which will contain everything needed to represent
        # the current state of the viewer
        self.viewer_state = self._state_cls()
        self.viewer_state.data_collection = session.data_collection

        # Set up the options widget, which will include options that control the
        # viewer state
        self.options = self._options_cls(viewer_state=self.viewer_state,
                                         session=session)

        add_callback(self.viewer_state, 'x_min', nonpartial(self.limits_to_mpl))
        add_callback(self.viewer_state, 'x_max', nonpartial(self.limits_to_mpl))
        add_callback(self.viewer_state, 'y_min', nonpartial(self.limits_to_mpl))
        add_callback(self.viewer_state, 'y_max', nonpartial(self.limits_to_mpl))

        self.axes.callbacks.connect('xlim_changed', nonpartial(self.limits_from_mpl))
        self.axes.callbacks.connect('ylim_changed', nonpartial(self.limits_from_mpl))

        self.viewer_state.add_callback('log_x', nonpartial(self.update_log_x))
        self.viewer_state.add_callback('log_y', nonpartial(self.update_log_y))

        self.axes.set_autoscale_on(False)

        # TODO: in future could move the following to a more basic data viewer class

        # When layer artists are removed from the layer artist container, we need
        # to make sure we remove matching layer states in the viewer state
        # layers attribute.
        self._layer_artist_container.on_changed(nonpartial(self._sync_state_layers))

        # And vice-versa when layer states are removed from the viewer state, we
        # need to keep the layer_artist_container in sync
        self.viewer_state.add_callback('layers', nonpartial(self._sync_layer_artist_container))

    def _sync_state_layers(self):
        # Remove layer state objects that no longer have a matching layer
        for layer_state in self.viewer_state.layers:
            if layer_state.layer not in self._layer_artist_container:
                self.viewer_state.layers.remove(layer_state)

    def _sync_layer_artist_container(self):
        # Remove layer artists that no longer have a matching layer state
        layer_states = set(layer_state.layer for layer_state in self.viewer_state.layers)
        for layer_artist in self._layer_artist_container:
            if layer_artist.layer not in layer_states:
                self._layer_artist_container.remove(layer_artist)

    def update_log_x(self):
        self.axes.set_xscale('log' if self.viewer_state.log_x else 'linear')

    def update_log_y(self):
        self.axes.set_yscale('log' if self.viewer_state.log_y else 'linear')

    @avoid_circular
    def limits_from_mpl(self):
        # TODO: delay callbacks here
        self.viewer_state.x_min, self.viewer_state.x_max = self.axes.get_xlim()
        self.viewer_state.y_min, self.viewer_state.y_max = self.axes.get_ylim()

    @avoid_circular
    def limits_to_mpl(self):
        self.axes.set_xlim(self.viewer_state.x_min, self.viewer_state.x_max)
        self.axes.set_ylim(self.viewer_state.y_min, self.viewer_state.y_max)
        self.axes.figure.canvas.draw()

    # TODO: shouldn't need this!
    @property
    def axes(self):
        return self._axes

    def add_data(self, data):

        if data in self._layer_artist_container:
            return True

        if data not in self.session.data_collection:
            raise IncompatibleDataException("Data not in DataCollection")

        # Create layer artist and add to container
        layer = self._data_artist_cls(data, self._axes, self.viewer_state)
        self._layer_artist_container.append(layer)
        layer.update()

        # Add existing subsets to viewer
        for subset in data.subsets:
            self.add_subset(subset)

        return True

    def remove_data(self, data):

        for layer_artist in self.viewer_state.layers[::-1]:
            if isinstance(layer_artist.layer, Data):
                if layer_artist.layer is data:
                    self.viewer_state.layers.remove(layer_artist)
            else:
                if layer_artist.layer.data is data:
                    self.viewer_state.layers.remove(layer_artist)

    def remove_subset(self, subset):
        if subset in self._layer_artist_container:
            self._layer_artist_container.pop(subset)
            self.axes.figure.canvas.draw()

    def add_subset(self, subset):

        # Make sure we add the data first if it doesn't already exist in viewer.
        # This will then auto add the subsets so can just return.
        if subset.data not in self._layer_artist_container:
            self.add_data(subset.data)
            return

        # Copy settings from data if present
        if subset.data in self._layer_artist_container:
            initial_layer_state = self._layer_artist_container[subset.data][0].layer_state
        else:
            initial_layer_state = None

        # Create scatter layer artist and add to container
        layer = self._subset_artist_cls(subset, self._axes, self.viewer_state,
                                        initial_layer_state=initial_layer_state)
        self._layer_artist_container.append(layer)
        layer.update()

        return True

    def _add_subset(self, message):
        self.add_subset(message.subset)

    def _update_subset(self, message):
        if message.subset in self._layer_artist_container:
            for layer_artist in self._layer_artist_container[message.subset]:
                layer_artist.update()
            self.axes.figure.canvas.draw()

    def _remove_subset(self, message):
        self.remove_subset(message.subset)

    def options_widget(self):
        return self.options

    def register_to_hub(self, hub):

        super(MatplotlibDataViewer, self).register_to_hub(hub)

        def subset_has_data(x):
            return x.sender.data in self._layer_artist_container.layers

        def has_data_or_subset(x):
            return x.sender in self._layer_artist_container.layers

        hub.subscribe(self, msg.SubsetCreateMessage,
                      handler=self._add_subset,
                      filter=subset_has_data)

        hub.subscribe(self, msg.SubsetUpdateMessage,
                      handler=self._update_subset,
                      filter=has_data_or_subset)

        hub.subscribe(self, msg.SubsetDeleteMessage,
                      handler=self._remove_subset,
                      filter=has_data_or_subset)

        hub.subscribe(self, msg.NumericalDataChangedMessage,
                      handler=self._update_subset,
                      filter=has_data_or_subset)

        hub.subscribe(self, msg.DataCollectionDeleteMessage,
                      handler=lambda x: self.remove_data(x.data))

        # hub.subscribe(self, msg.ComponentsChangedMessage,
        #               handler=self._update_data,
        #               filter=has_data)

    def unregister(self, hub):
        super(MatplotlibDataViewer, self).unregister(hub)
        hub.unsubscribe_all(self)
