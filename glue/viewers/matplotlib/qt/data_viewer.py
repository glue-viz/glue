from __future__ import absolute_import, division, print_function

from qtpy.QtCore import Qt

from glue.viewers.common.qt.data_viewer import DataViewer
from glue.viewers.matplotlib.qt.widget import MplWidget
from glue.viewers.common.viz_client import init_mpl, update_appearance_from_settings
from glue.external.echo import add_callback, delay_callback
from glue.utils import nonpartial, defer_draw
from glue.utils.decorators import avoid_circular
from glue.viewers.matplotlib.qt.toolbar import MatplotlibViewerToolbar
from glue.viewers.matplotlib.state import MatplotlibDataViewerState
from glue.core import message as msg
from glue.core import Data
from glue.core.exceptions import IncompatibleDataException
from glue.core.state import lookup_class_with_patches

__all__ = ['MatplotlibDataViewer']


class MatplotlibDataViewer(DataViewer):

    _toolbar_cls = MatplotlibViewerToolbar
    _state_cls = MatplotlibDataViewerState

    allow_duplicate_data = False

    def __init__(self, session, parent=None, wcs=None, state=None):

        super(MatplotlibDataViewer, self).__init__(session, parent)

        # Use MplWidget to set up a Matplotlib canvas inside the Qt window
        self.mpl_widget = MplWidget()
        self.setCentralWidget(self.mpl_widget)

        # TODO: shouldn't have to do this
        self.central_widget = self.mpl_widget

        self.figure, self._axes = init_mpl(self.mpl_widget.canvas.fig, wcs=wcs)

        # Set up the state which will contain everything needed to represent
        # the current state of the viewer
        self.state = state or self._state_cls()
        self.state.data_collection = session.data_collection

        # Set up the options widget, which will include options that control the
        # viewer state
        self.options = self._options_cls(viewer_state=self.state,
                                         session=session)

        add_callback(self.state, 'x_min', nonpartial(self.limits_to_mpl))
        add_callback(self.state, 'x_max', nonpartial(self.limits_to_mpl))
        add_callback(self.state, 'y_min', nonpartial(self.limits_to_mpl))
        add_callback(self.state, 'y_max', nonpartial(self.limits_to_mpl))

        self.axes.callbacks.connect('xlim_changed', nonpartial(self.limits_from_mpl))
        self.axes.callbacks.connect('ylim_changed', nonpartial(self.limits_from_mpl))

        self.state.add_callback('x_log', nonpartial(self.update_x_log))
        self.state.add_callback('y_log', nonpartial(self.update_y_log))

        self.axes.set_autoscale_on(False)

        # TODO: in future could move the following to a more basic data viewer class

        # When layer artists are removed from the layer artist container, we need
        # to make sure we remove matching layer states in the viewer state
        # layers attribute.
        self._layer_artist_container.on_changed(nonpartial(self._sync_state_layers))

        # And vice-versa when layer states are removed from the viewer state, we
        # need to keep the layer_artist_container in sync
        self.state.add_callback('layers', nonpartial(self._sync_layer_artist_container))

        self.central_widget.resize(600, 400)
        self.resize(self.central_widget.size())
        self.statusBar().setSizeGripEnabled(False)
        self.setFocusPolicy(Qt.StrongFocus)

    def _sync_state_layers(self):
        # Remove layer state objects that no longer have a matching layer
        for layer_state in self.state.layers:
            if layer_state.layer not in self._layer_artist_container:
                self.state.layers.remove(layer_state)

    def _sync_layer_artist_container(self):
        # Remove layer artists that no longer have a matching layer state
        layer_states = set(layer_state.layer for layer_state in self.state.layers)
        for layer_artist in self._layer_artist_container:
            if layer_artist.layer not in layer_states:
                self._layer_artist_container.remove(layer_artist)

    def update_x_log(self):
        self.axes.set_xscale('log' if self.state.x_log else 'linear')

    def update_y_log(self):
        self.axes.set_yscale('log' if self.state.y_log else 'linear')

    @avoid_circular
    def limits_from_mpl(self):
        with delay_callback(self.state, 'x_min', 'x_max', 'y_min', 'y_max'):
            self.state.x_min, self.state.x_max = self.axes.get_xlim()
            self.state.y_min, self.state.y_max = self.axes.get_ylim()

    @avoid_circular
    def limits_to_mpl(self):
        self.axes.set_xlim(self.state.x_min, self.state.x_max)
        self.axes.set_ylim(self.state.y_min, self.state.y_max)
        self.axes.figure.canvas.draw()

    # TODO: shouldn't need this!
    @property
    def axes(self):
        return self._axes

    @defer_draw
    def add_data(self, data):

        if not self.allow_duplicate_data and data in self._layer_artist_container:
            return True

        if data not in self.session.data_collection:
            raise IncompatibleDataException("Data not in DataCollection")

        # Create layer artist and add to container
        layer = self._data_artist_cls(self._axes, self.state, layer=data)

        if layer is None:
            return False

        self._layer_artist_container.append(layer)
        layer.update()

        # Add existing subsets to viewer
        for subset in data.subsets:
            self.add_subset(subset)

        self.axes.figure.canvas.draw()

        return True

    @defer_draw
    def remove_data(self, data):
        for layer_artist in self.state.layers[::-1]:
            if isinstance(layer_artist.layer, Data):
                if layer_artist.layer is data:
                    self.state.layers.remove(layer_artist)
            else:
                if layer_artist.layer.data is data:
                    self.state.layers.remove(layer_artist)
        self.axes.figure.canvas.draw()

    @defer_draw
    def add_subset(self, subset):

        # Make sure we add the data first if it doesn't already exist in viewer.
        # This will then auto add the subsets so can just return.
        if subset.data not in self._layer_artist_container:
            self.add_data(subset.data)
            return

        # Create scatter layer artist and add to container
        layer = self._subset_artist_cls(self._axes, self.state, layer=subset)
        self._layer_artist_container.append(layer)
        layer.update()

        self.axes.figure.canvas.draw()

        return True

    @defer_draw
    def remove_subset(self, subset):
        if subset in self._layer_artist_container:
            self._layer_artist_container.pop(subset)
            self.axes.figure.canvas.draw()

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

    def _subset_has_data(self, x):
        return x.sender.data in self._layer_artist_container.layers

    def _has_data_or_subset(self, x):
        return x.sender in self._layer_artist_container.layers

    def _remove_data(self, message):
        self.remove_data(message.data)

    def _is_appearance_settings(self, msg):
        return ('BACKGROUND_COLOR' in msg.settings
                or 'FOREGROUND_COLOR' in msg.settings)

    def register_to_hub(self, hub):

        super(MatplotlibDataViewer, self).register_to_hub(hub)

        hub.subscribe(self, msg.SubsetCreateMessage,
                      handler=self._add_subset,
                      filter=self._subset_has_data)

        hub.subscribe(self, msg.SubsetUpdateMessage,
                      handler=self._update_subset,
                      filter=self._has_data_or_subset)

        hub.subscribe(self, msg.SubsetDeleteMessage,
                      handler=self._remove_subset,
                      filter=self._has_data_or_subset)

        hub.subscribe(self, msg.NumericalDataChangedMessage,
                      handler=self._update_subset,
                      filter=self._has_data_or_subset)

        hub.subscribe(self, msg.DataCollectionDeleteMessage,
                      handler=self._remove_data)

        # hub.subscribe(self, msg.ComponentsChangedMessage,
        #               handler=self._update_data,
        #               filter=has_data)

        hub.subscribe(self, msg.SettingsChangeMessage,
                      self._update_appearance_from_settings,
                      filter=self._is_appearance_settings)

    def _update_appearance_from_settings(self, message=None):
        update_appearance_from_settings(self.axes)
        self.axes.figure.canvas.draw()

    def unregister(self, hub):
        super(MatplotlibDataViewer, self).unregister(hub)
        hub.unsubscribe_all(self)

    def __gluestate__(self, context):
        return dict(state=self.state.__gluestate__(context),
                    session=context.id(self._session),
                    size=self.viewer_size,
                    pos=self.position,
                    layers=list(map(context.do, self.layers)),
                    _protocol=1)

    def update_viewer_state(rec, context):
        pass

    @classmethod
    @defer_draw
    def __setgluestate__(cls, rec, context):

        if rec.get('_protocol', 0) < 1:
            cls.update_viewer_state(rec, context)

        session = context.object(rec['session'])

        viewer_state = cls._state_cls.__setgluestate__(rec['state'], context)

        viewer = cls(session, state=viewer_state)
        viewer.register_to_hub(session.hub)
        viewer.viewer_size = rec['size']
        x, y = rec['pos']
        viewer.move(x=x, y=y)

        # Restore layer artists
        for l in rec['layers']:
            cls = lookup_class_with_patches(l.pop('_type'))
            layer_state = context.object(l['state'])
            layer_artist = cls(viewer.axes, viewer.state, layer_state=layer_state)
            viewer._layer_artist_container.append(layer_artist)

        return viewer
