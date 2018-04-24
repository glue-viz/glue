from __future__ import absolute_import, division, print_function

from qtpy import QtWidgets
from qtpy.QtCore import Qt

from glue.core import message as msg
from glue.core import Data, Subset
from glue.core.qt.dialogs import warn
from glue.core.exceptions import IncompatibleDataException
from glue.core.state import lookup_class_with_patches
from glue.external import six
from glue.external.echo import delay_callback
from glue.utils import DeferDrawMeta, defer_draw
from glue.utils.noconflict import classmaker
from glue.viewers.common.qt.data_viewer import DataViewer

__all__ = ['DataViewerWithState']


@six.add_metaclass(classmaker(left_metas=(DeferDrawMeta,)))
class DataViewerWithState(DataViewer):

    allow_duplicate_data = False
    allow_duplicate_subset = False
    large_data_size = None

    _options_cls = None

    def __init__(self, session, parent=None, wcs=None, state=None):

        super(DataViewerWithState, self).__init__(session, parent)

        # Set up the state which will contain everything needed to represent
        # the current state of the viewer
        self.state = state or self._state_cls()
        self.state.data_collection = session.data_collection

        # Set up the options widget, which will include options that control the
        # viewer state
        if self._options_cls is None:
            self.options = QtWidgets.QWidget()
        else:
            self.options = self._options_cls(viewer_state=self.state,
                                             session=session)

        # When layer artists are removed from the layer artist container, we need
        # to make sure we remove matching layer states in the viewer state
        # layers attribute.
        self._layer_artist_container.on_changed(self._sync_state_layers)

        # And vice-versa when layer states are removed from the viewer state, we
        # need to keep the layer_artist_container in sync
        self.state.add_callback('layers', self._sync_layer_artist_container)

        self.statusBar().setSizeGripEnabled(False)
        self.setFocusPolicy(Qt.StrongFocus)

    def redraw(self):
        pass

    def _sync_state_layers(self, *args):
        # Remove layer state objects that no longer have a matching layer
        for layer_state in self.state.layers:
            if layer_state.layer not in self._layer_artist_container:
                self.state.layers.remove(layer_state)

    def _sync_layer_artist_container(self, *args):
        # Remove layer artists that no longer have a matching layer state
        layer_states = set(layer_state.layer for layer_state in self.state.layers)
        for layer_artist in self._layer_artist_container:
            if layer_artist.layer not in layer_states:
                self._layer_artist_container.remove(layer_artist)

    def add_data(self, data):

        # Check if data already exists in viewer
        if not self.allow_duplicate_data and data in self._layer_artist_container:
            return True

        if self.large_data_size is not None and data.size >= self.large_data_size:
            proceed = warn('Add large data set?', 'Data set {0:s} has {1:d} points, and '
                           'may render slowly.'.format(data.label, data.size), default='Cancel',
                           setting='show_large_data_warning')
            if not proceed:
                return False

        if data not in self.session.data_collection:
            raise IncompatibleDataException("Data not in DataCollection")

        # Create layer artist and add to container
        layer = self.get_data_layer_artist(data)

        if layer is None:
            return False

        self._layer_artist_container.append(layer)
        layer.update()

        # Add existing subsets to viewer
        for subset in data.subsets:
            self.add_subset(subset)

        self.redraw()

        return True

    def remove_data(self, data):
        with delay_callback(self.state, 'layers'):
            for layer_state in self.state.layers[::-1]:
                if isinstance(layer_state.layer, Data):
                    if layer_state.layer is data:
                        self.state.layers.remove(layer_state)
                else:
                    if layer_state.layer.data is data:
                        self.state.layers.remove(layer_state)
        self.redraw()

    def get_data_layer_artist(self, layer=None, layer_state=None):
        return self.get_layer_artist(self._data_artist_cls, layer=layer, layer_state=layer_state)

    def get_subset_layer_artist(self, layer=None, layer_state=None):
        return self.get_layer_artist(self._subset_artist_cls, layer=layer, layer_state=layer_state)

    def get_layer_artist(self, cls, layer=None, layer_state=None):
        return cls(layer=layer, layer_state=layer_state)

    def add_subset(self, subset):

        # Check if subset already exists in viewer
        if not self.allow_duplicate_subset and subset in self._layer_artist_container:
            return True

        # Make sure we add the data first if it doesn't already exist in viewer.
        # This will then auto add the subsets so can just return.
        if subset.data not in self._layer_artist_container:
            self.add_data(subset.data)
            return

        # Create scatter layer artist and add to container
        layer = self.get_subset_layer_artist(subset)
        self._layer_artist_container.append(layer)
        layer.update()

        self.redraw()

        return True

    def remove_subset(self, subset):
        if subset in self._layer_artist_container:
            self._layer_artist_container.pop(subset)
            self.redraw()

    def _add_subset(self, message):
        self.add_subset(message.subset)

    def _update_data(self, message):
        if message.data in self._layer_artist_container:
            for layer_artist in self._layer_artist_container:
                if isinstance(layer_artist.layer, Subset):
                    if layer_artist.layer.data is message.data:
                        layer_artist.update()
                else:
                    if layer_artist.layer is message.data:
                        layer_artist.update()
            self.redraw()

    def _update_subset(self, message):
        if message.attribute == 'style':
            return
        if message.subset in self._layer_artist_container:
            for layer_artist in self._layer_artist_container[message.subset]:
                layer_artist.update()
            self.redraw()

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
        return ('BACKGROUND_COLOR' in msg.settings or
                'FOREGROUND_COLOR' in msg.settings)

    def register_to_hub(self, hub):

        super(DataViewerWithState, self).register_to_hub(hub)

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
                      handler=self._update_data,
                      filter=self._has_data_or_subset)

        hub.subscribe(self, msg.DataCollectionDeleteMessage,
                      handler=self._remove_data)

        hub.subscribe(self, msg.ComponentsChangedMessage,
                      handler=self._update_data,
                      filter=self._has_data_or_subset)

        hub.subscribe(self, msg.ExternallyDerivableComponentsChangedMessage,
                      handler=self._update_data,
                      filter=self._has_data_or_subset)

        hub.subscribe(self, msg.SettingsChangeMessage,
                      self._update_appearance_from_settings,
                      filter=self._is_appearance_settings)

    def _update_appearance_from_settings(self, message=None):
        pass

    def unregister(self, hub):
        super(DataViewerWithState, self).unregister(hub)
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

        # Restore layer artists. Ideally we would delay instead of ignoring the
        # callback here.
        with viewer._layer_artist_container.ignore_empty():
            for l in rec['layers']:
                cls = lookup_class_with_patches(l.pop('_type'))
                layer_state = context.object(l['state'])
                layer_artist = viewer.get_layer_artist(cls, layer_state=layer_state)
                layer_state.viewer_state = viewer.state
                viewer._layer_artist_container.append(layer_artist)

        return viewer
