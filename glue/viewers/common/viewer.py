import os
import warnings

from IPython import get_ipython

from glue.core.hub import HubListener
from glue.core import BaseData, Subset
from glue.core import command
from glue.core.command import ApplySubsetState

from glue.core.state import save
from glue.core import message as msg
from glue.core.exceptions import IncompatibleDataException
from glue.core.state import lookup_class_with_patches
from echo import delay_callback, ignore_callback
from glue.core.layer_artist import LayerArtistContainer

from glue.viewers.common.state import ViewerState
from glue.viewers.common.layer_artist import LayerArtist

from glue.config import layer_artist_maker

__all__ = ['BaseViewer', 'Viewer']


def get_layer_artist_from_registry(data, viewer):
    """
    Check whether any plugins define an appropriate custom layer artist for
    the specified data and viewer.
    """
    for maker in layer_artist_maker.members:
        layer_artist = maker.function(viewer, data)
        if layer_artist is not None:
            return layer_artist


class BaseViewer(HubListener):
    """
    The base class for all viewers.
    """

    LABEL = None

    def __init__(self, session):
        self._session = session
        self._data = session.data_collection
        self._hub = None
        if self.LABEL is None:
            self.LABEL = str(self.__class__)

    def register_to_hub(self, hub):
        self._hub = hub

    @property
    def session(self):
        return self._session

    def request_add_layer(self, layer):
        """ Issue a command to add a layer """
        cmd = command.AddLayer(layer=layer, viewer=self)
        self._session.command_stack.do(cmd)

    def add_layer(self, layer):
        if isinstance(layer, BaseData):
            self.add_data(layer)
        elif isinstance(layer, Subset):
            self.add_subset(layer)

    def remove_layer(self, layer):
        pass

    def add_data(self, data):
        raise NotImplementedError()

    def add_subset(self, subset):
        raise NotImplementedError()

    def __str__(self):
        return self.LABEL

    def __gluestate__(self, context):
        return dict(session=context.id(self._session))

    @classmethod
    def __setgluestate__(cls, rec, context):
        session = context.object(rec['session'])
        return cls(session)

    def apply_subset_state(self, subset_state, override_mode=None):
        cmd = ApplySubsetState(data_collection=self._data,
                               subset_state=subset_state,
                               override_mode=override_mode)
        self._session.command_stack.do(cmd)


TEMPLATE_SCRIPT = """
# This script was produced by glue and can be used to further customize a
# particular plot.

### Package imports

{imports}

### Set up data

data_collection = load('{data}')

### Set up viewer

{header}

### Set up layers

{layers}

### Legend

{legend}

### Finalize viewer

{footer}
""".strip()


class Viewer(BaseViewer):
    """
    A viewer class that uses a state class to represent the overall viewer
    state, and uses layer artists and state classes to handle each dataset
    and subset in the data viewer.
    """

    # The LayerArtistContainer class/subclass to use
    _layer_artist_container_cls = LayerArtistContainer

    # The state class/subclass to use
    _state_cls = ViewerState

    _data_artist_cls = LayerArtist
    _subset_artist_cls = LayerArtist

    allow_duplicate_data = False
    allow_duplicate_subset = False

    large_data_size = None

    def __init__(self, session, state=None):

        super(Viewer, self).__init__(session)

        # Set up the state which will contain everything needed to represent
        # the current state of the viewer
        self.state = state or self._state_cls()
        self.state.data_collection = session.data_collection

        # Create the layer artist container, which is the object in which
        # we will add LayerArtist objects
        self._layer_artist_container = self._layer_artist_container_cls()

        # When layer artists are removed from the layer artist container, we
        # need to make sure we remove matching layer states in the viewer state
        # layers attribute.
        self._layer_artist_container.on_changed(self._sync_state_layers)

        # And vice-versa when layer states are removed from the viewer state, we
        # need to keep the layer_artist_container in sync
        self.state.add_callback('layers', self._sync_layer_artist_container, priority=10000)

        self.state.add_callback('layers', self.draw_legend)

    def draw_legend(self, *args):
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

    def warn(self, message, *args, **kwargs):
        warnings.warn(message)
        return True

    def add_data(self, data):

        # Check if data already exists in viewer
        if not self.allow_duplicate_data and data in self._layer_artist_container:
            return True

        if self.large_data_size is not None and data.size >= self.large_data_size:
            proceed = self.warn('Add large data set?', 'Data set {0:s} has {1:d} points, and '
                                'may render slowly.'.format(data.label, data.size),
                                default='Cancel', setting='show_large_data_warning')
            if not proceed:
                return False

        if data not in self.session.data_collection:
            raise IncompatibleDataException("Data not in DataCollection")

        # Create layer artist and add to container. First check whether any
        # plugins want to make a custom layer artist.
        layer = get_layer_artist_from_registry(data, self) or self.get_data_layer_artist(data)

        if layer is None:
            return False

        # When adding a layer artist to the layer artist container, zorder
        # gets set automatically - however since we call a forced update of the
        # layer after adding it to the container we can ignore any callbacks
        # related to zorder. We also then need to set layer.state.zorder manually.
        with ignore_callback(layer.state, 'zorder'):
            self._layer_artist_container.append(layer)
        layer.update()
        self.draw_legend()  # need to be called here because callbacks are ignored in previous step

        # Add existing subsets to viewer
        for subset in data.subsets:
            self.add_subset(subset)

        return True

    def remove_data(self, data):
        with delay_callback(self.state, 'layers'):
            for layer_state in self.state.layers[::-1]:
                if isinstance(layer_state.layer, BaseData):
                    if layer_state.layer is data:
                        self.state.layers.remove(layer_state)
                else:
                    if layer_state.layer.data is data:
                        self.state.layers.remove(layer_state)

    def get_data_layer_artist(self, layer=None, layer_state=None):
        return self.get_layer_artist(self._data_artist_cls, layer=layer, layer_state=layer_state)

    def get_subset_layer_artist(self, layer=None, layer_state=None):
        return self.get_layer_artist(self._subset_artist_cls, layer=layer, layer_state=layer_state)

    def get_layer_artist(self, cls, layer=None, layer_state=None):
        return cls(self.state, layer=layer, layer_state=layer_state)

    def add_subset(self, subset):

        # Check if subset already exists in viewer
        if not self.allow_duplicate_subset and subset in self._layer_artist_container:
            return True

        # Create layer artist and add to container. First check whether any
        # plugins want to make a custom layer artist.
        layer = get_layer_artist_from_registry(subset, self) or self.get_subset_layer_artist(subset)

        if layer is None:
            return False

        # When adding a layer artist to the layer artist container, zorder
        # gets set automatically - however since we call a forced update of the
        # layer after adding it to the container we can ignore any callbacks
        # related to zorder.
        with ignore_callback(layer.state, 'zorder'):
            self._layer_artist_container.append(layer)
        layer.update()
        self.draw_legend()  # need to be called here because callbacks are ignored in previous step

        return True

    def remove_subset(self, subset):
        if subset in self._layer_artist_container:
            self._layer_artist_container.pop(subset)

    def _add_subset(self, message):
        self.add_subset(message.subset)

    def _update_data_numerical(self, message):
        # For some viewers, we might want to do additional things when the
        # actual numerical values or shape of a dataset change, but by default
        # we just pass this on to _update_data
        self._update_data(message)

    def _update_data(self, message):
        if message.data in self._layer_artist_container:
            for layer_artist in self._layer_artist_container:
                if isinstance(layer_artist.layer, Subset):
                    if layer_artist.layer.data is message.data:
                        layer_artist.update()
                else:
                    if layer_artist.layer is message.data:
                        layer_artist.update()

    def _update_subset(self, message):
        if message.attribute == 'style':
            return
        if message.subset in self._layer_artist_container:
            for layer_artist in self._layer_artist_container[message.subset]:
                layer_artist.update()

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

        super(Viewer, self).register_to_hub(hub)

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
                      handler=self._update_data_numerical,
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

        hub.subscribe(self, msg.ComputationMessage,
                      self._update_computation,
                      filter=self._has_layer_artist)

        hub.subscribe(self, msg.LayerArtistDisabledMessage,
                      self.draw_legend,
                      filter=self._has_layer_artist)

    def _has_layer_artist(self, message):
        return message.layer_artist in self.layers

    def _update_computation(self, message=None):
        pass

    def _update_appearance_from_settings(self, message=None):
        pass

    def __gluestate__(self, context):
        return dict(state=self.state.__gluestate__(context),
                    session=context.id(self._session),
                    layers=list(map(context.do, self.layers)),
                    _protocol=1)

    @classmethod
    def __setgluestate__(cls, rec, context):

        session = context.object(rec['session'])
        viewer_state = cls._state_cls.__setgluestate__(rec['state'], context)

        viewer = cls(session, state=viewer_state)
        viewer.register_to_hub(session.hub)

        # Restore layer artists. Ideally we would delay instead of ignoring the
        # callback here.
        with viewer._layer_artist_container.ignore_callbacks():
            for l in rec['layers']:
                cls = lookup_class_with_patches(l.pop('_type'))
                layer_state = context.object(l['state'])
                layer_artist = viewer.get_layer_artist(cls, layer_state=layer_state)
                layer_state.viewer_state = viewer.state
                viewer._layer_artist_container.append(layer_artist)

        viewer.draw_legend()  # need to be called here because callbacks are ignored in previous step
        return viewer

    def cleanup(self):

        if self._hub is not None:
            self.unregister(self._hub)

        self._layer_artist_container.clear_callbacks()
        self._layer_artist_container.clear()

        # Remove any references to the viewer in the IPython namespace. We use
        # list() here to force an explicit copy since we are modifying the
        # dictionary in-place
        shell = get_ipython()
        if shell is not None:
            for key in list(shell.user_ns):
                if shell.user_ns[key] is self:
                    shell.user_ns.pop(key)

    def remove_layer(self, layer):
        self._layer_artist_container.pop(layer)

    @property
    def layers(self):
        """Return a tuple of layers in this viewer.

        A layer is a visual representation of a dataset or subset within
        the viewer"""
        return tuple(self._layer_artist_container)

    def _script_header(self):
        raise NotImplementedError()

    def _script_legend(self):
        return [], ""

    def _script_footer(self):
        raise NotImplementedError()

    def export_as_script(self, filename):

        data_filename = os.path.relpath(filename) + '.data'

        save(data_filename, self.session.data_collection)

        imports = ['from glue.core.state import load']

        imports_header, header = self._script_header()
        imports.extend(imports_header)

        layers = ""
        for ilayer, layer in enumerate(self.layers):
            if layer.layer.label:
                layers += '## Layer {0}: {1}\n\n'.format(ilayer + 1, layer.layer.label)
            else:
                layers += '## Layer {0}\n\n'.format(ilayer + 1)
            if layer.visible and layer.enabled:
                if isinstance(layer.layer, BaseData):
                    index = self.session.data_collection.index(layer.layer)
                    layers += "layer_data = data_collection[{0}]\n\n".format(index)
                else:
                    dindex = self.session.data_collection.index(layer.layer.data)
                    sindex = layer.layer.data.subsets.index(layer.layer)
                    layers += ("layer_data = data_collection[{0}].subsets[{1}]\n\n"
                               .format(dindex, sindex))
            imports_layer, layer_script = layer._python_exporter(layer)
            if layer_script is None:
                continue
            imports.extend(imports_layer)
            layers += layer_script.strip() + "\n"

        imports_legend, legend = self._script_legend()
        imports.extend(imports_legend)
        imports_footer, footer = self._script_footer()
        imports.extend(imports_footer)

        imports = os.linesep.join(sorted(set(imports),
                                         key=lambda s: s.strip('# ')))
        # The sorting key is added keep together similar but commented imports
        # Typical ex:
        #    matplotlib.use('Agg')
        #    # matplotlib.use('qt5Agg')

        script = TEMPLATE_SCRIPT.format(data=os.path.basename(data_filename),
                                        imports=imports.strip(),
                                        header=header.strip(),
                                        layers=layers.strip(),
                                        legend=legend.strip(),
                                        footer=footer.strip())

        with open(filename, 'w') as f:
            f.write(script)
