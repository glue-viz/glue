from __future__ import absolute_import, division, print_function

import traceback
from functools import wraps

from glue.core.session import Session
from glue.core.edit_subset_mode import EditSubsetMode
from glue.core.hub import HubListener
from glue.core import Data, Subset
from glue.core import command
from glue.core.data_factories import load_data
from glue.core.data_collection import DataCollection
from glue.config import settings
from glue.utils import as_list, PropertySetMixin


__all__ = ['Application', 'ViewerBase']


def catch_error(msg):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                m = "%s\n%s" % (msg, str(e))
                detail = str(traceback.format_exc())
                self = args[0]
                self.report_error(m, detail)
        return wrapper
    return decorator


def as_flat_data_list(data):
    datasets = []
    if isinstance(data, Data):
        datasets.append(data)
    else:
        for d in data:
            datasets.extend(as_flat_data_list(d))
    return datasets


class Application(HubListener):

    def __init__(self, data_collection=None, session=None):
        if session is not None:
            self._session = session
            session.application = self
            self._data = session.data_collection
        else:
            self._data = data_collection or DataCollection()
            self._session = Session(data_collection=self._data,
                                    application=self)

        EditSubsetMode().data_collection = self._data
        self._hub = self._session.hub
        self._cmds = self._session.command_stack
        self._cmds.add_callback(lambda x: self._update_undo_redo_enabled())

        self._settings = {}
        for key, value, validator in settings:
            self._settings[key] = [value, validator]

    @property
    def session(self):
        return self._session

    @property
    def data_collection(self):
        return self.session.data_collection

    def new_data_viewer(self, viewer_class, data=None):
        """
        Create a new data viewer, add it to the UI,
        and populate with data
        """
        if viewer_class is None:
            return

        c = viewer_class(self._session)
        c.register_to_hub(self._session.hub)

        if data and not c.add_data(data):
            c.close(warn=False)
            return

        self.add_widget(c)
        c.show()
        return c

    @catch_error("Failed to save session")
    def save_session(self, path, include_data=False):
        """ Save the data collection and hub to file.

        Can be restored via restore_session

        Note: Saving of client is not currently supported. Thus,
        restoring this session will lose all current viz windows
        """
        from glue.core.state import GlueSerializer
        gs = GlueSerializer(self, include_data=include_data)
        state = gs.dumps(indent=2)
        with open(path, 'w') as out:
            out.write(state)

    @staticmethod
    def restore_session(path):
        """
        Reload a previously-saved session

        Parameters
        ----------
        path : str
            Path to the file to load

        Returns
        -------
        app : :class:`Application`
            The loaded application
        """
        from glue.core.state import GlueUnSerializer

        with open(path) as infile:
            state = GlueUnSerializer.load(infile)

        return state.object('__main__')

    def new_tab(self):
        raise NotImplementedError()

    def add_widget(self, widget, label=None, tab=None):
        raise NotImplementedError()

    def close_tab(self):
        raise NotImplementedError()

    def get_setting(self, key):
        """
        Fetch the value of an application setting
        """
        return self._settings[key][0]

    def set_setting(self, key, value):
        """
        Set the value of an application setting

        Raises a KeyError if the setting does not exist
        Raises a ValueError if the value is invalid
        """
        validator = self._settings[key][1]
        self._settings[key][0] = validator(value)

    @property
    def settings(self):
        """Iterate over settings"""
        for key, (value, _) in self._settings.items():
            yield key, value

    @catch_error("Could not load data")
    def load_data(self, path):
        d = load_data(path)
        self.add_datasets(self.data_collection, d)

    @catch_error("Could not add data")
    def add_data(self, *args, **kwargs):
        """
        Add data to the session.

        Positional arguments are interpreted using the data factories, while
        keyword arguments are interpreted using the same infrastructure as the
        `qglue` command.
        """

        datasets = []

        for path in args:
            datasets.append(load_data(path))

        links = kwargs.pop('links', None)

        from glue.qglue import parse_data, parse_links

        for label, data in kwargs.items():
            datasets.extend(parse_data(data, label))

        self.add_datasets(self.data_collection, datasets)

        if links is not None:
            self.data_collection.add_link(parse_links(self.data_collection, links))

    def report_error(self, message, detail):
        """ Report an error message to the user.
        Must be implemented in a subclass

        Parameters
        ----------
        message : str
            The message to display
        detail : str
            Longer context about the error
        """
        raise NotImplementedError()

    def do(self, command):
        self._cmds.do(command)

    def undo(self):
        try:
            self._cmds.undo()
        except RuntimeError:
            pass

    def redo(self):
        try:
            self._cmds.redo()
        except RuntimeError:
            pass

    def _update_undo_redo_enabled(self):
        raise NotImplementedError()

    @classmethod
    def add_datasets(cls, data_collection, datasets):
        """ Utility method to interactively add datasets to a
        data_collection

        Parameters
        ----------
        data_collection : :class:`~glue.core.data_collection.DataCollection`
        datasets : :class:`~glue.core.data.Data` or list of Data
            One or more :class:`~glue.core.data.Data` instances

        Adds datasets to the collection
        """

        datasets = as_flat_data_list(datasets)
        data_collection.extend(datasets)

        # We now check whether any of the datasets can be merged. We need to
        # make sure that datasets are only ever shown once, as we don't want
        # to repeat the menu multiple times.

        suggested = []

        for data in datasets:

            # If the data was already suggested, we skip over it
            if data in suggested:
                continue

            shp = data.shape
            other = [d for d in data_collection
                     if d.shape == shp and d is not data]

            # If no other datasets have the same shape, we go to the next one
            if not other:
                continue

            merges, label = cls._choose_merge(data, other)

            if merges:
                data_collection.merge(*merges, label=label)

            suggested.append(data)
            suggested.extend(other)

    @staticmethod
    def _choose_merge(data, other):
        """
        Present an interface to the user for approving or rejecting
        a proposed data merger. Returns a list of datasets from other
        that the user has approved to merge with data
        """
        raise NotImplementedError

    @property
    def viewers(self):
        """Return a tuple of tuples of viewers currently open
        The i'th tuple stores the viewers in the i'th close_tab
        """
        return []

    def set_data_color(self, color, alpha):
        """
        Reset all the data colors to that specified.
        """
        for data in self.data_collection:
            data.style.color = color
            data.style.alpha = alpha

    def __gluestate__(self, context):
        viewers = [list(map(context.id, tab)) for tab in self.viewers]
        data = self.session.data_collection
        from glue.main import _loaded_plugins
        return dict(session=context.id(self.session), viewers=viewers,
                    data=context.id(data), plugins=_loaded_plugins)

    @classmethod
    def __setgluestate__(cls, rec, context):
        self = cls(data_collection=context.object(rec['data']))
        # manually register the newly-created session, which
        # the viewers need
        context.register_object(rec['session'], self.session)
        for i, tab in enumerate(rec['viewers']):
            if self.tab(i) is None:
                self.new_tab()
            for v in tab:
                viewer = context.object(v)
                self.add_widget(viewer, tab=i, hold_position=True)
        return self


class ViewerBase(HubListener, PropertySetMixin):

    """ Base class for data viewers in an application """

    # the glue.core.layer_artist.LayerArtistContainer
    # class/subclass to use
    _layer_artist_container_cls = None

    def __init__(self, session):

        HubListener.__init__(self)
        PropertySetMixin.__init__(self)

        self._session = session
        self._data = session.data_collection
        self._hub = None
        self._layer_artist_container = self._layer_artist_container_cls()

    def register_to_hub(self, hub):
        self._hub = hub

    def unregister(self, hub):
        """ Abstract method to unsubscribe from messages """
        raise NotImplementedError

    def request_add_layer(self, layer):
        """ Issue a command to add a layer """
        cmd = command.AddLayer(layer=layer, viewer=self)
        self._session.command_stack.do(cmd)

    def add_layer(self, layer):
        if isinstance(layer, Data):
            self.add_data(layer)
        elif isinstance(layer, Subset):
            self.add_subset(layer)
        # else: SubsetGroup

    def add_data(self, data):
        """ Add a data instance to the viewer

        This must be overridden by a subclass

        Parameters
        ----------
        data : :class:`~glue.core.data.Data`
            Data object to add.
        """
        raise NotImplementedError

    def add_subset(self, subset):
        """ Add a subset to the viewer

        This must be overridden by a subclass

        Parameters
        ----------
        subset : :class:`~glue.core.subset.Subset`
            Subset instance to add.
        """
        raise NotImplementedError

    def apply_roi(self, roi):
        """ Apply an ROI to the client

        Parameters
        ----------
        roi : :class:`~glue.core.roi.Roi`
            The ROI to apply.
        """
        cmd = command.ApplyROI(client=self.client, roi=roi)
        self._session.command_stack.do(cmd)

    @property
    def session(self):
        return self._session

    @property
    def axes(self):
        return self.client.axes

    def layer_view(self):
        raise NotImplementedError()

    def options_widget(self):
        raise NotImplementedError()

    def move(self, x=None, y=None):
        """ Reposition a viewer within the application.

        x : int, optional
            Offset of viewer's left edge from the left edge of the parent
            window.
        y : int, optional
            Offset of the viewer's top edge from the top edge of the parent
            window.
        """
        raise NotImplementedError()

    @property
    def position(self):
        """
        Return the location of the viewer as a tuple of ``(x, y)``
        """
        raise NotImplementedError()

    @property
    def viewer_size(self):
        """
        Return the size of the viewer as a tuple of ``(width, height)``
        """
        raise NotImplementedError()

    @viewer_size.setter
    def viewer_size(self, value):
        """ Resize the width and/or height of the viewer

        Parameters
        ----------
        value : tuple of int
            The width and height of the viewer.
        width : int, optional
            New width.
        height : int, optional
            New height.
        """
        raise NotImplementedError()

    def restore_layers(self, rec, context):
        """
        Given a list of glue-serialized layers, restore them
        to the viewer
        """
        # if this viewer manages a client, rely on it to restore layers
        if hasattr(self, 'client'):
            return self.client.restore_layers(rec, context)
        raise NotImplementedError()

    @property
    def layers(self):
        """Return a tuple of layers in this viewer.

        A layer is a visual representation of a dataset or subset within
        the viewer"""
        return tuple(self._layer_artist_container)

    def __gluestate__(self, context):
        return dict(session=context.id(self._session),
                    size=self.viewer_size,
                    pos=self.position,
                    properties=dict((k, context.id(v))
                                    for k, v in self.properties.items()),
                    layers=list(map(context.do, self.layers))
                    )

    @classmethod
    def __setgluestate__(cls, rec, context):
        session = context.object(rec['session'])
        result = cls(session)
        result.register_to_hub(session.hub)
        result.viewer_size = rec['size']
        x, y = rec['pos']
        result.move(x=x, y=y)

        prop = dict((k, context.object(v)) for
                    k, v in rec['properties'].items())
        result.restore_layers(rec['layers'], context)
        result.properties = prop
        return result
