from __future__ import absolute_import, division, print_function

from functools import wraps, partial
import traceback

from .data_collection import DataCollection
from .data_factories import load_data
from . import command
from . import Data, Subset
from .hub import HubListener
from .util import PropertySetMixin, as_list
from .edit_subset_mode import EditSubsetMode
from .session import Session
from ..config import settings

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
        self._load_settings()

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
    def save_session(self, path):
        """ Save the data collection and hub to file.

        Can be restored via restore_session

        Note: Saving of client is not currently supported. Thus,
        restoring this session will lose all current viz windows
        """
        from .state import GlueSerializer
        gs = GlueSerializer(self)
        with open(path, 'w') as out:
            gs.dump(out, indent=2)

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

    def _load_settings(self, path=None):
        raise NotImplementedError()

    @catch_error("Could not load data")
    def load_data(self, path):
        d = load_data(path)
        self.add_datasets(self.data_collection, d)

    def report_error(self, message, detail):
        """ Report an error message to the user.
        Must be implemented in a subclass

        :param message: the message to display
        :type message: str

        :detail: Longer context about the error
        :type message: str
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
        """
        Utility method to interactively add datasets to a
        data_collection

        :param data_collection: :class:`~glue.core.data_collection.DataCollection`
        :param datasets: one or more :class:`~glue.core.data.Data` instances

        Adds datasets to the collection
        """

        datasets = as_list(datasets)
        data_collection.extend(datasets)
        list(map(partial(cls._suggest_mergers, data_collection), datasets))

    @classmethod
    def _suggest_mergers(cls, data_collection, data):
        """
        When loading a new dataset, check if any existing
        data has the same shape. If so, offer to
        merge the two datasets
        """
        shp = data.shape
        other = [d for d in data_collection
                 if d.shape == shp and d is not data]
        if not other:
            return

        merges = cls._choose_merge(data, other)

        if merges:
            data_collection.merge(*merges)

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
        raise NotImplementedError()

    def __gluestate__(self, context):
        viewers = [list(map(context.id, tab)) for tab in self.viewers]
        data = self.session.data_collection

        return dict(session=context.id(self.session), viewers=viewers,
                    data=context.id(data))

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

    # the glue.clients.layer_artist.LayerArtistContainer
    # class/subclass to use
    _container_cls = None

    def __init__(self, session):
        super(ViewerBase, self).__init__()

        self._session = session
        self._data = session.data_collection
        self._hub = None
        self._container = self._container_cls()

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

        :param data: Data object to add
        :type data: :class:`~glue.core.data.Data`
        """
        raise NotImplementedError

    def add_subset(self, subset):
        """ Add a subset to the viewer

        This must be overridden by a subclass

        :param subset: Subset instance to add
        :type subset: :class:`~glue.core.subset.Subset`
        """
        raise NotImplementedError

    def apply_roi(self, roi):
        """
        Apply an ROI to the client

        :param roi: The ROI to apply
        :type roi: :class:`~glue.core.roi.Roi`
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

        :param x: Offset of viewer's left edge from the left edge
                  of the parent window. Optional
        :type x: int

        :param y: Offset of the viewer's top edge from the top edge
                  of the parent window. Optional
        :type y: int
        """
        raise NotImplementedError()

    @property
    def position(self):
        """ Return the location of the viewer

        :rtype: (x, y). Tuple of 2 integers
        """
        raise NotImplementedError()

    @property
    def viewer_size(self):
        """ Return the size of the viewer

        :rtype: (width, height). Tuple of 2 ints
        """
        raise NotImplementedError()

    @viewer_size.setter
    def viewer_size(self, value):
        """ Resize the width and/or height of the viewer

        :param value: (width, height)

        :param width: new width. Optional.
        :type width: int

        :param height: new height. Optional.
        :type height: int
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
        return tuple(self._container)

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
