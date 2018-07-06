from __future__ import absolute_import, division, print_function

import os
import traceback
from functools import wraps

from glue.core.data import Data, Subset
from glue.external.six import string_types
from glue.core.session import Session
from glue.core.hub import HubListener
from glue.core import BaseData
from glue.core.data_factories import load_data
from glue.core.data_collection import DataCollection
from glue.config import settings


__all__ = ['Application']


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
    if isinstance(data, BaseData):
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

        self._hub = self._session.hub
        self._cmds = self._session.command_stack
        self._cmds.add_callback(self._update_undo_redo_enabled)

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

        if data is not None:
            if isinstance(data, Data):
                result = c.add_data(data)
            elif isinstance(data, Subset):
                result = c.add_subset(data)
            if not result:
                c.close(warn=False)
                return

        self.add_widget(c)
        c.show()
        return c

    @catch_error("Failed to save session")
    def save_session(self, path, include_data=False, absolute_paths=True):
        """ Save the data collection and hub to file.

        Can be restored via restore_session

        Note: Saving of client is not currently supported. Thus,
        restoring this session will lose all current viz windows
        """

        from glue.core.state import GlueSerializer
        gs = GlueSerializer(self,
                            include_data=include_data,
                            absolute_paths=absolute_paths)

        # In case relative paths are needed in the session file, we do the
        # serialization while setting the current directory to the directory
        # in which the session file will be saved so that the relative paths
        # are relative to the session file, not the current working directory.
        start_dir = os.path.abspath('.')
        session_dir = os.path.dirname(path) or '.'

        try:
            os.chdir(session_dir)
            state = gs.dumps(indent=2)
        finally:
            os.chdir(start_dir)

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

        # In case relative paths are needed in the session file, we do the
        # loading while setting the current directory to the directory
        # in which the session file is so that relative paths are interpreted
        # as relative to the session file.
        start_dir = os.path.abspath('.')
        session_dir = os.path.dirname(path) or '.'
        session_file = os.path.basename(path)

        try:
            os.chdir(session_dir)
            with open(session_file) as infile:
                state = GlueUnSerializer.load(infile)
            return state.object('__main__')
        finally:
            os.chdir(start_dir)

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
    def load_data(self, paths, skip_merge=False, auto_merge=False):
        """
        Given a path to a file, load the file as a Data object and add it to
        the current session.

        This returns the added `Data` object.
        """

        if isinstance(paths, string_types):
            paths = [paths]

        datasets = []
        for path in paths:
            result = load_data(path)
            if isinstance(result, BaseData):
                datasets.append(result)
            else:
                datasets.extend(result)

        self.add_datasets(self.data_collection, datasets,
                          skip_merge=skip_merge, auto_merge=auto_merge)

        if len(datasets) == 1:
            return datasets[0]
        else:
            return datasets

    @catch_error("Could not add data")
    def add_data(self, *args, **kwargs):
        """
        Add data to the session.

        Positional arguments are interpreted using the data factories, while
        keyword arguments are interpreted using the same infrastructure as the
        `qglue` command.

        This returns a list of added `Data` objects.
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

        return datasets

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
        raise Exception(message + "(" + detail + ")")

    def do(self, command):
        return self._cmds.do(command)

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
    def add_datasets(cls, data_collection, datasets, skip_merge=False, auto_merge=False):
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
            if not other or skip_merge:
                continue

            suggested_label = data_collection.suggest_merge_label(data, *other)

            if auto_merge:
                merges, label = [data] + other, suggested_label
            else:
                merges, label = cls._choose_merge(data, other, suggested_label)

            if merges:
                data_collection.merge(*merges, label=label)

            suggested.append(data)
            suggested.extend(other)

    @staticmethod
    def _choose_merge(data, other, suggested_label):
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
