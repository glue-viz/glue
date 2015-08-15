""" Factory methods to build Data objects from files

Implementation notes:

Each factory method conforms to the folowing structure, which
helps the GUI Frontend easily load data:

1) The first argument is a file name to open

2) The return value is a Data object

3) The function has a .label attribute that describes (in human
language) what kinds of files it understands

4) The function has a callable .identifier attribute that returns
whether it can handle a requested filename and keyword set

5) The function is added to the __factories__ list

Putting this together, the simplest data factory code looks like this::

    def dummy_factory(file_name):
        return glue.core.Data()
    dummy_factory.label = "Foo file"
    dummy_factory.identifier = has_extension('foo FOO')
    __factories__.append(dummy_factory)
"""

from __future__ import absolute_import, division, print_function

import os
import warnings

from ..data import Component, Data
from ...utils import as_list
from ...backends import get_backend
from ...config import auto_refresh
from ..contracts import contract

__all__ = ['FileWatcher', 'LoadLog',
           'auto_data', 'data_label', 'find_factory',
           'has_extension', 'load_data',
           '_extension', '__factories__']


__factories__ = []


def _extension(path):
    # extract the extension type from a path
    #  test.fits -> fits
    #  test.gz -> fits.gz (special case)
    #  a.b.c.fits -> fits
    _, path = os.path.split(path)
    if '.' not in path:
        return ''
    stems = path.split('.')[1:]

    # special case: test.fits.gz -> fits.gz
    if len(stems) > 1 and any(x == stems[-1]
                              for x in ['gz', 'gzip', 'bz', 'bz2']):
        return '.'.join(stems[-2:])
    return stems[-1]


def has_extension(exts):
    """
    A simple default filetype identifier function

    It returns a function that tests whether its input
    filename contains a particular extension

    Inputs
    ------
    exts : str
      A space-delimited string listing the extensions
      (e.g., 'txt', or 'txt csv fits')

    Returns
    -------
    A function suitable as a factory identifier function
    """

    def tester(x, **kwargs):
        return _extension(x) in set(exts.split())
    return tester


class LoadLog(object):

    """
    This class attaches some metadata to data created
    from load_data, so that the data can be re-constructed
    when loading saved state.

    It also watches the path for changes, and auto-reloads the data

    This is an internal class only meant to be used with load_data
    """

    def __init__(self, path, factory, kwargs):
        self.path = os.path.abspath(path)
        self.factory = factory
        self.kwargs = kwargs
        self.components = []
        self.data = []

        if auto_refresh():
            self.watcher = FileWatcher(path, self.reload)
        else:
            self.watcher = None

    def _log_component(self, component):
        self.components.append(component)

    def _log_data(self, data):
        self.data.append(data)

    def log(self, obj):
        if isinstance(obj, Component):
            self._log_component(obj)
        elif isinstance(obj, Data):
            self._log_data(obj)
        obj._load_log = self

    def id(self, component):
        return self.components.index(component)

    def component(self, index):
        return self.components[index]

    def reload(self):
        """
        Re-read files, and update data
        """
        try:
            d = load_data(self.path, factory=self.factory, **self.kwargs)
        except (OSError, IOError) as exc:
            warnings.warn("Could not reload %s.\n%s" % (self.path, exc))
            if self.watcher is not None:
                self.watcher.stop()
            return

        log = as_list(d)[0]._load_log

        for dold, dnew in zip(self.data, as_list(d)):
            if dold.shape != dnew.shape:
                warnings.warn("Cannot refresh data -- data shape changed")
                return

            mapping = dict((c, log.component(self.id(c)).data)
                           for c in dold._components.values()
                           if c in self.components
                           and type(c) == Component)
            dold.coords = dnew.coords
            dold.update_components(mapping)

    def __gluestate__(self, context):
        return dict(path=self.path,
                    factory=context.do(self.factory),
                    kwargs=[list(self.kwargs.items())])

    @classmethod
    def __setgluestate__(cls, rec, context):
        fac = context.object(rec['factory'])
        kwargs = dict(*rec['kwargs'])
        d = load_data(rec['path'], factory=fac, **kwargs)
        return as_list(d)[0]._load_log


class FileWatcher(object):

    """
    Watch a path for modifications, and perform an action on change
    """

    def __init__(self, path, callback, poll_interval=1000):
        """
        :param path: The path to watch, str
        :param callback: A function to call when the path changes
        :param poll_interval: Time to wait between checks, in ms
        """
        self.path = path
        self.callback = callback
        self.poll_interval = poll_interval
        self.watcher = get_backend().Timer(poll_interval,
                                           self.check_for_changes)

        try:
            self.stat_cache = os.stat(path).st_mtime
            self.start()
        except OSError:
            # file probably gone, no use watching
            self.stat_cache = None

    def stop(self):
        self.watcher.stop()

    def start(self):
        self.watcher.start()

    def check_for_changes(self):
        try:
            stat = os.stat(self.path).st_mtime
        except OSError:
            warnings.warn("Cannot access %s" % self.path)
            return

        if stat != self.stat_cache:
            self.stat_cache = stat
            self.callback()


@contract(path='string', factory='callable|None',
          returns='inst($Data)|list(inst($Data))')
def load_data(path, factory=None, **kwargs):
    """Use a factory to load a file and assign a label.

    This is the preferred interface for loading data into Glue,
    as it logs metadata about how data objects relate to files
    on disk.

    :param path: Path to a file
    :param factory: factory function to use. Defaults to :func:`auto_data`

    Extra keywords are passed through to factory functions
    """
    from ...qglue import parse_data

    def as_data_objects(ds, lbl):
        # pack other container types like astropy tables
        # into glue data objects
        for d in ds:
            if isinstance(d, Data):
                yield d
                continue
            for item in parse_data(d, lbl):
                yield item

    factory = factory or auto_data
    lbl = data_label(path)

    d = as_list(factory(path, **kwargs))
    d = list(as_data_objects(d, lbl))
    log = LoadLog(path, factory, kwargs)
    for item in d:
        if item.label is '':
            item.label = lbl
        log.log(item)  # attaches log metadata to item
        for cid in item.primary_components:
            log.log(item.get_component(cid))

    if len(d) == 1:
        # unpack single-length lists for user convenience
        return d[0]

    return d


def data_label(path):
    """Convert a file path into a data label, by stripping out
    slashes, file extensions, etc."""
    _, fname = os.path.split(path)
    name, _ = os.path.splitext(fname)
    return name


@contract(extension='string', factory='callable')
def set_default_factory(extension, factory):
    warnings.warn("set_default_factory is deprecated and no longer has any effect")


@contract(extension='string', returns='callable|None')
def get_default_factory(extension):
    warnings.warn("get_default_factory is deprecated and will always return None")
    return None


@contract(filename='string')
def find_factory(filename, **kwargs):

    from ...config import data_factory

    # We no longer try the 'default' factory first because we actually need to
    # try all identifiers and select the one to use based on the priority. This
    # allows us to define more specialized loaders take priority over more
    # general ones. For example, a FITS file that is a dendrogram should be
    # loaded as a dendrogram, not a plain FITS file.

    valid_formats = []

    for func, label, identifier, priority in data_factory:

        if func is auto_data:
            continue

        is_format = identifier(filename, **kwargs)

        if is_format:

            valid_formats.append((priority, label, func))

    print(valid_formats)

    priorities = list(zip(*valid_formats))[0]

    highest_priority = max(priorities)

    if priorities.count(highest_priority) > 1:
        raise ValueError("Ambiguous file type")

    func = valid_formats[priorities.index(highest_priority)][2]

    return func

@contract(filename='string')
def auto_data(filename, *args, **kwargs):
    """Attempt to automatically construct a data object"""
    fac = find_factory(filename, **kwargs)
    if fac is None:
        raise KeyError("Don't know how to open file: %s" % filename)
    return fac(filename, *args, **kwargs)

auto_data.label = 'Auto'
auto_data.identifier = lambda x: True
__factories__.append(auto_data)
