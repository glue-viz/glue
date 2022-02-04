""" Factory methods to build Data objects from files

Implementation notes:

Each factory method conforms to the folowing structure, which
helps the GUI Frontend easily load data:

1) The first argument is a file name to open

2) The return value is a Data object

3) The function should be decorated with data_factory and the decorator should
be given a label parameter that describes (in human language) what kinds of
files it understands, as well as a callable identifier parameter that returns
whether it can handle a requested filename and keyword set

Putting this together, the simplest data factory code looks like this::

    from glue.config import data_factory
    @data_factory(label="Foo file", identifier=has_extension('foo FOO'))
    def dummy_factory(file_name):
        return glue.core.Data()
"""

import os
import warnings

from glue.core.contracts import contract
from glue.core.coordinates import IdentityCoordinates
from glue.core.component import CoordinateComponent
from glue.core.data import Component, BaseData, Data
from glue.config import auto_refresh, data_factory
from glue.backends import get_timer
from glue.utils import as_list
from glue.logger import logger

__all__ = ['FileWatcher', 'LoadLog',
           'auto_data', 'data_label', 'find_factory',
           'has_extension', 'load_data',
           '_extension']


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

    Parameters
    ----------
    exts : str
      A space-delimited string listing the extensions
      (e.g., 'txt', or 'txt csv fits')

    Returns
    -------
    A function suitable as a factory identifier function
    """

    def tester(x, **kwargs):
        return _extension(x).lower() in set(exts.lower().split())
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

        self._absolute = True

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
                           if c in self.components and
                           type(c) == Component)
            dold.coords = dnew.coords
            dold.update_components(mapping)

    def __gluestate__(self, context):
        if context.absolute_paths:
            path = os.path.abspath(self.path)
        else:
            path = os.path.relpath(self.path)

        # If there are twice as many world coordinates as number of dimensions
        # and Data.coords is set to Coordinates or None, we need to set
        # force_coords to True to make sure that we always restore world
        # coordinate components even if the transform is an identity transform.
        n_coords = len([comp for comp in self.components
                        if isinstance(comp, CoordinateComponent)])
        if n_coords == self.components[0].ndim * 2:
            force_coords = True
        else:
            force_coords = False

        return dict(path=path,
                    factory=context.do(self.factory),
                    kwargs=[list(self.kwargs.items())],
                    force_coords=force_coords,
                    _protocol=2)

    @classmethod
    def __setgluestate__(cls, rec, context):
        fac = context.object(rec['factory'])
        kwargs = dict(*rec['kwargs'])
        kwargs['coord_first'] = rec.get('_protocol', 0) >= 1
        kwargs['force_coords'] = rec.get('_protocol', 0) < 2 or rec.get('force_coords')
        d = load_data(rec['path'], factory=fac, **kwargs)
        return as_list(d)[0]._load_log


class FileWatcher(object):

    """
    Watch a path for modifications, and perform an action on change

    Parameters
    ----------
    path : str
        The path to watch.

    callback : callable
        A function to call when the path changes.

    poll_interval : int
        Time to wait between checks, in ms.
    """

    def __init__(self, path, callback, poll_interval=1000):
        self.path = path
        self.callback = callback
        self.poll_interval = poll_interval
        self.watcher = get_timer()(poll_interval,
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
    """
    Use a factory to load a file and assign a label.

    This is the preferred interface for loading data into Glue,
    as it logs metadata about how data objects relate to files
    on disk.

    Parameters
    ----------
    path : str
        Path to the file.

    factory : callable
        Factory function to use. Defaults to :func:`glue.core.data_factories.auto_data`
        callback.

        Extra keywords are passed through to factory functions.
    """
    from glue.qglue import parse_data

    coord_first = kwargs.pop('coord_first', True)
    force_coords = kwargs.pop('force_coords', False)

    def as_data_objects(ds, lbl):
        # pack other container types like astropy tables
        # into glue data objects
        for d in ds:
            if isinstance(d, BaseData):
                yield d
                continue
            for item in parse_data(d, lbl):
                yield item

    if factory is None:
        factory = find_factory(path, **kwargs)
        if factory is None:
            raise KeyError("Don't know how to open file: %s" % path)
    lbl = data_label(path)

    d = as_list(factory(path, **kwargs))
    d = list(as_data_objects(d, lbl))

    log = LoadLog(path, factory, kwargs)
    for item in d:

        # NOTE: The LoadLog infrastructure is specifically designed for Data
        # objects in mind and not more general data classes.
        if not isinstance(item, Data):
            continue

        if item.coords is None and force_coords:
            item.coords = IdentityCoordinates(n_dim=item.ndim)

        if not item.label:
            item.label = lbl
        log.log(item)  # attaches log metadata to item

        if coord_first:
            # We just follow the order in which the components are now loaded,
            # which is coordinate components first, followed by all other
            # components
            for cid in item.coordinate_components + item.main_components:
                log.log(item.get_component(cid))
        else:
            # In this case the first component was the first one that is not a
            # coordinate component, followed by the coordinate components,
            # followed by the remaining components.
            cid = item.main_components[0]
            log.log(item.get_component(cid))
            for icid, cid in enumerate(item.coordinate_components):
                log.log(item.get_component(cid))
            for icid, cid in enumerate(item.main_components[1:]):
                log.log(item.get_component(cid))

    if len(d) == 1:
        # unpack single-length lists for user convenience
        return d[0]

    return d


def data_label(path):
    """
    Convert a file path into a data label, by stripping out
    slashes, file extensions, etc.
    """
    if os.path.basename(path) == '':
        path = os.path.dirname(path)
    _, fname = os.path.split(path)
    name, _ = os.path.splitext(fname)
    return name


@contract(extension='string', factory='callable')
def set_default_factory(extension, factory):  # pragma: no cover
    warnings.warn("set_default_factory is deprecated and no longer has any effect")


@contract(extension='string', returns='callable|None')
def get_default_factory(extension):  # pragma: no cover
    warnings.warn("get_default_factory is deprecated and will always return None")
    return None


@contract(filename='string')
def find_factory(filename, **kwargs):

    from glue.config import data_factory

    # We no longer try the 'default' factory first because we actually need to
    # try all identifiers and select the one to use based on the priority. This
    # allows us to define more specialized loaders take priority over more
    # general ones. For example, a FITS file that is a dendrogram should be
    # loaded as a dendrogram, not a plain FITS file.

    best_priority = None
    valid_formats = []

    # Iterating over the data factory returns the formats sorted by decreasing
    # alphabetical order then by label (alphabetically) in order to be
    # deterministic. This is implemented in DataFactoryRegistry.__iter__.

    for df in data_factory:

        logger.info('Trying data factory {0}'.format(df.label))

        # Once we've found a match, and iterated through the rest of the
        # importers with the same priority, we can exit the loop.
        if best_priority is not None and df.priority < best_priority:
            break

        if df.function is auto_data:
            continue

        try:
            is_format = df.identifier(filename, **kwargs)
        except ImportError:  # dependencies missing
            continue
        except Exception:  # any other issue
            continue

        if is_format:
            valid_formats.append(df)
            best_priority = df.priority

    logger.info('Valid formats: {0}'.format(valid_formats))

    if len(valid_formats) == 0:
        return None
    elif len(valid_formats) > 1:
        labels = ["'{0}'".format(x.label) for x in valid_formats]
        warnings.warn("Multiple data factories matched the input: {0}. Choosing {1}.".format(', '.join(labels), labels[0]))

    func = valid_formats[0].function

    return func


@data_factory(label='Auto', identifier=lambda x: True)
@contract(filename='string')
def auto_data(filename, *args, **kwargs):
    """Attempt to automatically construct a data object"""
    fac = find_factory(filename, **kwargs)
    if fac is None:
        raise KeyError("Don't know how to open file: %s" % filename)
    return fac(filename, *args, **kwargs)
