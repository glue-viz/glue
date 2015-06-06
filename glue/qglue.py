"""
Utility function to load a variety of python objects into glue
"""

# Note: this is imported with Glue. We want
# to minimize imports so that utilities like glue-deps
# can run on systems with missing dependencies

from __future__ import absolute_import, division, print_function

from contextlib import contextmanager
import sys

import numpy as np

try:
    from .core import Data
except ImportError:
    # let qglue import, even though this won't work
    # qglue will throw an ImportError
    Data = None

from .external import six

__all__ = ['qglue']


@contextmanager
def restore_io():
    stdin = sys.stdin
    stdout = sys.stdout
    stderr = sys.stderr
    _in = sys.__stdin__
    _out = sys.__stdout__
    _err = sys.__stderr__
    try:
        yield
    finally:
        sys.stdin = stdin
        sys.stdout = stdout
        sys.stderr = stderr
        sys.__stdin__ = _in
        sys.__stdout__ = _out
        sys.__stderr__ = _err


def _parse_data_dataframe(data, label):
    label = label or 'Data'
    result = Data(label=label)
    for c in data.columns:
        result.add_component(data[c], str(c))
    return [result]


def _parse_data_dict(data, label):
    result = Data(label=label)

    for label, component in data.items():
        result.add_component(component, label)

    return [result]


def _parse_data_recarray(data, label):
    kwargs = dict((n, data[n]) for n in data.dtype.names)
    return [Data(label=label, **kwargs)]


def _parse_data_astropy_table(data, label):
    kwargs = dict((c, data[c]) for c in data.columns)
    return [Data(label=label, **kwargs)]


def _parse_data_glue_data(data, label):
    data.label = label
    return [data]


def _parse_data_numpy(data, label):
    return [Data(**{label: data, 'label': label})]


def _parse_data_path(path, label):
    from .core.data_factories import load_data, as_list

    data = load_data(path)
    for d in as_list(data):
        d.label = label
    return as_list(data)


def _parse_data_hdulist(data, label):
    """
    Parse all HDUs in an HDUList into a data object, and build coords.
    Assumes all extensions have the same shape
    """
    from .core.io import filter_hdulist_by_shape
    from .core.coordinates import coordinates_from_header

    result = Data(label=label)

    hdulist = filter_hdulist_by_shape(data)
    header = hdulist[0].header

    result.coords = coordinates_from_header(header)
    for hdu in hdulist:
        result.add_component(hdu.data, label=hdu.name)

    return [result]

# (base class, parser function)
_parsers = [
    (Data, _parse_data_glue_data),
    (six.string_types, _parse_data_path),
    (dict, _parse_data_dict),
    (np.recarray, _parse_data_recarray),
    (np.ndarray, _parse_data_numpy),
    (list, _parse_data_numpy)]


def parse_data(data, label):
    for typ, prsr in _parsers:
        if isinstance(data, typ):
            try:
                return prsr(data, label)
            except Exception as e:
                raise ValueError("Invalid format for data '%s'\n\n%s" %
                                 (label, e))

    raise TypeError("Invalid data description: %s" % data)

try:
    import pandas as pd
    _parsers.append((pd.DataFrame, _parse_data_dataframe))
except ImportError:
    pass

try:
    from astropy.table import Table
    from astropy.io.fits import HDUList
    _parsers.append((Table, _parse_data_astropy_table))
    # Put HDUList parser before list parser
    _parsers = [(HDUList, _parse_data_hdulist)] + _parsers
except ImportError:
    pass


def _parse_links(dc, links):
    from .core.link_helpers import MultiLink
    from .core import ComponentLink

    data = dict((d.label, d) for d in dc)
    result = []

    def find_cid(s):
        dlabel, clabel = s.split('.')
        d = data[dlabel]
        c = d.find_component_id(clabel)
        if c is None:
            raise ValueError("Invalid link (no component named %s)" % s)
        return c

    for link in links:
        f, t = link[0:2]  # from and to component names
        u = u2 = None
        if len(link) >= 3:  # forward translation function
            u = link[2]
        if len(link) == 4:  # reverse translation function
            u2 = link[3]

        # component names -> component IDs
        if isinstance(f, six.string_types):
            f = [find_cid(f)]
        else:
            f = [find_cid(item) for item in f]

        if isinstance(t, six.string_types):
            t = find_cid(t)
            result.append(ComponentLink(f, t, u))
        else:
            t = [find_cid(item) for item in t]
            result += MultiLink(f, t, u, u2)

    return result


def qglue(**kwargs):
    """
    Quickly send python variables to Glue for visualization.

    The generic calling sequence is::

      qglue(label1=data1, label2=data2, ..., [links=links])

    The kewyords label1, label2, ... can be named anything besides ``links``

    data1, data2, ... can be in many formats:
      * A pandas data frame
      * A path to a file
      * A numpy array, or python list
      * A numpy rec array
      * A dictionary of numpy arrays with the same shape
      * An astropy Table

    ``Links`` is an optional list of link descriptions, each of which has
    the format: ([left_ids], [right_ids], forward, backward)

    Each ``left_id``/``right_id`` is a string naming a component in a dataset
    (i.e., ``data1.x``). ``forward`` and ``backward`` are functions which
    map quantities on the left to quantities on the right, and vice
    versa. `backward` is optional

    Examples::

        balls = {'kg': [1, 2, 3], 'radius_cm': [10, 15, 30]}
        cones = {'lbs': [5, 3, 3, 1]}
        def lb2kg(lb):
            return lb / 2.2
        def kg2lb(kg):
            return kg * 2.2

        links = [(['balls.kg'], ['cones.lbs'], lb2kg, kg2lb)]
        qglue(balls=balls, cones=cones, links=links)

    :returns: A :class:`~glue.qt.glue_application.GlueApplication` object
    """
    from .core import DataCollection
    from glue.qt.glue_application import GlueApplication

    links = kwargs.pop('links', None)

    dc = DataCollection()
    for label, data in kwargs.items():
        dc.extend(parse_data(data, label))

    if links is not None:
        dc.add_link(_parse_links(dc, links))

    with restore_io():
        ga = GlueApplication(dc)
        ga.start()

    return ga
