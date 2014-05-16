from contextlib import contextmanager
import sys

import numpy as np

from .core import Data, DataCollection, ComponentLink
from .core.link_helpers import MultiLink
from .core.data_factories import load_data, as_list

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
        result.add_component(data[c], c)
    return [result]


def _parse_data_dict(data, label):
    result = Data(label=label)

    for label, component in data.items():
        result.add_component(component, label)

    return [result]


def _parse_data_recarray(data, label):
    print data.dtype.names
    return [Data(label=label, **{n: data[n] for n in data.dtype.names})]


def _parse_data_astropy_table(data, label):
    return [Data(label=label, **{c: data[c] for c in data.columns})]


def _parse_data_glue_data(data, label):
    data.label = label
    return [data]


def _parse_data_path(path, label):
    data = load_data(path)
    for d in as_list(data):
        d.label = label
    return as_list(data)


_parsers = {}  # map base classes -> parser functions
_parsers[dict] = _parse_data_dict
_parsers[np.recarray] = _parse_data_recarray
_parsers[Data] = _parse_data_glue_data
_parsers[basestring] = _parse_data_path


def _parse_data(data, label):
    for typ, prsr in _parsers.items():
        if isinstance(data, typ):
            try:
                return prsr(data, label)
            except Exception as e:
                raise ValueError("Invalid format for data '%s'\n\n%s" %
                                 (label, e))

    raise TypeError("Invalid data description: %s" % data)

try:
    import pandas as pd
    _parsers[pd.DataFrame] = _parse_data_dataframe
except ImportError:
    pass

try:
    from astropy.table import Table
    _parsers[Table] = _parse_data_astropy_table
except ImportError:
    pass


def _parse_links(dc, links):
    data = {d.label: d for d in dc}
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
        if isinstance(f, basestring):
            f = [find_cid(f)]
        else:
            f = [find_cid(item) for item in f]

        if isinstance(t, basestring):
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
      * A numpy rec array
      * A pandas data frame
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

        links = [(['balls.kg'], ['cones.lbs'], lb2kg, kb2lb)]
        qglue(balls=balls, cones=cones, links=links)

    :returns: A :class:`~glue.core.data_collection.DataCollection` object
    """
    from glue.qt.glue_application import GlueApplication

    links = kwargs.pop('links', None)

    dc = DataCollection()
    for label, data in kwargs.items():
        dc.extend(_parse_data(data, label))

    if links is not None:
        dc.add_link(_parse_links(dc, links))

    with restore_io():
        ga = GlueApplication(dc)
        ga.start()
    return dc
