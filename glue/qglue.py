"""
Utility function to load a variety of python objects into glue
"""

# Note: this is imported with Glue. We want
# to minimize imports so that utilities like glue-deps
# can run on systems with missing dependencies

from __future__ import absolute_import, division, print_function

import sys
from contextlib import contextmanager

import numpy as np

from glue.external import six
from glue.config import qglue_parser

try:
    from glue.core import Data
except ImportError:
    # let qglue import, even though this won't work
    # qglue will throw an ImportError
    Data = None


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


@qglue_parser(dict)
def _parse_data_dict(data, label):
    result = Data(label=label)
    for label, component in data.items():
        result.add_component(component, label)
    return [result]


@qglue_parser(np.recarray)
def _parse_data_recarray(data, label):
    kwargs = dict((n, data[n]) for n in data.dtype.names)
    return [Data(label=label, **kwargs)]


@qglue_parser(Data)
def _parse_data_glue_data(data, label):
    data.label = label
    return [data]


@qglue_parser(np.ndarray)
def _parse_data_numpy(data, label):
    return [Data(**{label: data, 'label': label})]


@qglue_parser(list)
def _parse_data_list(data, label):
    return [Data(**{label: data, 'label': label})]


@qglue_parser(six.string_types)
def _parse_data_path(path, label):
    from glue.core.data_factories import load_data, as_list

    data = load_data(path)
    for d in as_list(data):
        d.label = label
    return as_list(data)


def parse_data(data, label):
    for item in qglue_parser:
        data_class = item.data_class
        parser = item.parser
        if isinstance(data, data_class):
            try:
                return parser(data, label)
            except Exception as e:
                raise ValueError("Invalid format for data '%s'\n\n%s" %
                                 (label, e))

    raise TypeError("Invalid data description: %s" % data)


def parse_links(dc, links):
    from glue.core.link_helpers import multi_link
    from glue.core import ComponentLink

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
            result += multi_link(f, t, u, u2)

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

    :returns: A :class:`~glue.app.qt.application.GlueApplication` object
    """
    from glue.core import DataCollection
    from glue.app.qt import GlueApplication

    links = kwargs.pop('links', None)

    dc = DataCollection()
    for label, data in kwargs.items():
        dc.extend(parse_data(data, label))

    if links is not None:
        dc.add_link(parse_links(dc, links))

    with restore_io():
        ga = GlueApplication(dc)
        ga.start()

    return ga
