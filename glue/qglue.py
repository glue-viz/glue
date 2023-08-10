"""
Utility function to load a variety of python objects into glue
"""

# Note: this is imported with Glue. We want
# to minimize imports so that utilities like glue-deps
# can run on systems with missing dependencies

import sys
from contextlib import contextmanager

try:
    from glue.core import BaseData, Data
    from glue.core.parsers import parse_data, parse_links
except ImportError:
    # let qglue import, even though this won't work
    # qglue will throw an ImportError
    BaseData = Data = None


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
    from glue.dialogs.autolinker.qt import run_autolinker

    links = kwargs.pop('links', None)

    dc = DataCollection()
    for label, data in kwargs.items():
        dc.extend(parse_data(data, label))

    if links is not None:
        dc.add_link(parse_links(dc, links))

    with restore_io():
        ga = GlueApplication(dc)
        run_autolinker(dc)
        ga.start()

    return ga
