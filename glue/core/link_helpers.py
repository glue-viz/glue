"""
This module provides several classes and LinkCollection classes to
assist in linking data.

The functions in this class (and stored in the __LINK_FUNCTIONS__
list) define common coordinate transformations. They are meant to be
used for the `using` parameter in
:class:`glue.core.component_link.ComponentLink` instances.

The :class:`LinkCollection` class and its sublcasses are factories to create
multiple ComponentLinks easily. They are meant to be passed to
:meth:`~glue.core.data_collection.DataCollection.add_link()`
"""

from __future__ import absolute_import, division, print_function

from .component_link import ComponentLink
from .data import ComponentID
from ..external import six

import numpy as np

__all__ = ['LinkCollection', 'LinkSame', 'LinkTwoWay', 'MultiLink',
           'LinkAligned']

__LINK_FUNCTIONS__ = []
__LINK_HELPERS__ = []


def identity(x):
    return x
identity.output_args = ['y']


def lengths_to_volume(width, height, depth):
    """Compute volume from linear measurements of a box"""
    # included for demonstration purposes
    return width * height * depth


lengths_to_volume.output_args = ['area']

__LINK_FUNCTIONS__.append(identity)
__LINK_FUNCTIONS__.append(lengths_to_volume)


class PartialResult(object):

    def __init__(self, func, index):
        self.func = func
        self.index = index
        self.__name__ = '%s_%i' % (func.__name__, index + 1)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)[self.index]

    def __gluestate__(self, context):
        return dict(func=context.do(self.func), index=self.index)

    @classmethod
    def __setgluestate__(cls, rec, context):
        return cls(context.object(rec['func']), rec['index'])


def _toid(arg):
    """Coerce the input to a ComponentID, if possible"""
    if isinstance(arg, ComponentID):
        return arg
    elif isinstance(arg, six.string_types):
        return ComponentID(arg)
    else:
        raise TypeError('Cannot be cast to a ComponentID: %s' % arg)


class LinkCollection(list):
    pass


class LinkSame(LinkCollection):

    """
    Return ComponentLinks to represent that two componentIDs
    describe the same piece of information
    """

    def __init__(self, cid1, cid2):
        self.append(ComponentLink([_toid(cid1)], _toid(cid2)))


class LinkTwoWay(LinkCollection):

    def __init__(self, cid1, cid2, forwards, backwards):
        """ Return 2 links that connect input ComponentIDs in both directions

        :param cid1: First ComponentID to link
        :param cid2: Second ComponentID to link
        :param forwards: Function which maps cid1 to cid2 (e.g. cid2=f(cid1))
        :param backwards: Function which maps cid2 to cid1 (e.g. cid1=f(cid2))

        :returns: Two :class:`~glue.core.component_link.ComponentLink`
                  instances, specifying the link in each direction
        """
        self.append(ComponentLink([_toid(cid1)], _toid(cid2), forwards))
        self.append(ComponentLink([_toid(cid2)], _toid(cid1), backwards))


class MultiLink(LinkCollection):
    """
    Compute all the ComponentLinks to link groups of ComponentIDs

    :param cids_left: first collection of ComponentIDs
    :param cids_right: second collection of ComponentIDs
    :param forwards:
        Function that maps ``cids_left -> cids_right``. Assumed to have
        signature ``cids_right = forwards(*cids_left)``, and assumed
        to return a tuple. If not provided, the relevant ComponentIDs
        will not be generated
    :param backwards:
       The inverse function to forwards. If not provided, the relevant
       ComponentIDs will not be generated

    :returns: a collection of :class:`~glue.core.component_link.ComponentLink`
              objects.
    """

    def __init__(self, cids_left, cids_right, forwards=None, backwards=None):
        cids_left = list(map(_toid, cids_left))
        cids_right = list(map(_toid, cids_right))

        if forwards is None and backwards is None:
            raise TypeError("Must supply either forwards or backwards")

        if forwards is not None:
            for i, r in enumerate(cids_right):
                func = PartialResult(forwards, i)
                self.append(ComponentLink(cids_left, r, func))

        if backwards is not None:
            for i, l in enumerate(cids_left):
                func = PartialResult(backwards, i)
                self.append(ComponentLink(cids_right, l, func))


class LinkAligned(LinkCollection):

    """Compute all the links to specify that the input data are pixel-aligned.

    :param data: An iterable of :class:`~glue.core.data.Data` instances
                 that are aligned at the pixel level. They must be the
                 same shape.
    """

    def __init__(self, data):
        shape = data[0].shape
        ndim = data[0].ndim
        for i, d in enumerate(data[1:]):
            if d.shape != shape:
                raise TypeError("Input data do not have the same shape")
            for j in range(ndim):
                self.extend(LinkSame(data[0].get_pixel_component_id(j),
                                     data[i + 1].get_pixel_component_id(j)))


# DEPRECATED: for backward-compatibility we import the following celestial
# conversions. This is needed because glue saved sessions will refer to the
# functions at this location. We can only remove this if we are ok with
# breaking compatibility with glue session files at some point.
from ..plugins.coordinate_helpers.deprecated import (Galactic2Equatorial,
                                                     radec2glon, radec2glat,
                                                     lb2ra, lb2dec)
