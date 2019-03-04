"""
This module provides several classes and LinkCollection classes to
assist in linking data.

The :class:`LinkCollection` class and its sub-classes are factories to create
multiple ComponentLinks easily. They are meant to be passed to
:meth:`~glue.core.data_collection.DataCollection.add_link()`
"""

from __future__ import absolute_import, division, print_function

import types

from glue.config import link_function
from glue.external import six
from glue.core.data import ComponentID
from glue.core.component_link import ComponentLink

try:
    from inspect import getfullargspec
except ImportError:  # Python 2.7
    from inspect import getargspec as getfullargspec


__all__ = ['LinkCollection', 'LinkSame', 'LinkTwoWay', 'MultiLink',
           'LinkAligned']


@link_function("Link conceptually identical components",
               output_labels=['y'])
def identity(x):
    return x


@link_function("Convert between linear measurements and volume",
               output_labels=['volume'])
def lengths_to_volume(width, height, depth):
    return width * height * depth


class PartialResult(object):

    def __init__(self, func, index, name_prefix=""):
        self.func = func
        self.index = index
        self.__name__ = '%s%s_%i' % (name_prefix, func.__name__, index + 1)

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


class LinkCollection(object):

    def __init__(self, links=None):
        self._links = []

    def append(self, link):
        self._links.append(link)

    def extend(self, links):
        self._links.extend(links)

    def __iter__(self):
        for link in self._links:
            yield link

    def __len__(self):
        return len(self._links)

    def __getitem__(self, item):
        return self._links[item]

    def __contains__(self, cid):
        for link in self:
            if cid in link:
                return True
        return False

    def get_from_ids(self):
        from_ids = []
        for link in self:
            from_ids.extend(link.get_from_ids())
        return list(set(from_ids))

    def get_to_ids(self):
        return [link.get_to_id() for link in self]

    def __gluestate__(self, context):
        state = {}
        state['values'] = context.id(self._links)
        return state

    @classmethod
    def __setgluestate__(cls, rec, context):
        self = cls(context.object(rec['values']))
        return self


class LinkSame(LinkCollection):

    """
    Return ComponentLinks to represent that two componentIDs
    describe the same piece of information
    """

    def __init__(self, cid1, cid2):
        super(LinkSame, self).__init__()
        self._cid1 = cid1
        self._cid2 = cid2
        self.append(ComponentLink([_toid(cid1)], _toid(cid2)))

    def __gluestate__(self, context):
        state = {}
        state['cid1'] = context.id(self._cid1)
        state['cid2'] = context.id(self._cid2)
        return state

    @classmethod
    def __setgluestate__(cls, rec, context):
        self = cls(context.object(rec['cid1']), context.object(rec['cid2']))
        return self


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
        super(LinkTwoWay, self).__init__()
        self._cid1 = cid1
        self._cid2 = cid2
        self.forwards = forwards
        self.backwards = backwards
        self.append(ComponentLink([_toid(cid1)], _toid(cid2), forwards))
        self.append(ComponentLink([_toid(cid2)], _toid(cid1), backwards))

    def __gluestate__(self, context):
        state = {}
        state['cid1'] = context.id(self._cid1)
        state['cid2'] = context.id(self._cid2)
        state['forwards'] = context.id(self.forwards)
        state['backwards'] = context.id(self.backwards)
        return state

    @classmethod
    def __setgluestate__(cls, rec, context):
        self = cls(context.object(rec['cid1']),
                   context.object(rec['cid2']),
                   context.object(rec['forwards']),
                   context.object(rec['backwards']))
        return self


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

    def __init__(self, cids_left, cids_right, forwards=None, backwards=None,
                 labels_left=None, labels_right=None, description=None):

        super(MultiLink, self).__init__()

        self.cids_left = cids_left
        self.cids_right = cids_right
        self.forwards = forwards
        self.backwards = backwards
        self.description = description or ''

        if forwards is None and backwards is None:
            raise TypeError("Must supply either forwards or backwards")

        if forwards is not None:
            for i, r in enumerate(cids_right):
                func = PartialResult(forwards, i, name_prefix=self.__class__.__name__ + ".")
                self.append(ComponentLink(cids_left, r, func))

        if backwards is not None:
            for i, l in enumerate(cids_left):
                func = PartialResult(backwards, i, name_prefix=self.__class__.__name__ + ".")
                self.append(ComponentLink(cids_right, l, func))

        if forwards is None:
            self.labels_left = []
        else:
            if labels_left is None:
                if isinstance(forwards, types.MethodType):
                    self.labels_left = getfullargspec(forwards)[0][1:]
                else:
                    self.labels_left = getfullargspec(forwards)[0]
            else:
                self.labels_left = labels_left

        if backwards is None:
            self.labels_right = []
        else:
            if labels_right is None:
                if isinstance(backwards, types.MethodType):
                    self.labels_right = getfullargspec(backwards)[0][1:]
                else:
                    self.labels_right = getfullargspec(backwards)[0]
            else:
                self.labels_right = labels_right

    def __gluestate__(self, context):
        state = {}
        state['cids_left'] = context.id(self.cids_left)
        state['cids_right'] = context.id(self.cids_right)
        state['forwards'] = context.id(self.forwards)
        state['backwards'] = context.id(self.backwards)
        return state

    @classmethod
    def __setgluestate__(cls, rec, context):
        self = cls(context.object(rec['cids_left']),
                   context.object(rec['cids_right']),
                   context.object(rec['forwards']),
                   context.object(rec['backwards']))
        return self


class LinkAligned(LinkCollection):

    """Compute all the links to specify that the input data are pixel-aligned.

    :param data: An iterable of :class:`~glue.core.data.Data` instances
                 that are aligned at the pixel level. They must be the
                 same shape.
    """

    def __init__(self, data):
        super(LinkAligned, self).__init__()
        shape = data[0].shape
        ndim = data[0].ndim
        for i, d in enumerate(data[1:]):
            if d.shape != shape:
                raise TypeError("Input data do not have the same shape")
            for j in range(ndim):
                self.extend(LinkSame(data[0].pixel_component_ids[j],
                                     data[i + 1].pixel_component_ids[j]))


class FunctionalMultiLink(MultiLink):

    def __init__(self, cids, function, description=None, labels_left=None):

        self.cids = cids
        self.forwards = function
        self.backwards = None
        self.description = description or ''
        self.labels_left = labels_left or getfullargspec(function)[0]
        self.labels_right = []

        self.extend(self.function(*self.cids))
