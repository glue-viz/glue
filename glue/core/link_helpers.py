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
    """
    A collection of links between two datasets.

    Parameters
    ----------
    data1 : `~glue.core.data.Data`
        The first dataset being linked
    data2 : `~glue.core.data.Data`
        The second dataset being linked
    links : list
        The initial links to add to the collection.
    """

    display = ''
    labels1 = []
    labels2 = []
    description = ''

    def __init__(self, data1=None, data2=None,
                 cids1=None, cids2=None,
                 labels1=None, labels2=None,
                 description=None,
                 links=None):
        self._links = links or []

        self.data1 = data1 or cids1[0].parent
        self.data2 = data2 or cids2[0].parent

        self.cids1 = cids1
        self.cids2 = cids2

        if labels1 is not None:
            self.labels1 = labels1

        if labels2 is not None:
            self.labels2 = labels2

        if description is not None:
            self.description = description

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

    def __gluestate__(self, context):
        state = {}
        state['data1'] = context.id(self.data1)
        state['data2'] = context.id(self.data2)
        state['cids1'] = context.id(self.cids1)
        state['cids2'] = context.id(self.cids2)
        state['labels1'] = context.id(self.cids2)
        state['labels2'] = context.id(self.labels2)
        state['values'] = context.id(self._links)
        return state

    @classmethod
    def __setgluestate__(cls, rec, context):
        # TODO: update this
        self = cls(context.object(rec['values']))
        return self


class FixedMethodsMultiLink(LinkCollection):
    """
    A link collection that is generated on-the-fly based on forward and
    backward transformation functions and lists of input/output component IDs.

    Parameters
    ----------
    data1 : `~glue.core.data.Data`
        The first dataset being linked
    data2 : `~glue.core.data.Data`
        The second dataset being linked
    cids1 : list of `~glue.core.component_id.ComponentID`
        The list of component IDs in the first dataset that are used in the links
    cids2 : list of `~glue.core.component_id.ComponentID`
        The list of component IDs in the second dataset that are used in the links
    labels1 : list of str, optional
        The human-readable names for the inputs to the ``forwards`` function.
        This is used for example in the graphical link editor. If not specified,
        the names of the arguments to ``forwards`` are used.
    labels2 : list of str, optional
        The human-readable names for the inputs to the ``backwards`` function.
        This is used for example in the graphical link editor. If not specified,
        the names of the arguments to ``backwards`` are used.
    description : str, optional
        A human-readable description of the link.
    """

    def __init__(self, data1=None, data2=None,
                 cids1=None, cids2=None,
                 forwards=None, backwards=None,
                 labels1=None, labels2=None, description=None):

        if labels1 is None:
            if isinstance(self.forwards, types.MethodType):
                labels1 = getfullargspec(self.forwards)[0][1:]
            else:
                labels1 = getfullargspec(self.forwards)[0]

        if labels2 is None:
            if isinstance(self.backwards, types.MethodType):
                labels2 = getfullargspec(self.backwards)[0][1:]
            else:
                labels2 = getfullargspec(self.backwards)[0]

        super(FixedMethodsMultiLink, self).__init__(data1=data1, data2=data2,
                                                    cids1=cids1, cids2=cids2,
                                                    labels1=labels1, labels2=labels2,
                                                    description=description)

        if len(cids2) == 1:
            self.append(ComponentLink(cids1, cids2[0], forwards))
        else:
            for i, r in enumerate(cids2):
                func = PartialResult(self.forwards, i, name_prefix=self.__class__.__name__ + ".")
                self.append(ComponentLink(cids1, r, func))

        if len(cids1) == 1:
            self.append(ComponentLink(cids2, cids1[0], backwards))
        else:
            for i, l in enumerate(cids1):
                func = PartialResult(self.backwards, i, name_prefix=self.__class__.__name__ + ".")
                self.append(ComponentLink(cids2, l, func))

    def forwards(self):
        raise NotImplementedError()

    def backwards(self):
        raise NotImplementedError()

    def __gluestate__(self, context):
        state = {}
        state['data1'] = context.id(self.data1)
        state['data2'] = context.id(self.data2)
        state['cids1'] = context.id(self.cids1)
        state['cids2'] = context.id(self.cids2)
        return state

    @classmethod
    def __setgluestate__(cls, rec, context):
        self = cls(data1=context.object(rec.get('data1', None)),
                   data2=context.object(rec.get('data2', None)),
                   cids1=context.object(rec['cids1']),
                   cids2=context.object(rec['cids2']))
        return self


class MultiLink(FixedMethodsMultiLink):
    """
    forwards : function
        Function that maps ``cids1`` to  ``cids2``. This should have
        the signature ``cids2 = forwards(*cids1)``, and is assumed
        to return a tuple. If not specifed, no forward links are calculated.
    backwards : function
        The inverse function to ``forwards``. If not specifed, no forward links
        are calculated.
    """

    def __init__(self, forwards=None, backwards=None, labels1=None, labels2=None, **kwargs):

        if forwards is None and backwards is None:
            raise TypeError("Must supply either forwards or backwards")

        self._forwards = forwards
        self._backwards = backwards

        if labels1 is None:
            if isinstance(forwards, types.MethodType):
                labels1 = getfullargspec(forwards)[0][1:]
            else:
                labels1 = getfullargspec(forwards)[0]

        if labels2 is None:
            if isinstance(backwards, types.MethodType):
                labels2 = getfullargspec(backwards)[0][1:]
            else:
                labels2 = getfullargspec(backwards)[0]

        super(MultiLink, self).__init__(labels1=labels1, labels2=labels2, **kwargs)

    def forwards(self, *args):
        return self._forwards(*args)

    def backwards(self, *args):
        return self._backwards(*args)

    def __gluestate__(self, context):
        state = super(MultiLink, self).__gluestate__(context)
        state['forwards'] = context.id(self._forwards)
        state['backwards'] = context.id(self._backwards)
        return state

    @classmethod
    def __setgluestate__(cls, rec, context):
        self = super(MultiLink, cls).__setgluestate__(rec, context)
        self._forwards = context.object(rec['forwards'])
        self._backwards = context.object(rec['backwards'])
        return self


class LinkSame(MultiLink):
    """
    A bi-directional identity link between two components.
    """

    def __init__(self, cid1=None, cid2=None, **kwargs):

        if cid1 is None:
            cid1 = kwargs['cids1']
        else:
            kwargs['data1'] = cid1.parent
            kwargs['cids1'] = [cid1]
            kwargs['forwards'] = identity

        if cid2 is None:
            cid2 = kwargs['cids2']
        else:
            kwargs['data2'] = cid2.parent
            kwargs['cids2'] = [cid2]
            kwargs['backwards'] = identity

        self._cid1 = cid1
        self._cid2 = cid2

        super(LinkSame, self).__init__(**kwargs)

    def __gluestate__(self, context):
        state = {}
        state['cid1'] = context.id(self._cid1)
        state['cid2'] = context.id(self._cid2)
        return state

    @classmethod
    def __setgluestate__(cls, rec, context):
        self = cls(context.object(rec['cid1']), context.object(rec['cid2']))
        return self


class LinkTwoWay(MultiLink):

    def __init__(self, cid1=None, cid2=None, forwards=None, backwards=None, **kwargs):
        """ Return 2 links that connect input ComponentIDs in both directions

        :param cid1: First ComponentID to link
        :param cid2: Second ComponentID to link
        :param forwards: Function which maps cid1 to cid2 (e.g. cid2=f(cid1))
        :param backwards: Function which maps cid2 to cid1 (e.g. cid1=f(cid2))

        :returns: Two :class:`~glue.core.component_link.ComponentLink`
                  instances, specifying the link in each direction
        """

        if cid1 is None:
            cid1 = kwargs['cids1']
        else:
            kwargs['data1'] = cid1.parent
            kwargs['cids1'] = [cid1]

        if cid2 is None:
            cid2 = kwargs['cids2']
        else:
            kwargs['data2'] = cid2.parent
            kwargs['cids2'] = [cid2]

        super(LinkTwoWay, self).__init__(forwards=forwards, backwards=backwards, **kwargs)

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


class LinkAligned(LinkCollection):

    """Compute all the links to specify that the input data are pixel-aligned.
    """

    def __init__(self, data1=None, data2=None):
        super(LinkAligned, self).__init__(data1=data1, data2=data2)
        if data1.shape != data2.shape:
            raise TypeError("Input data do not have the same shape")
        for j in range(data1.ndim):
            self.extend(LinkSame(data1.pixel_component_ids[j],
                                 data2.pixel_component_ids[j]))


def functional_link_collection(function):

    class FunctionalLinkCollection(LinkCollection):

        def __init__(self, data1=None, data2=None,
                     cids1=None, cids2=None,
                     labels1=None, labels2=None, description=None):
            super(FunctionalLinkCollection, self).__init__(data1=data1, data2=data2,
                                                           cids1=cids1, cids2=cids2,
                                                           labels1=labels1, labels2=labels2,
                                                           description=description)
            self.extend(function(*cids1, *cids2))

    return FunctionalLinkCollection
