from __future__ import absolute_import, division, print_function

import operator

import numpy as np

from glue.external import six
from glue.core.component_link import BinaryComponentLink
from glue.core.subset import InequalitySubsetState


__all__ = ['PixelComponentID', 'ComponentID', 'ComponentIDDict']

# access to ComponentIDs via .item[name]


class ComponentIDDict(object):

    def __init__(self, data, **kwargs):
        self.data = data

    def __getitem__(self, key):
        result = self.data.find_component_id(key)
        if result is None:
            raise KeyError("ComponentID not found or not unique: %s"
                           % key)
        return result


class ComponentID(object):
    """
    References a :class:`glue.core.component.Component` object within a :class:`~glue.core.data.Data` object.

    ComponentIDs behave as keys::

       component_id = data.id[name]
       data[component_id] -> numpy array
    """

    def __init__(self, label, hidden=False):
        """:param label: Name for the ID
           :type label: str"""
        self._label = str(label)
        self._hidden = hidden

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        """Change label.

        .. warning::
            Label changes are not currently tracked by client
            classes. Label's should only be changd before creating other
            client objects
        """
        self._label = str(value)

    @property
    def hidden(self):
        """Whether to hide the component by default"""
        return self._hidden

    def __str__(self):
        return str(self._label)

    def __repr__(self):
        return str(self._label)

    def __eq__(self, other):
        if np.issubsctype(type(other), np.number):
            return InequalitySubsetState(self, other, operator.eq)
        return other is self

    # In Python 3, if __eq__ is defined, then __hash__ has to be re-defined
    if six.PY3:
        __hash__ = object.__hash__

    def __ne__(self, other):
        if np.issubsctype(type(other), np.number):
            return InequalitySubsetState(self, other, operator.ne)
        return other is not self

    def __gt__(self, other):
        return InequalitySubsetState(self, other, operator.gt)

    def __ge__(self, other):
        return InequalitySubsetState(self, other, operator.ge)

    def __lt__(self, other):
        return InequalitySubsetState(self, other, operator.lt)

    def __le__(self, other):
        return InequalitySubsetState(self, other, operator.le)

    def __add__(self, other):
        return BinaryComponentLink(self, other, operator.add)

    def __radd__(self, other):
        return BinaryComponentLink(other, self, operator.add)

    def __sub__(self, other):
        return BinaryComponentLink(self, other, operator.sub)

    def __rsub__(self, other):
        return BinaryComponentLink(other, self, operator.sub)

    def __mul__(self, other):
        return BinaryComponentLink(self, other, operator.mul)

    def __rmul__(self, other):
        return BinaryComponentLink(other, self, operator.mul)

    def __div__(self, other):
        return BinaryComponentLink(self, other, operator.div)

    def __rdiv__(self, other):
        return BinaryComponentLink(other, self, operator.div)

    def __truediv__(self, other):
        return BinaryComponentLink(self, other, operator.truediv)

    def __rtruediv__(self, other):
        return BinaryComponentLink(other, self, operator.truediv)

    def __pow__(self, other):
        return BinaryComponentLink(self, other, operator.pow)

    def __rpow__(self, other):
        return BinaryComponentLink(other, self, operator.pow)


class PixelComponentID(ComponentID):
    """
    The ID of a component which is a pixel position in the data - this allows
    us to make assumptions in certain places. For example when a polygon
    selection is done in pixel space, it can easily be broadcast along
    dimensions.
    """

    def __init__(self, axis, label, hidden=False):
        self.axis = axis
        super(PixelComponentID, self).__init__(label, hidden=hidden)
