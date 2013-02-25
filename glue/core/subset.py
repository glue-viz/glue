import operator
import numpy as np

from .visual import VisualAttributes, RED
from .decorators import memoize_attr_check
from .message import SubsetDeleteMessage, SubsetUpdateMessage
from .registry import Registry
from .util import split_component_view, view_shape

__all__ = ['Subset', 'SubsetState', 'RoiSubsetState', 'CompositeSubsetState',
           'OrState', 'AndState', 'XorState', 'InvertState',
           'ElementSubsetState', 'RangeSubsetState']

OPSYM = {operator.ge: '>=', operator.gt: '>',
         operator.le: '<=', operator.lt: '<',
         operator.and_: '&', operator.or_: '|',
         operator.xor: '^'}


class Subset(object):
    """Base class to handle subsets of data.

    These objects both describe subsets of a dataset, and relay any
    state changes to the hub that their parent data are assigned to.

    This base class only directly impements the logic that relays
    state changes back to the hub. Subclasses implement the actual
    description and manipulation of data subsets

    :param data:
        The dataset that this subset describes
    :type data: :class:`~glue.core.data.Data`

    :param style: VisualAttributes instance
        Describes visual attributes of the subset
    """

    def __init__(self, data, color=RED, alpha=0.5, label=None):
        """ Create a new subclass object.

        Note: the preferred way for creating subsets is through
        the data objects new_subset method. If instantiating
        new subsets manually, you will need to explicitly need
        to call the data objects add_subset method to inform
        the data set of the new subset
        """
        self._broadcasting = False  # must be first def
        self.data = data
        self.color = color
        self._label = None
        self.label = label  # trigger disambiguation
        self._style = VisualAttributes(parent=self)
        self._style.markersize *= 2.5
        self._style.color = color
        self._style.alpha = alpha
        self._subset_state = None
        self.subset_state = SubsetState()  # calls proper setter method

    @property
    def subset_state(self):
        return self._subset_state

    @subset_state.setter
    def subset_state(self, state):
        state.parent = self
        self._subset_state = state

    @property
    def style(self):
        return self._style

    @style.setter
    def style(self, value):
        value.parent = self
        self._style = value

    @property
    def label(self):
        """ Convenience access to subset's label """
        return self._label

    @label.setter
    def label(self, value):
        """Set the subset's label

        Subset labels within a data object must be unique. The input
        will be auto-disambiguated if necessary
        """
        value = Registry().register(self, value, group=self.data)
        self._label = value

    def register(self):
        """ Register a subset to its data, and start broadcasting
        state changes

        """
        self.data.add_subset(self)
        self.do_broadcast(True)

    def to_index_list(self):
        """
        Convert the current subset to a list of indices. These index
        the elements in the (flattened) data object that belong to the subset.

        If x is the numpy array corresponding to some component.data,
        the two following statements are equivalent::

           x.flat[subset.to_index_list()]
           x[subset.to_mask()]

        Returns:

           A numpy array, giving the indices of elements in the data that
           belong to this subset.

        Raises:

           IncompatibleDataException: if an index list cannot be created
           for the requested data set.

        """
        return self.subset_state.to_index_list()

    def to_mask(self, view=None):
        """
        Convert the current subset to a mask.

        :param view: An optional view into the dataset (e.g. a slice)
        If present, the mask will pertain to the view and not the
        entire dataset.

        Returns:

           A boolean numpy array, the same shape as the data, that
           defines whether each element belongs to the subset.

        """
        return self.subset_state.to_mask(view)

    def do_broadcast(self, value):
        """
        Set whether state changes to the subset are relayed to a hub.

        It can be useful to turn off broadcasting, when modifying the
        subset in ways that don't impact any of the clients.

        Attributes:
        value: Whether the subset should broadcast state changes (True/False)

        """
        object.__setattr__(self, '_broadcasting', value)

    def broadcast(self, attribute=None):
        """
        Explicitly broadcast a SubsetUpdateMessage to the hub

        :param attribute:
                   The name of the attribute (if any) that should be
                   broadcast as updated.
        :type attribute: ``str``

        """
        if not hasattr(self, 'data') or not hasattr(self.data, 'hub'):
            return

        if self._broadcasting and self.data.hub:
            msg = SubsetUpdateMessage(self, attribute=attribute)
            self.data.hub.broadcast(msg)

    def delete(self):
        """Broadcast a SubsetDeleteMessage to the hub, and stop broadcasting

        Also removes subset reference from parent data's subsets list
        """

        dobroad = self._broadcasting and self.data is not None and \
            self.data.hub is not None

        self.do_broadcast(False)

        if self.data is not None and self in self.data.subsets:
            self.data._subsets.remove(self)

        if dobroad:
            msg = SubsetDeleteMessage(self)
            self.data.hub.broadcast(msg)

        Registry().unregister(self, group=self.data)

    def write_mask(self, file_name, format="fits"):
        """ Write a subset mask out to file

        :param file_name: name of file to write to
        format:
           Name of format to write to. Currently, only "fits" is
           supported

        """
        mask = np.short(self.to_mask())
        if format == 'fits':
            try:
                from ..external.astro import fits
                fits.writeto(file_name, mask, clobber=True)
            except ImportError:
                raise ImportError("Cannot write mask -- requires astropy")
        else:
            raise AttributeError("format not supported: %s" % format)

    def read_mask(self, file_name):
        try:
            from ..external.astro import fits
            mask = fits.open(file_name)[0].data
        except ImportError:
            raise ImportError("Cannot write mask -- requires astropy")
        except IOError:
            raise IOError("Could not read %s (not a fits file?)" % file_name)
        ind = np.where(mask.flat)[0]
        state = ElementSubsetState(indices=ind)
        self.subset_state = state

    def __del__(self):
        self.delete()

    def __setattr__(self, attribute, value):
        object.__setattr__(self, attribute, value)
        if not attribute.startswith('_'):
            self.broadcast(attribute)

    def __getitem__(self, view):
        """ Retrieve the elements from a data view within the subset

        :param view: View of the data. See data.__getitem__ for detils
        """
        c, v = split_component_view(view)
        ma = self.to_mask(v)
        return self.data[view][ma]

    def paste(self, other_subset):
        """paste subset state from other_subset onto self """
        state = other_subset.subset_state.copy()
        state.parent = self
        self.subset_state = state

    def __str__(self):
        dlabel = "(no data)"
        if self.data is not None:
            dlabel = "(data: %s)" % self.data.label
        slabel = "Subset: (no label)"
        if self.label:
            slabel = "Subset: %s" % self.label
        return "%s %s" % (slabel, dlabel)

    def __repr__(self):
        return self.__str__()

    def __or__(self, other):
        return _combine([self, other], operator.or_)

    def __and__(self, other):
        return _combine([self, other], operator.and_)

    def __invert__(self):
        return _combine([self], operator.invert)

    def __xor__(self, other):
        return _combine([self, other], operator.xor)


class SubsetState(object):
    def __init__(self):
        self._parent = None

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        self._parent = value

    @property
    def subset_state(self):  # convenience method, mimic interface of Subset
        return self

    def to_index_list(self):
        return np.where(self.to_mask().flat)[0]

    def to_mask(self, view=None):
        shp = view_shape(self.parent.data.shape, view)
        return np.zeros(shp, dtype=bool)

    def copy(self):
        return SubsetState()

    def __or__(self, other_state):
        return OrState(self, other_state)

    def __and__(self, other_state):
        return AndState(self, other_state)

    def __invert__(self):
        return InvertState(self)

    def __xor__(self, other_state):
        return XorState(self, other_state)


class RoiSubsetState(SubsetState):
    def __init__(self):
        super(RoiSubsetState, self).__init__()
        self.xatt = None
        self.yatt = None
        self.roi = None

    @memoize_attr_check('parent')
    def to_mask(self, view=None):
        x = self.parent.data[self.xatt, view]
        y = self.parent.data[self.yatt, view]
        result = self.roi.contains(x, y)
        assert x.shape == result.shape
        return result

    def copy(self):
        result = RoiSubsetState()
        result.xatt = self.xatt
        result.yatt = self.yatt
        result.roi = self.roi
        return result


class RangeSubsetState(SubsetState):
    def __init__(self, lo, hi, att=None):
        super(RangeSubsetState, self).__init__()
        self.lo = lo
        self.hi = hi
        self.att = att

    def to_mask(self, view=None):
        x = self.parent.data[self.att, view]
        result = (x >= self.lo) & (x <= self.hi)
        return result

    def copy(self):
        return RangeSubsetState(self.lo, self.hi, self.att)


class CompositeSubsetState(SubsetState):
    op = None

    def __init__(self, state1, state2=None):
        super(CompositeSubsetState, self).__init__()
        self.state1 = state1.copy()
        if state2:
            state2 = state2.copy()
        self.state2 = state2
        self.parent = None

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent):
        if self.state1 is not None:
            self.state1.parent = parent
        if self.state2 is not None:
            self.state2.parent = parent
        self._parent = parent

    def copy(self):
        return type(self)(self.state1, self.state2)

    @memoize_attr_check('parent')
    def to_mask(self, view=None):
        assert self.state1.parent is self._parent
        assert self.state2.parent is self._parent
        return self.op(self.state1.to_mask(view), self.state2.to_mask(view))

    def __str__(self):
        sym = OPSYM.get(self.op, self.op)
        return "(%s %s %s)" % (self.state1, sym, self.state2)


class OrState(CompositeSubsetState):
    op = operator.or_


class AndState(CompositeSubsetState):
    op = operator.and_


class XorState(CompositeSubsetState):
    op = operator.xor


class InvertState(CompositeSubsetState):
    @memoize_attr_check('parent')
    def to_mask(self, view=None):
        return ~self.state1.to_mask(view)

    def __str__(self):
        return "(~%s)" % self.state1


class ElementSubsetState(SubsetState):
    def __init__(self, indices=None):
        super(ElementSubsetState, self).__init__()
        self._indices = indices

    @memoize_attr_check('parent')
    def to_mask(self, view=None):
        #XXX this is inefficient for views
        result = np.zeros(self.parent.data.shape, dtype=bool)
        if self._indices is not None:
            result.flat[self._indices] = True
        if view is not None:
            result = result[view]
        return result

    def copy(self):
        return ElementSubsetState(self._indices)


class InequalitySubsetState(SubsetState):
    def __init__(self, left, right, op):
        from .component_link import ComponentLink

        super(InequalitySubsetState, self).__init__()
        from .data import ComponentID
        valid_ops = [operator.gt, operator.ge,
                     operator.lt, operator.le]
        if op not in valid_ops:
            raise TypeError("Invalid boolean operator: %s" % op)
        if not isinstance(left, ComponentID) and not \
                operator.isNumberType(left) and not \
                isinstance(left, ComponentLink):
            raise TypeError("Input must be ComponenID or NumberType: %s"
                            % type(left))
        if not isinstance(right, ComponentID) and not \
                operator.isNumberType(right) and not \
                isinstance(right, ComponentLink):
            raise TypeError("Input must be ComponenID or NumberType: %s"
                            % type(right))
        self._left = left
        self._right = right
        self._operator = op

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right

    @property
    def operator(self):
        return self._operator

    @memoize_attr_check('parent')
    def to_mask(self, view=None):
        from .data import ComponentID
        left = self._left
        if not operator.isNumberType(self._left):
            left = self.parent.data[self._left, view]

        right = self._right
        if not operator.isNumberType(self._right):
            right = self.parent.data[self._right, view]

        return self._operator(left, right)

    def copy(self):
        return InequalitySubsetState(self._left, self._right, self._operator)

    def __str__(self):
        sym = OPSYM.get(self._operator, self._operator)
        return "(%s %s %s)" % (self._left, sym, self._right)

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, self)


def _combine(subsets, operator):
    state = operator(*[s.subset_state for s in subsets])
    result = Subset(None)
    result.subset_state = state
    return result
