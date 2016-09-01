from __future__ import absolute_import, division, print_function

import numbers
import operator

import numpy as np

from glue.external.six import PY3
from glue.core.roi import CategoricalROI
from glue.core.contracts import contract
from glue.core.util import split_component_view
from glue.core.registry import Registry
from glue.core.exceptions import IncompatibleAttribute
from glue.core.message import SubsetDeleteMessage, SubsetUpdateMessage
from glue.core.decorators import memoize
from glue.core.visual import VisualAttributes
from glue.config import settings
from glue.utils import view_shape, broadcast_to


__all__ = ['Subset', 'SubsetState', 'RoiSubsetState', 'CategoricalROISubsetState',
           'RangeSubsetState', 'MultiRangeSubsetState', 'CompositeSubsetState',
           'OrState', 'AndState', 'XorState', 'InvertState', 'MaskSubsetState', 'CategorySubsetState',
           'ElementSubsetState', 'InequalitySubsetState', 'combine_multiple',
           'CategoricalMultiRangeSubsetState', 'CategoricalROISubsetState2D']


OPSYM = {operator.ge: '>=', operator.gt: '>',
         operator.le: '<=', operator.lt: '<',
         operator.and_: '&', operator.or_: '|',
         operator.xor: '^', operator.eq: '==',
         operator.ne: '!='}
SYMOP = dict((v, k) for k, v in OPSYM.items())


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
    """

    @contract(data='isinstance(Data)|None',
              color='color',
              alpha=float,
              label='string|None')
    def __init__(self, data, color=settings.SUBSET_COLORS[0], alpha=0.5, label=None):
        """ Create a new subset object.

        Note: the preferred way for creating subsets is
        via DataCollection.new_subset_group. Manually-instantiated
        subsets will probably *not* be represented properly by the UI
        """
        self._broadcasting = False  # must be first def
        self.data = data
        self._subset_state = None
        self._label = None
        self._style = None
        self._setup(color, alpha, label)

    @contract(color='color', alpha='float', label='string|None')
    def _setup(self, color, alpha, label):
        self.color = color
        self.label = label  # trigger disambiguation
        self.style = VisualAttributes(parent=self)
        self.style.markersize *= 1.5
        self.style.color = color
        self.style.alpha = alpha
        self.subset_state = SubsetState()  # calls proper setter method

    @property
    def subset_state(self):
        return self._subset_state

    @subset_state.setter
    def subset_state(self, state):
        if isinstance(state, np.ndarray):
            if self.data.shape != state.shape:
                raise ValueError("Shape of mask doesn't match shape of data")
            cids = self.data.pixel_component_ids
            state = MaskSubsetState(state, cids)
        if not isinstance(state, SubsetState):
            raise TypeError("State must be a SubsetState instance or array")
        self._subset_state = state

    @property
    def style(self):
        return self._style

    @style.setter
    @contract(value=VisualAttributes)
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

    @property
    def attributes(self):
        """
        Returns a tuple of the ComponentIDs that this subset
        depends upon
        """
        return self.subset_state.attributes

    def register(self):
        """ Register a subset to its data, and start broadcasting
        state changes

        """
        self.data.add_subset(self)
        self.do_broadcast(True)

    @contract(returns='array[N]')
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
        try:
            return self.subset_state.to_index_list(self.data)
        except IncompatibleAttribute as exc:
            try:
                return self._to_index_list_join()
            except IncompatibleAttribute:
                raise exc

    def _to_index_list_join(self):
        return np.where(self._to_mask_join(None).flat)[0]

    def _to_mask_join(self, view):
        """
        Convert the subset to a mask through an entity join to another
        dataset.
        """
        for other, (cid1, cid2) in self.data._key_joins.items():

            if getattr(other, '_recursing', False):
                continue

            try:
                self.data._recursing = True
                s2 = Subset(other)
                s2.subset_state = self.subset_state
                mask_right = s2.to_mask()
            except IncompatibleAttribute:
                continue
            finally:
                self.data._recursing = False

            if len(cid1) == 1 and len(cid2) == 1:

                key_left = self.data[cid1[0], view]
                key_right = other[cid2[0], mask_right]
                mask = np.in1d(key_left.ravel(), key_right.ravel())

                return mask.reshape(key_left.shape)

            elif len(cid1) == len(cid2):

                key_left_all = []
                key_right_all = []

                for cid1_i, cid2_i in zip(cid1, cid2):
                    key_left_all.append(self.data[cid1_i, view].ravel())
                    key_right_all.append(other[cid2_i, mask_right].ravel())

                # TODO: The following is slow because we are looping in Python.
                #       This could be made significantly faster by switching to
                #       C/Cython.

                key_left_all = zip(*key_left_all)
                key_right_all = set(zip(*key_right_all))

                result = [key in key_right_all for key in key_left_all]
                result = np.array(result)

                return result.reshape(self.data[cid1_i, view].shape)

            elif len(cid1) == 1:

                key_left = self.data[cid1[0], view].ravel()
                mask = np.zeros_like(key_left, dtype=bool)
                for cid2_i in cid2:
                    key_right = other[cid2_i, mask_right].ravel()
                    mask |= np.in1d(key_left, key_right)

                return mask.reshape(self.data[cid1[0], view].shape)

            elif len(cid2) == 1:

                key_right = other[cid2[0], mask_right].ravel()
                mask = np.zeros_like(self.data[cid1[0], view].ravel(), dtype=bool)
                for cid1_i in cid1:
                    key_left = self.data[cid1_i, view].ravel()
                    mask |= np.in1d(key_left, key_right)

                return mask.reshape(self.data[cid1[0], view].shape)

            else:

                raise Exception("Either the number of components in the key join sets "
                                "should match, or one of the component sets should ",
                                "contain a single component.")

        raise IncompatibleAttribute

    @contract(view='array_view', returns='array')
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
        try:
            return self.subset_state.to_mask(self.data, view)
        except IncompatibleAttribute as exc:
            try:
                return self._to_mask_join(view)
            except IncompatibleAttribute:
                raise exc

    @contract(value=bool)
    def do_broadcast(self, value):
        """
        Set whether state changes to the subset are relayed to a hub.

        It can be useful to turn off broadcasting, when modifying the
        subset in ways that don't impact any of the clients.

        Attributes:
        value: Whether the subset should broadcast state changes (True/False)

        """
        object.__setattr__(self, '_broadcasting', value)

    @contract(attribute='string')
    def broadcast(self, attribute):
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

    @contract(file_name='string')
    def write_mask(self, file_name, format="fits"):
        """ Write a subset mask out to file

        :param file_name: name of file to write to
        :param format:
           Name of format to write to. Currently, only "fits" is
           supported

        """
        mask = np.short(self.to_mask())
        if format == 'fits':
            try:
                from astropy.io import fits
                fits.writeto(file_name, mask, clobber=True)
            except ImportError:
                raise ImportError("Cannot write mask -- requires astropy")
        else:
            raise AttributeError("format not supported: %s" % format)

    @contract(file_name='string')
    def read_mask(self, file_name):
        try:
            from astropy.io import fits
            mask = fits.open(file_name)[0].data
        except ImportError:
            raise ImportError("Cannot read mask -- requires astropy")
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

    @contract(other_subset='isinstance(Subset)')
    def paste(self, other_subset):
        """paste subset state from other_subset onto self """
        state = other_subset.subset_state.copy()
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

    @contract(other='isinstance(Subset)', returns='isinstance(Subset)')
    def __or__(self, other):
        return _combine([self, other], operator.or_)

    @contract(other='isinstance(Subset)', returns='isinstance(Subset)')
    def __and__(self, other):
        return _combine([self, other], operator.and_)

    @contract(returns='isinstance(Subset)')
    def __invert__(self):
        return _combine([self], operator.invert)

    @contract(other='isinstance(Subset)', returns='isinstance(Subset)')
    def __xor__(self, other):
        return _combine([self, other], operator.xor)

    def __eq__(self, other):
        if not isinstance(other, Subset):
            return False
        # XXX need to add equality specification for subset states
        return (self.subset_state == other.subset_state and
                self.style == other.style)

    def state_as_mask(self):
        """
        Convert the current SubsetState to a MaskSubsetState
        """
        try:
            m = self.to_mask()
        except IncompatibleAttribute:
            m = np.zeros(self.data.shape, dtype=np.bool)
        cids = self.data.pixel_component_ids
        return MaskSubsetState(m, cids)

    # In Python 2 we need to do this explicitly
    def __ne__(self, other):
        return not self.__eq__(other)

    # In Python 3, if __eq__ is defined, then __hash__ has to be re-defined
    if PY3:
        __hash__ = object.__hash__


class SubsetState(object):

    def __init__(self):
        pass

    @property
    def attributes(self):
        return tuple()

    @property
    def subset_state(self):  # convenience method, mimic interface of Subset
        return self

    @contract(data='isinstance(Data)')
    def to_index_list(self, data):
        return np.where(self.to_mask(data).flat)[0]

    @contract(data='isinstance(Data)', view='array_view')
    def to_mask(self, data, view=None):
        shp = view_shape(data.shape, view)
        return np.zeros(shp, dtype=bool)

    @contract(returns='isinstance(SubsetState)')
    def copy(self):
        return SubsetState()

    @contract(other_state='isinstance(SubsetState)',
              returns='isinstance(SubsetState)')
    def __or__(self, other_state):
        return OrState(self, other_state)

    @contract(other_state='isinstance(SubsetState)',
              returns='isinstance(SubsetState)')
    def __and__(self, other_state):
        return AndState(self, other_state)

    @contract(returns='isinstance(SubsetState)')
    def __invert__(self):
        return InvertState(self)

    @contract(other_state='isinstance(SubsetState)',
              returns='isinstance(SubsetState)')
    def __xor__(self, other_state):
        return XorState(self, other_state)


class RoiSubsetState(SubsetState):

    def __init__(self, xatt=None, yatt=None, roi=None):
        super(RoiSubsetState, self).__init__()
        self.xatt = xatt
        self.yatt = yatt
        self.roi = roi

    @property
    def attributes(self):
        return (self.xatt, self.yatt)

    @contract(data='isinstance(Data)', view='array_view')
    def to_mask(self, data, view=None):

        # TODO: make sure that pixel components don't actually take up much
        #       memory and are just views

        x = data[self.xatt, view]
        y = data[self.yatt, view]

        # We do the import here to avoid circular imports
        from glue.core.component_id import PixelComponentID

        if (x.ndim == data.ndim and
            isinstance(self.xatt, PixelComponentID) and
            isinstance(self.yatt, PixelComponentID)):

            # This is a special case - the ROI is defined in pixel space, so we
            # can apply it to a single slice and then broadcast it to all other
            # dimensions. We start off by extracting a slice which takes only
            # the first elements of all dimensions except the attributes in
            # question, for which we take all the elements. We need to preserve
            # the dimensionality of the array, hence the use of slice(0, 1).
            # Note that we can only do this if the view (if present) preserved
            # the dimensionality, which is why we checked that x.ndim == data.ndim

            subset = []
            for i in range(data.ndim):
                if i == self.xatt.axis or i == self.yatt.axis:
                    subset.append(slice(None))
                else:
                    subset.append(slice(0, 1))

            x_slice = x[subset]
            y_slice = y[subset]

            if self.roi.defined():
                result = self.roi.contains(x_slice, y_slice)
            else:
                result = np.zeros(x_slice.shape, dtype=bool)

            result = broadcast_to(result, x.shape)

        else:

            if self.roi.defined():
                result = self.roi.contains(x, y)
            else:
                result = np.zeros(x.shape, dtype=bool)

        if result.shape != x.shape:
            raise ValueError("Unexpected error: boolean mask has incorrect dimensions")

        return result

    def copy(self):
        result = RoiSubsetState()
        result.xatt = self.xatt
        result.yatt = self.yatt
        result.roi = self.roi
        return result


class CategoricalROISubsetState(SubsetState):

    def __init__(self, att=None, roi=None):
        super(CategoricalROISubsetState, self).__init__()
        self.att = att
        self.roi = roi

    @property
    def attributes(self):
        return self.att,

    @memoize
    @contract(data='isinstance(Data)', view='array_view')
    def to_mask(self, data, view=None):
        x = data.get_component(self.att)._categorical_data[view]
        result = self.roi.contains(x, None)
        assert x.shape == result.shape
        return result.ravel()

    def copy(self):
        result = CategoricalROISubsetState()
        result.att = self.att
        result.roi = self.roi
        return result

    @staticmethod
    def from_range(component, att, lo, hi):

        roi = CategoricalROI.from_range(component, lo, hi)
        subset = CategoricalROISubsetState(roi=roi,
                                           att=att)
        return subset


class RangeSubsetState(SubsetState):

    def __init__(self, lo, hi, att=None):
        super(RangeSubsetState, self).__init__()
        self.lo = lo
        self.hi = hi
        self.att = att

    @property
    def attributes(self):
        return (self.att,)

    @contract(data='isinstance(Data)', view='array_view')
    def to_mask(self, data, view=None):
        x = data[self.att, view]
        result = (x >= self.lo) & (x <= self.hi)
        return result

    def copy(self):
        return RangeSubsetState(self.lo, self.hi, self.att)


class MultiRangeSubsetState(SubsetState):
    """
    A subset state defined by multiple discontinuous ranges

    Parameters
    ----------
    pairs : list
        A list of (lo, hi) tuples
    """

    def __init__(self, pairs, att=None):
        super(MultiRangeSubsetState, self).__init__()
        self.pairs = pairs
        self.att = att

    @property
    def attributes(self):
        return (self.att,)

    @contract(data='isinstance(Data)', view='array_view')
    def to_mask(self, data, view=None):
        x = data[self.att, view]
        result = np.zeros_like(x, dtype=bool)
        for lo, hi in self.pairs:
            result |= (x >= lo) & (x <= hi)
        return result

    def copy(self):
        return MultiRangeSubsetState(self.pairs, self.att)


class CategoricalROISubsetState2D(object):
    """
    A 2D subset state where both attributes are categorical.

    Parameters
    ----------
    categories : dict
        A dictionary containing for each label of one categorical component an
        interable of labels for the other categorical component (using sets will
        provide the best performance)
    att1 : :class:`~glue.core.component_id.ComponentID`
        The component ID matching the keys of the ``categories`` dictionary
    att2 : :class:`~glue.core.component_id.ComponentID`
        The component ID matching the values of the ``categories`` dictionary
    """
    def __init__(self, categories, att1, att2):
        self.categories = categories
        self.att1 = att1
        self.att2 = att2

    @property
    def attributes(self):
        return (self.att1, self.att2)

    @memoize
    @contract(data='isinstance(Data)', view='array_view')
    def to_mask(self, data, view=None):

        # Extract categories and numerical values
        labels1 = data.get_component(self.att1).labels
        labels2 = data.get_component(self.att2).labels

        if view is not None:
            labels1 = labels1[view]
            labels2 = labels2[view]

        # Initialize empty mask
        mask = np.zeros(labels1.shape, dtype=bool)

        # A loop over all values here is actually reasonably efficient compared
        # to alternatives. Any improved implementation, even vectorized, should
        # ensure that it is more efficient for large numbers of categories and
        # values.
        for i in range(len(labels1)):
            if labels1[i] in self.categories:
                if labels2[i] in self.categories[labels1[i]]:
                    mask[i] = True

        return mask

    def copy(self):
        result = CategoricalROISubsetState2D(self.categories,
                                             self.att1, self.att2)
        return result


class CategoricalMultiRangeSubsetState(SubsetState):
    """
    A 2D subset state where one attribute is categorical and the other is
    numerical, and where for each category, there are multiple possible subset
    ranges.

    Parameters
    ----------
    ranges : dict
        A dictionary containing for each category (key), a list of tuples
        giving the ranges of values for the numerical attribute.
    cat_att : :class:`~glue.core.component_id.ComponentID`
        The component ID for the categorical attribute
    num_att : :class:`~glue.core.component_id.ComponentID`
        The component ID for the numerical attribute
    """

    def __init__(self, ranges, cat_att, num_att):
        self.ranges = ranges
        self.cat_att = cat_att
        self.num_att = num_att

    @property
    def attributes(self):
        return (self.cat_att, self._num_att)

    @memoize
    @contract(data='isinstance(Data)', view='array_view')
    def to_mask(self, data, view=None):

        # Extract categories and numerical values
        labels = data.get_component(self.cat_att).labels
        values = data[self.num_att]

        if view is not None:
            labels = labels[view]
            values = values[view]

        # Initialize empty mask
        mask = np.zeros(values.shape, dtype=bool)

        # A loop over all values here is actually reasonably efficient compared
        # to alternatives. Any improved implementation, even vectorized, should
        # ensure that it is more efficient for large numbers of categories and
        # values. For example, using 10000 categories and 1000000 data points
        # takes 1.2 seconds on a laptop.
        for i in range(len(values)):
            if labels[i] in self.ranges:
                for lo, hi in self.ranges[labels[i]]:
                    if values[i] >= lo and values[i] <= hi:
                        mask[i] = True
                        break

        return mask

    def copy(self):
        result = CategoricalMultiRangeSubsetState(self.ranges,
                                                  self.cat_att,
                                                  self.num_att)
        return result


class CompositeSubsetState(SubsetState):
    op = None

    def __init__(self, state1, state2=None):
        super(CompositeSubsetState, self).__init__()
        self.state1 = state1.copy()
        if state2:
            state2 = state2.copy()
        self.state2 = state2

    def copy(self):
        return type(self)(self.state1, self.state2)

    @property
    def attributes(self):
        att = self.state1.attributes
        if self.state2 is not None:
            att += self.state2.attributes
        return tuple(sorted(set(att)))

    @memoize
    @contract(data='isinstance(Data)', view='array_view')
    def to_mask(self, data, view=None):
        return self.op(self.state1.to_mask(data, view),
                       self.state2.to_mask(data, view))

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

    @memoize
    @contract(data='isinstance(Data)', view='array_view')
    def to_mask(self, data, view=None):
        return ~self.state1.to_mask(data, view)

    def __str__(self):
        return "(~%s)" % self.state1


class MaskSubsetState(SubsetState):

    """
    A subset defined by boolean pixel mask
    """

    def __init__(self, mask, cids):
        """
        :param cids: List of ComponentIDs, defining the pixel coordinate space of the mask
        :param mask: Boolean ndarray
        """
        self.cids = cids
        self.mask = mask

    def copy(self):
        return MaskSubsetState(self.mask, self.cids)

    def to_mask(self, data, view=None):

        view = view or slice(None)

        # shortcut for data on the same pixel grid
        if data.pixel_component_ids == self.cids:
            return self.mask[view].copy()

        # locate each element of data in the coordinate system of the mask
        vals = [data[c, view].astype(np.int) for c in self.cids]
        result = self.mask[vals]

        for v, n in zip(vals, data.shape):
            result &= ((v >= 0) & (v < n))

        return result

    def __gluestate__(self, context):
        return dict(cids=[context.id(c) for c in self.cids],
                    mask=context.do(self.mask))

    @classmethod
    def __setgluestate__(cls, rec, context):
        return cls(context.object(rec['mask']),
                   [context.object(c) for c in rec['cids']])


class CategorySubsetState(SubsetState):

    def __init__(self, attribute, values):
        super(CategorySubsetState, self).__init__()
        self._attribute = attribute
        self._values = np.asarray(values).ravel()

    @memoize
    def to_mask(self, data, view=None):
        vals = data[self._attribute, view]
        result = np.in1d(vals.ravel(), self._values)
        return result.reshape(vals.shape)

    def copy(self):
        return CategorySubsetState(self._attribute, self._values.copy())

    def __gluestate__(self, context):
        return dict(att=context.id(self._attribute),
                    vals=context.do(self._values))

    @classmethod
    def __setgluestate__(cls, rec, context):
        return cls(context.object(rec['att']),
                   context.object(rec['vals']))


class ElementSubsetState(SubsetState):

    def __init__(self, indices=None, data=None):
        super(ElementSubsetState, self).__init__()
        self._indices = indices
        if data is None:
            self._data_uuid = None
        else:
            self._data_uuid = data.uuid

    @memoize
    def to_mask(self, data, view=None):
        if data.uuid == self._data_uuid or self._data_uuid is None:
            # XXX this is inefficient for views
            result = np.zeros(data.shape, dtype=bool)
            if self._indices is not None:
                result.flat[self._indices] = True
            if view is not None:
                result = result[view]
            return result
        else:
            raise IncompatibleAttribute()

    def copy(self):
        state = ElementSubsetState(indices=self._indices)
        state._data_uuid = self._data_uuid
        return state

    def __gluestate__(self, context):
        return dict(indices=context.do(self._indices),
                    data_uuid=self._data_uuid)

    @classmethod
    def __setgluestate__(cls, rec, context):
        state = cls(indices=context.object(rec['indices']))
        state._data_uuid = rec['data_uuid']
        return state


class InequalitySubsetState(SubsetState):

    def __init__(self, left, right, op):
        from glue.core.component_link import ComponentLink

        super(InequalitySubsetState, self).__init__()
        from glue.core.data import ComponentID
        valid_ops = [operator.gt, operator.ge,
                     operator.lt, operator.le,
                     operator.eq, operator.ne]
        if op not in valid_ops:
            raise TypeError("Invalid boolean operator: %s" % op)
        if not isinstance(left, ComponentID) and not \
                isinstance(left, numbers.Number) and not \
                isinstance(left, ComponentLink):
            raise TypeError("Input must be ComponenID or NumberType: %s"
                            % type(left))
        if not isinstance(right, ComponentID) and not \
                isinstance(right, numbers.Number) and not \
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

    @memoize
    def to_mask(self, data, view=None):
        left = self._left
        if not isinstance(self._left, numbers.Number):
            left = data[self._left, view]

        right = self._right
        if not isinstance(self._right, numbers.Number):
            right = data[self._right, view]

        return self._operator(left, right)

    def copy(self):
        return InequalitySubsetState(self._left, self._right, self._operator)

    def __str__(self):
        sym = OPSYM.get(self._operator, self._operator)
        return "(%s %s %s)" % (self._left, sym, self._right)

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, self)


@contract(subsets='list(isinstance(Subset))', returns=Subset)
def _combine(subsets, operator):
    state = operator(*[s.subset_state for s in subsets])
    result = Subset(None)
    result.subset_state = state
    return result


def combine_multiple(subsets, operator):
    if len(subsets) == 0:
        return SubsetState()
    else:
        combined = subsets[0]
        for subset in subsets[1:]:
            combined = operator(combined, subset)
        return combined
