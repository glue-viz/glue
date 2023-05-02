import uuid
import numbers
import operator

import numpy as np

from glue.core.roi import (PolygonalROI, CategoricalROI, RangeROI, XRangeROI,
                           YRangeROI, RectangularROI, CircularROI, EllipticalROI, Projected3dROI)
from glue.core.contracts import contract
from glue.core.util import split_component_view
from glue.core.registry import Registry
from glue.core.exceptions import IncompatibleAttribute
from glue.core.message import SubsetDeleteMessage, SubsetUpdateMessage
from glue.core.decorators import memoize
from glue.core.visual import VisualAttributes
from glue.config import settings
from glue.utils import (categorical_ndarray, combine_slices, floodfill, iterate_chunks,
                        polygon_line_intersections, view_shape)


__all__ = ['Subset', 'SubsetState', 'RoiSubsetStateNd', 'RoiSubsetState', 'CategoricalROISubsetState',
           'RangeSubsetState', 'MultiRangeSubsetState', 'CompositeSubsetState',
           'OrState', 'AndState', 'XorState', 'InvertState', 'MaskSubsetState', 'CategorySubsetState',
           'ElementSubsetState', 'InequalitySubsetState', 'combine_multiple',
           'CategoricalMultiRangeSubsetState', 'CategoricalROISubsetState2D',
           'SliceSubsetState', 'roi_to_subset_state', 'MultiOrState']


OPSYM = {operator.ge: '>=', operator.gt: '>',
         operator.le: '<=', operator.lt: '<',
         operator.and_: '&', operator.or_: '|',
         operator.xor: '^', operator.eq: '==',
         operator.ne: '!='}
SYMOP = dict((v, k) for k, v in OPSYM.items())


class Subset(object):

    """
    Base class to handle subsets of data.

    These objects both describe subsets of a dataset, and relay any
    state changes to the hub that their parent data are assigned to.

    This base class only directly implements the logic that relays
    state changes back to the hub. Subclasses implement the actual
    description and manipulation of data subsets

    Parameters
    ----------
    data : :class:`~glue.core.data.Data`
        The dataset that this subset describes
    """

    @contract(data='isinstance(Data)|None',
              color='color',
              alpha=float,
              label='string|None')
    def __init__(self, data, **kwargs):
        """Create a new subset object.

        Note: the preferred way for creating subsets is via
        :func:`~glue.core.data_collection.DataCollection.new_subset_group`.
        Manually-instantiated subsets will probably *not*
        be represented properly by the UI
        """

        self._broadcasting = False  # must be first def

        self.data = data
        self.label = kwargs.get("label", None)  # trigger disambiguation

        self.subset_state = SubsetState()  # calls proper setter method

        visual_args = {k: v for k, v in kwargs.items() if k in VisualAttributes.DEFAULT_ATTS}
        visual_args.setdefault("color", settings.SUBSET_COLORS[0])
        visual_args.setdefault("alpha", 0.5)
        visual_args.setdefault("linewidth", 2.5)
        visual_args.setdefault("markersize", 7)

        self.style = VisualAttributes(parent=self, **visual_args)

        # We assign a UUID which can then be used for example in equations
        # for derived components - the idea is that this doesn't change over
        # the life cycle of glue, so it is a more reliable way to refer to
        # components in strings than using labels
        self._uuid = str(uuid.uuid4())

    @property
    def uuid(self):
        return self._uuid

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
        """
        Convenience access to subset's label.
        """
        return self._label

    @label.setter
    def label(self, value):
        """
        Set the subset's label

        Subset labels within a data object must be unique. The input
        will be auto-disambiguated if necessary
        """
        value = Registry().register(self, value, group=self.data)
        self._label = value

    @property
    def attributes(self):
        """
        Returns a tuple of the ComponentIDs that this subset depends upon.
        """
        return self.subset_state.attributes

    def register(self):
        """Register a subset to its data, and start broadcasting state changes"""
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

        Returns
        -------
        :class:`~numpy.ndarray`
            A numpy array, giving the indices of elements in the data that
            belong to this subset.

        Raises
        ------
        IncompatibleDataException
            If an index list cannot be created for the requested data set.
        """
        return self.subset_state.to_index_list(self.data)

    @contract(view='array_view', returns='array')
    def to_mask(self, view=None):
        """
        Convert the current subset to a mask.

        Parameters
        ----------
        view : object
            An optional view into the dataset (e.g. a slice)
            If present, the mask will pertain to the view and not the entire dataset.

        Returns
        -------
        :class:`~numpy.ndarray`
            A boolean numpy array, the same shape as the data, that
            defines whether each element belongs to the subset.
        """
        return self.data.get_mask(self.subset_state, view=view)

    @contract(value=bool)
    def do_broadcast(self, value):
        """
        Set whether state changes to the subset are relayed to a hub.

        It can be useful to turn off broadcasting, when modifying the
        subset in ways that don't impact any of the clients.

        Attributes
        ----------
        value : bool
            Whether the subset should broadcast state changes (True/False)

        """
        object.__setattr__(self, '_broadcasting', value)

    @contract(attribute='string')
    def broadcast(self, attribute):
        """
        Explicitly broadcast a SubsetUpdateMessage to the hub.

        Parameters
        ----------
        attribute : str
            The name of the attribute (if any) that should be broadcast as updated.
        """

        if not hasattr(self, 'data') or not hasattr(self.data, 'hub'):
            return

        if self._broadcasting and self.data.hub:
            msg = SubsetUpdateMessage(self, attribute=attribute)
            self.data.hub.broadcast(msg)

    def delete(self):
        """
        Broadcast a SubsetDeleteMessage to the hub, and stop broadcasting.

        Also removes subset reference from parent data's subsets list.
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

        Parameters
        ----------
        file_name : str
            Name of file to write to
        format : str, optional
            Name of format to write to. Currently, only "fits" is supported.
        """
        mask = np.short(self.to_mask())
        if format == 'fits':
            from astropy.io import fits
            try:
                fits.writeto(file_name, mask, overwrite=True)
            except TypeError:
                fits.writeto(file_name, mask, clobber=True)
        else:
            raise AttributeError("format not supported: %s" % format)

    @contract(file_name='string')
    def read_mask(self, file_name):
        try:
            from astropy.io import fits
            with fits.open(file_name) as hdulist:
                mask = hdulist[0].data
        except IOError:
            raise IOError("Could not read %s (not a fits file?)" % file_name)
        ind = np.where(mask.flat)[0]
        state = ElementSubsetState(indices=ind)
        self.subset_state = state

    def __del__(self):
        try:
            self.delete()
        except Exception:
            pass

    def __setattr__(self, attribute, value):
        had_attribute = hasattr(self, attribute)
        before = getattr(self, attribute, None)
        object.__setattr__(self, attribute, value)
        if not attribute.startswith('_') and (not had_attribute or np.any(before != value)):
            self.broadcast(attribute)

    def __getitem__(self, view):
        """
        Retrieve the elements from a data view within the subset.

        Parameters
        ----------
        view : object
            View of the data. See ``data.__getitem__`` for details.
        """
        c, v = split_component_view(view)
        ma = self.to_mask(v)
        return self.data[view][ma]

    @contract(other_subset='isinstance(Subset)')
    def paste(self, other_subset):
        """
        Paste subset state from other_subset onto self.
        """
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
        if self is other:
            return True
        return (self.subset_state == other.subset_state and
                self.style == other.style)

    def state_as_mask(self):
        """
        Convert the current :class:`~glue.core.subset.SubsetState` to a
        :class:`~glue.core.subset.MaskSubsetState`.
        """
        try:
            m = self.to_mask()
        except IncompatibleAttribute:
            m = np.zeros(self.data.shape, dtype=np.bool)
        cids = self.data.pixel_component_ids
        return MaskSubsetState(m, cids)

    # If __eq__ is defined, then __hash__ has to be re-defined
    __hash__ = object.__hash__

    # Provide convenient access to Data methods/properties that make sense
    # here too.

    def component_ids(self):
        return self.data.component_ids()

    @property
    def components(self):
        return self.data.components

    @property
    def coordinate_components(self):
        return self.data.coordinate_components

    @property
    def derived_components(self):
        return self.data.derived_components

    @property
    def main_components(self):
        return self.data.main_components

    @property
    def pixel_component_ids(self):
        return self.data.pixel_component_ids

    @property
    def world_component_ids(self):
        return self.data.world_component_ids

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    @property
    def hub(self):
        return self.data.hub

    # DEPRECATED (warnings raised in Data)

    @property
    def primary_components(self):
        return self.data.primary_components

    @property
    def visible_components(self):
        return self.data.visible_components


class SubsetState(object):
    """
    The base class for all subset states.

    This defaults to an empty subset.
    """

    def __init__(self):
        pass

    @property
    def attributes(self):
        """
        The attributes that the subset state depends on.
        """
        return tuple()

    @property
    def subset_state(self):  # convenience method, mimic interface of Subset
        return self

    def center(self):
        """Return center of underlying ROI, if any."""
        return  # None until explicitly implemented by subclass

    def move_to(self, *args):
        """Move any underlying ROI to the new given center."""
        pass  # no-op until explicitly implemented by subclass

    @contract(data='isinstance(Data)')
    def to_index_list(self, data):
        return np.where(data.get_mask(self.subset_state).flat)[0]

    @contract(data='isinstance(Data)', view='array_view')
    def to_mask(self, data, view=None):
        """
        Compute the mask for this subset state.

        Parameters
        ----------
        data : :class:`~glue.core.data.Data`
            The dataset to compute the mask for.
        view
            Any object that returns a valid view for a Numpy array.
        """
        shp = view_shape(data.shape, view)
        return np.broadcast_to(False, shp)

    @contract(returns='isinstance(SubsetState)')
    def copy(self):
        """
        Return a copy of the subset state.
        """
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


class RoiSubsetStateNd(SubsetState):
    """
    A subset defined as the set of points in N dimensions that lie inside
    a region of interest (ROI).

    The dimensions are defined as numerical data attributes.

    Parameters
    ----------
    atts : list of :class:`~glue.core.component_id.ComponentID`
        The data attributes that define the dimensions of the region.
    roi : :class:`~glue.core.roi.Roi`
        The region of interest.
    pretransform : callable, optional
        A function that can be optionally applied to the data before
        checking points against the region.
    """

    def __init__(self, atts=[], roi=None, pretransform=None):
        self._atts = atts
        self._roi = roi
        self._pretransform = pretransform

    @property
    def roi(self):
        """
        The region of interest.
        """
        return self._roi

    @roi.setter
    def roi(self, value):
        self._roi = value

    @property
    def pretransform(self):
        """
        An optional transformation function to apply before checking if points are in the ROI.
        """
        return self._pretransform

    @pretransform.setter
    def pretransform(self, value):
        if not callable(value) and value is not None:
            raise TypeError("The pretransform must be callable or None.")
        self._pretransform = value

    @property
    def attributes(self):
        return tuple(self._atts)

    def center(self):
        return self._roi.center()

    def move_to(self, *args):
        self._roi.move_to(*args)

    @contract(data='isinstance(Data)', view='array_view')
    def to_mask(self, data, view=None):

        # TODO: make sure that pixel components don't actually take up much
        #       memory and are just views

        raw_comps = []
        for att in self._atts:
            raw_comps.append(data[att, view])
        res_shape = raw_comps[0].shape
        if not self.roi.defined():
            return np.zeros(raw_comps[0].shape, dtype=bool)

        if raw_comps[0].ndim == data.ndim and all([att in data.pixel_component_ids for att in self._atts]):
            # This is a special case - the ROI is defined in pixel space, so we
            # can apply it to a single slice and then broadcast it to all other
            # dimensions. We start off by extracting a slice which takes only
            # the first elements of all dimensions except the attributes in
            # question, for which we take all the elements. We need to preserve
            # the dimensionality of the array, hence the use of slice(0, 1).
            # Note that we can only do this if the view (if present) preserved
            # the dimensionality, which is why we checked that raw_comps[0].ndim == data.ndim.
            axis_ids = [att.axis for att in self._atts]
            subset = []
            for i in range(data.ndim):
                if i in axis_ids:
                    subset.append(slice(None))
                else:
                    subset.append(slice(0, 1))
            for i in range(len(raw_comps)):
                raw_comps[i] = raw_comps[i][tuple(subset)]

        if self.pretransform:
            transformed_points = []
            for slices in iterate_chunks(raw_comps[0].shape, n_max=1000000):
                comp_subsets = []
                for raw_comp in raw_comps:
                    comp_subsets.append(raw_comp[slices])
                res = self.pretransform(*comp_subsets)

                # Do this here in case the pretransform changes the dimensionality
                # e.g. 3D input to a 2D projection like Projected3dROI does internally
                while len(transformed_points) < len(res):
                    transformed_points.append(np.zeros(raw_comps[0].shape))

                for i in range(len(res)):
                    transformed_points[i][slices] = res[i]
        else:
            transformed_points = raw_comps

        if isinstance(self.roi, Projected3dROI):
            result = self.roi.contains3d(*transformed_points)
        else:
            result = self.roi.contains(*transformed_points)

        if result.shape != res_shape:
            result = np.broadcast_to(result, res_shape)

        return result


class RoiSubsetState(RoiSubsetStateNd):
    """
    A subset defined as the set of points in two dimensions that lie inside
    a region of interest (ROI).

    The two dimensions are defined as two numerical data attributes.

    Parameters
    ----------
    xatt : :class:`~glue.core.component_id.ComponentID`
        The data attribute on the x axis.
    yatt : :class:`~glue.core.component_id.ComponentID`
        The data attribute on the y axis.
    roi : :class:`~glue.core.roi.Roi`
        The region of interest.
    pretransform: callable, optional
        A function that can be optionally applied to the data before
        checking points against the region.
    """

    @contract(xatt='isinstance(ComponentID)', yatt='isinstance(ComponentID)')
    def __init__(self, xatt=None, yatt=None, roi=None, pretransform=None):
        super(RoiSubsetState, self).__init__(atts=[xatt, yatt], roi=roi, pretransform=pretransform)

    @property
    def xatt(self):
        """
        The data attribute on the x axis.
        """
        return self._atts[0]

    @xatt.setter
    def xatt(self, value):
        self._atts[0] = value

    @property
    def yatt(self):
        """
        The data attribute on the y axis.
        """
        return self._atts[1]

    @yatt.setter
    def yatt(self, value):
        self._atts[1] = value

    def copy(self):
        result = RoiSubsetState()
        result.xatt = self.xatt
        result.yatt = self.yatt
        result.roi = self.roi
        result.pretransform = self.pretransform
        return result


class CategoricalROISubsetState(SubsetState):
    """
    A subset defined as the set of values for a categorical data attribute that
    fall inside a categorical region of interest (ROI).

    Parameters
    ----------
    att : :class:`~glue.core.component_id.ComponentID`
        The categorical data attribute used for the subset.
    roi : :class:`~glue.core.roi.CategoricalROI`
        The categorical region of interest.
    """

    def __init__(self, att=None, roi=None):
        super(CategoricalROISubsetState, self).__init__()
        self._att = att
        self._roi = roi

    @property
    def att(self):
        """
        The categorical data attribute used for the subset.
        """
        return self._att

    @att.setter
    def att(self, value):
        self._att = value

    @property
    def roi(self):
        """
        The categorical region of interest.
        """
        return self._roi

    @roi.setter
    def roi(self, value):
        self._roi = value

    @property
    def attributes(self):
        return self.att,

    @memoize
    @contract(data='isinstance(Data)', view='array_view')
    def to_mask(self, data, view=None):
        x = data[self.att, view]
        result = self.roi.contains(x, None)
        assert x.shape == result.shape
        return result

    def copy(self):
        result = CategoricalROISubsetState()
        result.att = self.att
        result.roi = self.roi
        return result

    @staticmethod
    def from_range(categories, att, lo, hi):

        roi = CategoricalROI.from_range(categories, lo, hi)
        subset = CategoricalROISubsetState(roi=roi, att=att)
        return subset

    def __gluestate__(self, context):
        return dict(att=context.id(self.att),
                    roi=context.id(self.roi))

    @classmethod
    def __setgluestate__(cls, rec, context):
        return cls(att=context.object(rec['att']), roi=context.object(rec['roi']))


class RangeSubsetState(SubsetState):
    """
    A subset defined as the set of values inside a range.

    The range is defined as being inclusive (that is, values equal to the lower
    or upper bounds are considered to be inside the subset).

    Parameters
    ----------
    lo : `float`
        The lower limit of the range.
    hi : `float`
        The upper limit of the range.
    att : :class:`~glue.core.component_id.ComponentID`
        The attribute being used for the subset.
    """

    def __init__(self, lo, hi, att=None):
        super(RangeSubsetState, self).__init__()
        self._lo = lo
        self._hi = hi
        self._att = att

    @property
    def lo(self):
        """
        The lower limit of the range.
        """
        return self._lo

    @lo.setter
    def lo(self, value):
        self._lo = value

    @property
    def hi(self):
        """
        The upper limit of the range.
        """
        return self._hi

    @hi.setter
    def hi(self, value):
        self._hi = value

    @property
    def att(self):
        """
        The attribute being used for the subset.
        """
        return self._att

    @att.setter
    def att(self, value):
        self._att = value

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

    The ranges are defined as being inclusive (that is, values equal to the
    lower or upper bounds are considered to be inside the subset).

    Parameters
    ----------
    pairs : list
        A list of (lo, hi) tuples.
    att : :class:`~glue.core.component_id.ComponentID`
        The attribute being used for the subset.
    """

    def __init__(self, pairs, att=None):
        super(MultiRangeSubsetState, self).__init__()
        self._pairs = pairs
        self._att = att

    @property
    def pairs(self):
        """
        A list of (lo, hi) tuples.
        """
        return self._pairs

    @pairs.setter
    def pairs(self, value):
        self._pairs = value

    @property
    def att(self):
        """
        The attribute being used for the subset.
        """
        return self._att

    @att.setter
    def att(self, value):
        self._att = value

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


class CategoricalROISubsetState2D(SubsetState):
    """
    A subset defined as the set of values for two categorical data attributes
    that fall inside a categorical region of interest (ROI).

    Parameters
    ----------
    categories : dict
        A dictionary containing for each label of one categorical component an
        iterable of labels for the other categorical component (using sets will
        provide the best performance)
    att1 : :class:`~glue.core.component_id.ComponentID`
        The component ID matching the keys of the ``categories`` dictionary
    att2 : :class:`~glue.core.component_id.ComponentID`
        The component ID matching the values of the ``categories`` dictionary
    """

    def __init__(self, categories, att1, att2):
        self._categories = categories
        self._att1 = att1
        self._att2 = att2

    @property
    def categories(self):
        """
        A dictionary containing for each label of one categorical component an
        iterable of labels for the other categorical component.
        """
        return self._categories

    @categories.setter
    def categories(self, value):
        self._categories = value

    @property
    def att1(self):
        """
        The component ID matching the keys of the ``categories`` dictionary.
        """
        return self._att1

    @att1.setter
    def att1(self, value):
        self._att1 = value

    @property
    def att2(self):
        """
        The component ID matching the values of the ``categories`` dictionary.
        """
        return self._att2

    @att2.setter
    def att2(self, value):
        self._att2 = value

    @property
    def attributes(self):
        return (self.att1, self.att2)

    @memoize
    @contract(data='isinstance(Data)', view='array_view')
    def to_mask(self, data, view=None):

        # Extract categories and numerical values
        labels1 = data[self.att1, view]
        labels2 = data[self.att2, view]

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

    def __gluestate__(self, context):
        return dict(categories=self.categories,
                    att1=context.id(self.att1),
                    att2=context.id(self.att2))

    @classmethod
    def __setgluestate__(cls, rec, context):
        return cls(categories=rec['categories'],
                   att1=context.object(rec['att1']),
                   att2=context.object(rec['att2']))


class CategoricalMultiRangeSubsetState(SubsetState):
    """
    A subset state defined by two attributes where one attribute is categorical
    and the other is numerical, and where for each category, there are multiple
    possible subset ranges.

    Parameters
    ----------
    ranges : dict
        A dictionary containing for each category (key), a list of tuples
        giving the ranges of values for the numerical attribute.
    cat_att : :class:`~glue.core.component_id.ComponentID`
        The component ID for the categorical attribute.
    num_att : :class:`~glue.core.component_id.ComponentID`
        The component ID for the numerical attribute.
    """

    def __init__(self, ranges, cat_att, num_att):
        self.ranges = ranges
        self.cat_att = cat_att
        self.num_att = num_att

    @property
    def ranges(self):
        """
        A dictionary containing for each category (key), a list of tuples
        giving the ranges of values for the numerical attribute.
        """
        return self._ranges

    @ranges.setter
    def ranges(self, value):
        self._ranges = value

    @property
    def cat_att(self):
        """
        The component ID for the categorical attribute.
        """
        return self._cat_att

    @cat_att.setter
    def cat_att(self, value):
        self._cat_att = value

    @property
    def num_att(self):
        """
        The component ID for the numerical attribute.
        """
        return self._num_att

    @num_att.setter
    def num_att(self, value):
        self._num_att = value

    @property
    def attributes(self):
        return (self.cat_att, self._num_att)

    @memoize
    @contract(data='isinstance(Data)', view='array_view')
    def to_mask(self, data, view=None):

        # Extract categories and numerical values
        labels = data[self.cat_att]
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

    def __gluestate__(self, context):
        return dict(ranges=self.ranges,
                    cat_att=context.id(self.cat_att),
                    num_att=context.id(self.num_att))

    @classmethod
    def __setgluestate__(cls, rec, context):
        return cls(ranges=rec['ranges'],
                   cat_att=context.object(rec['cat_att']),
                   num_att=context.object(rec['num_att']))


class CompositeSubsetState(SubsetState):
    """
    The base class for combinations of subset states.
    """

    op = None

    def __init__(self, state1, state2=None):
        super(CompositeSubsetState, self).__init__()
        self.state1 = state1.copy()
        if state2:
            state2 = state2.copy()
        self.state2 = state2

    def copy(self):
        return type(self)(self.state1, self.state2)

    def center(self):
        cen = self.state1.center()
        if cen is None and self.state2:
            cen = self.state2.center()
        return cen

    def move_to(self, *args):
        """Move any underlying ROI to the new given center."""
        if self.state2:
            cen1 = self.state1.center()
            cen2 = self.state2.center()
            if cen2 is not None and cen1 is not None:
                offset = np.asarray(cen2) - np.asarray(cen1)
                if np.isscalar(offset):
                    mt_args = (args[0] + offset, )
                else:
                    mt_args = tuple(map(operator.add, args, offset))
            else:
                mt_args = args
            self.state2.move_to(*mt_args)
        self.state1.move_to(*args)

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
    """
    An 'or' logical combination of subset states.

    The two states can be accessed using the attributes ``state1`` and
    ``state2``.
    """
    op = operator.or_


class AndState(CompositeSubsetState):
    """
    An 'and' logical combination of subset states.

    The two states can be accessed using the attributes ``state1`` and
    ``state2``.
    """
    op = operator.and_


class XorState(CompositeSubsetState):
    """
    An 'exclusive or' logical combination of subset states.

    The two states can be accessed using the attributes ``state1`` and
    ``state2``.
    """
    op = operator.xor


class InvertState(CompositeSubsetState):
    """
    An inverted subset state.

    Values inside the original subset are now considered outside, and vice-versa.
    The original subset state can be accessed using the attribute ``state1``.
    """

    @memoize
    @contract(data='isinstance(Data)', view='array_view')
    def to_mask(self, data, view=None):
        return ~self.state1.to_mask(data, view)

    def __str__(self):
        return "(~%s)" % self.state1


class MultiOrState(SubsetState):
    """
    A state for many states to be combined together with an 'or' operation.

    This is meant to be used for cases where many subset states are meant to be
    combined together and provides significant performance enhancements compared
    to chaining individual OrStates
    """

    def __init__(self, states):
        super(MultiOrState, self).__init__()
        if len(states) < 1:
            raise ValueError("states should contain at least one subset state")
        self.states = states

    def copy(self):
        return type(self)(self.states)

    @property
    def attributes(self):
        att = self.states[0].attributes
        for state in self.states[1:]:
            att += state.attributes
        return tuple(sorted(set(att)))

    @memoize
    @contract(data='isinstance(Data)', view='array_view')
    def to_mask(self, data, view=None):
        # Copy the first mask so that we can then modify it in-place
        result = self.states[0].to_mask(data, view=view).copy()
        for state in self.states[1:]:
            result |= state.to_mask(data, view=view)
        return result

    def __str__(self):
        return "('or' combination of {0} individual states)".format(len(self.states))


class MaskSubsetState(SubsetState):
    """
    A subset defined by a boolean mask.

    Parameters
    ----------
    mask : :class:`~numpy.ndarray`
        The boolean mask to apply to the data.
    cids : iterable of :class:`~glue.core.component_id.ComponentID`
        The component IDs along which the mask applies.
    """

    def __init__(self, mask, cids):
        self._cids = cids
        self._mask = np.asarray(mask, dtype=bool)

    @property
    def mask(self):
        """
        The boolean mask to apply to the data.
        """
        return self._mask

    @mask.setter
    def mask(self, value):
        self._mask = value

    @property
    def cids(self):
        """
        The component IDs along which the mask applies.
        """
        return self._cids

    @cids.setter
    def cids(self, value):
        self._cids = value

    @property
    def attributes(self):
        return self._cids

    def copy(self):
        return MaskSubsetState(self.mask, self.cids)

    def to_mask(self, data, view=None):

        if view is None:
            view = slice(None)

        # shortcut for data on the same pixel grid
        if data.pixel_component_ids == self.cids:
            return self.mask[view].copy()

        # locate each element of data in the coordinate system of the mask
        vals = [data[c, view].astype(int) for c in self.cids]
        result = self.mask[tuple(vals)]

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


class SliceSubsetState(SubsetState):
    """
    A subset defined by a set of array slices.

    Parameters
    ----------
    reference_data : :class:`~glue.core.data.Data`
        The data in whose space the slices are defined.
    slices : iterable of :class:`slice`
        An iterable containing :class:`slice` objects to apply to the data.
    """

    def __init__(self, reference_data, slices):
        self._reference_data = reference_data
        self._slices = slices
        self._pad_slices()

    @property
    def reference_data(self):
        """
        The data in whose space the slices are defined.
        """
        return self._reference_data

    @reference_data.setter
    def reference_data(self, value):
        self._reference_data = value

    @property
    def slices(self):
        """
        An iterable containing :class:`slice` objects to apply to the data.
        """
        return self._slices

    @slices.setter
    def slices(self, value):
        self._slices = value

    def _pad_slices(self):
        from glue.core.data import BaseCartesianData
        if isinstance(self.reference_data, BaseCartesianData) and len(self.slices) < self.reference_data.ndim:
            self.slices = self.slices + [slice(None)] * (self.reference_data.ndim - len(self.slices))

    @property
    def attributes(self):
        return self._reference_data.pixel_component_ids

    def copy(self):
        return SliceSubsetState(self.reference_data, self.slices)

    def to_mask(self, data, view=None):

        if view is None:
            view = Ellipsis
        elif isinstance(view, slice) or np.isscalar(view):
            view = (view,)

        # Figure out the shape of the final mask given the requested view
        shape = view_shape(data.shape, view)

        if data is self.reference_data:

            slices = self.slices

        else:

            # Check if we can transform list of slices to match this dataset
            order = data.pixel_aligned_data.get(self.reference_data, None)

            if order is None:
                # We use broadcast_to for minimal memory usage
                return np.broadcast_to(False, shape)
            else:
                # Reorder slices
                slices = [self.slices[idx] for idx in order]

        if (isinstance(view, np.ndarray) or
                (isinstance(view, (tuple, list)) and isinstance(view[0], np.ndarray))):
            mask = np.zeros(data.shape, dtype=bool)
            mask[tuple(slices)] = True
            return mask[view]

        # The original slices assume the full array, not the array with the view
        # applied, so we need to now adjust the slices accordingly.
        if view is Ellipsis:
            subslices = slices
        else:
            subslices = []
            for i in range(data.ndim):
                if i >= len(view):
                    subslices.append(slices[i])
                elif np.isscalar(view[i]):
                    beg, end, stp = slices[i].indices(data.shape[i])
                    if view[i] < beg or view[i] >= end or (view[i] - beg) % stp != 0:
                        return np.broadcast_to(False, shape)
                elif isinstance(view[i], slice):
                    if view[i].step is not None and view[i].step < 0:
                        beg, end, step = view[i].indices(data.shape[i])
                        v = slice(end + 1, beg + 1, -step)
                    else:
                        v = view[i]
                    subslices.append(combine_slices(v, slices[i], data.shape[i]))
                else:
                    raise TypeError("Unexpected view item: {0}".format(view[i]))

        # Create mask with final shape
        mask = np.zeros(shape, dtype=bool)
        mask[tuple(subslices)] = True

        return mask

    def to_array(self, data, att):
        if data is self.reference_data:
            slices = self.slices
        else:
            order = data.pixel_aligned_data.get(self.reference_data, None)
            if order is None:
                raise IncompatibleAttribute()
            slices = [self.slices[idx] for idx in order]
        return data[att, tuple(slices)]

    def __gluestate__(self, context):
        return dict(slices=context.do(self.slices),
                    reference_data=context.id(self.reference_data))

    @classmethod
    def __setgluestate__(cls, rec, context):
        return cls(rec['reference_data'], context.object(rec['slices']))

    def __setgluestate_callback__(self, context):
        self.reference_data = context.object(self.reference_data)
        self._pad_slices()


class CategorySubsetState(SubsetState):
    """
    A subset defined by the set of categorical values that are equal to
    a set of categories.

    Parameters
    ----------
    att : :class:`~glue.core.component_id.ComponentID`
        The categorical data attribute used for the subset.
    categories : iterable
        The categories that the attribute should be equal to. These should be
        given as the integer codes, not the categorical labels.
    """

    def __init__(self, att, categories):
        super(CategorySubsetState, self).__init__()
        self._att = att
        self._categories = np.asarray(categories).ravel()

    @property
    def att(self):
        """
        The categorical data attribute used for the subset.
        """
        return self._att

    @att.setter
    def att(self, value):
        self._att = value

    @property
    def categories(self):
        """
        The categories that the attribute should be equal to. These should be
        given as the integer codes, not the categorical labels.
        """
        return self._categories

    @categories.setter
    def categories(self, value):
        self._categories = value

    @property
    def attributes(self):
        return self._att,

    @memoize
    def to_mask(self, data, view=None):
        vals = data[self._att, view]
        if isinstance(vals, categorical_ndarray):
            vals = vals.codes
        result = np.in1d(vals.ravel(), self._categories)
        return result.reshape(vals.shape)

    def copy(self):
        return CategorySubsetState(self._att, self._categories.copy())

    def __gluestate__(self, context):
        return dict(att=context.id(self._att),
                    vals=context.do(self._categories))

    @classmethod
    def __setgluestate__(cls, rec, context):
        return cls(context.object(rec['att']),
                   context.object(rec['vals']))


class ElementSubsetState(SubsetState):
    """
    A subset defined by a set of indices to apply to the data.

    Parameters
    ----------
    indices
        Any valid object that can be used to index a Numpy array.
    data : :class:`~glue.core.data.Data`
        The data in whose space the indices are defined.
    """

    def __init__(self, indices=None, data=None):
        super(ElementSubsetState, self).__init__()
        self._indices = indices
        if data is None:
            self._data_uuid = None
        else:
            self._data_uuid = data.uuid

    @property
    def indices(self):
        """
        The indices which when applied to the data give the subset.
        """
        return self._indices

    @indices.setter
    def indices(self, value):
        self._indices = value

    @property
    def data(self):
        """
        The UUID of the data in whose space the indices are defined.
        """
        return self._data_uuid

    @data.setter
    def data(self, value):
        self._data = value

    @memoize
    def to_mask(self, data, view=None):
        if data.uuid == self._data_uuid or self._data_uuid is None:
            # XXX this is inefficient for views
            result = np.zeros(data.shape, dtype=bool)
            if self._indices is not None:
                try:
                    result.flat[self._indices] = True
                except IndexError:
                    if self._data_uuid is None:
                        raise IncompatibleAttribute()
                    else:
                        raise
            if view is not None:
                result = result[view]
            return result
        else:
            raise IncompatibleAttribute()

    @property
    def attributes(self):
        return self._data.pixel_component_ids

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
        try:
            state._data_uuid = rec['data_uuid']
        except KeyError:  # BACKCOMPAT
            pass
        return state


VALID_INEQUALTIY_OPS = [operator.gt, operator.ge,
                        operator.lt, operator.le,
                        operator.eq, operator.ne]


class InequalitySubsetState(SubsetState):
    """
    A subset defined by a mathematical comparison of a attribute values to
    a reference value or attribute values.

    Parameters
    ----------
    left : float or `~glue.core.component_id.ComponentID` or str
        The value or component on the left hand side of the comparison.
    right : float or `~glue.core.component_id.ComponentID` or str
        The value or component on the right hand side of the comparison.
    operator : operator
        The comparison operator (from the :mod:`operator` module).
    """

    def __init__(self, left, right, operator):
        from glue.core.component_link import ComponentLink

        super(InequalitySubsetState, self).__init__()
        from glue.core.data import ComponentID
        if operator not in VALID_INEQUALTIY_OPS:
            raise TypeError("Invalid boolean operator: %s" % operator)
        if not isinstance(left, (ComponentID, numbers.Number,
                                 ComponentLink, str)):
            raise TypeError("Input must be ComponentID or NumberType or string: %s"
                            % type(left))

        if not isinstance(right, (ComponentID, numbers.Number,
                                  ComponentLink, str)):
            raise TypeError("Input must be ComponentID or NumberType or string: %s"
                            % type(right))
        self._left = left
        self._right = right
        self._operator = operator

    @property
    def left(self):
        """
        The value or component on the left hand side of the comparison.
        """
        return self._left

    @left.setter
    def left(self, value):
        self._left = value

    @property
    def right(self):
        """
        The value or component on the right hand side of the comparison.
        """
        return self._right

    @right.setter
    def right(self, value):
        self._right = value

    @property
    def operator(self):
        """
        The comparison operator (from the :mod:`operator` module).
        """
        return self._operator

    @operator.setter
    def operator(self, value):
        self._operator = value

    @memoize
    def to_mask(self, data, view=None):

        # FIXME: the default view in glue should be ... not None, because
        # if x is a Numpy array, x[None] has one more dimension than x. For
        # now we just fix this for the scope of this method.
        if view is None:
            view = Ellipsis

        if isinstance(self._left, (numbers.Number, str)):
            left = self._left
        else:
            left = data[self._left, view]

        if isinstance(self._right, (numbers.Number, str)):
            right = self._right
        else:
            right = data[self._right, view]

        return self._operator(left, right)

    def copy(self):
        return InequalitySubsetState(self._left, self._right, self._operator)

    def __str__(self):
        sym = OPSYM.get(self._operator, self._operator)
        return "(%s %s %s)" % (self._left, sym, self._right)

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, self)


class FloodFillSubsetState(MaskSubsetState):
    """
    A subset representing a flood-fill operation, which is computed on-the-fly.

    Parameters
    ----------
    data : :class:`~glue.core.data.Data`
        The data on which the flood fill is computed.
    att : :class:`glue.core.component_id.ComponentID`
        The attribute defining the values to use for the flood fill.
    start_coords : tuple
        The pixel coordinates of the starting point.
    threshold : float
        A value greater or equal to 1 describing the extent of the flood filling.
        The range of values selected by the flood fill is
        ``start_value * (2 -threshold)`` to ``start_value * threshold`` where
        ``start_value`` is the value of the data at ``start_coords``.
    """

    # TODO: we need to recompute the mask if the numerical values of the
    # data changes.

    def __init__(self, data, att, start_coords, threshold):

        if len(start_coords) != data.ndim:
            raise ValueError("start_coords should have as many values as data "
                             "has dimensions.")

        self._att = att
        self._data = data
        self._start_coords = tuple(start_coords)
        self._threshold = float(threshold)
        self._cids = self.data.pixel_component_ids

        self._compute_mask()

    @property
    def data(self):
        """
        The data on which the flood fill is computed.
        """
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def att(self):
        """
        The attribute defining the values to use for the flood fill.
        """
        return self._att

    @att.setter
    def att(self, value):
        self._att = value

    @property
    def start_coords(self):
        """
        The pixel coordinates of the starting point.
        """
        return self._start_coords

    @start_coords.setter
    def start_coords(self, value):
        self._start_coords = value

    @property
    def threshold(self):
        """
        A value greater or equal to 1 describing the extend of the flood
        filling. The range of values selected by the flood fill is
        ``start_value * (2 -threshold)`` to ``start_value * threshold`` where
        ``start_value`` is the value of the data at ``start_coords``.
        """
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        self._threshold = value

    def _compute_mask(self):
        mask = floodfill(self.data[self.att],
                         self.start_coords, self.threshold)
        self._mask_cache = (self._hash, mask)

    @property
    def _hash(self):
        return self.data, self.att, self.start_coords, self.threshold, self.cids

    @property
    def mask(self):
        if self._mask_cache[0] != self._hash:
            self._compute_mask()
        return self._mask_cache[1]

    @property
    def attributes(self):
        return list(self._data.pixel_component_ids) + [self.att]

    def copy(self):
        return FloodFillSubsetState(self.data, self.att, self.start_coords,
                                    self.threshold)

    def __gluestate__(self, context):
        # We don't store the data since this would cause a circular reference.
        # However we can recover the data from the attribute ComponentID.
        return dict(attribute=context.id(self.att),
                    start_coords=self.start_coords,
                    threshold=self.threshold)

    @classmethod
    def __setgluestate__(cls, rec, context):
        att = context.object(rec['attribute'])
        return cls(att.parent, att,
                   context.object(rec['start_coords']),
                   context.object(rec['threshold']))


class RoiSubsetState3d(RoiSubsetStateNd):
    """
    A subset defined as the set of points in three dimensions that lie inside
    a 3-d region of interest (ROI).

    The three dimensions are defined as three numerical data attributes.

    Parameters
    ----------
    xatt : :class:`~glue.core.component_id.ComponentID`
        The data attribute on the x axis.
    yatt : :class:`~glue.core.component_id.ComponentID`
        The data attribute on the y axis.
    zatt : :class:`~glue.core.component_id.ComponentID`
        The data attribute on the z axis.
    roi : :class:`~glue.core.roi.Roi`
        The region of interest (which should implement ``contains3d``)
    pretransform: callable, optional
        A function that can be optionally applied to the data before checking points
        against the region.
    """

    @contract(xatt='isinstance(ComponentID)', yatt='isinstance(ComponentID)', zatt='isinstance(ComponentID)')
    def __init__(self, xatt=None, yatt=None, zatt=None, roi=None, pretransform=None):
        super(RoiSubsetState3d, self).__init__(atts=[xatt, yatt, zatt], roi=roi, pretransform=pretransform)

    @property
    def xatt(self):
        """
        The data attribute on the x axis.
        """
        return self._atts[0]

    @xatt.setter
    def xatt(self, value):
        self._atts[0] = value

    @property
    def yatt(self):
        """
        The data attribute on the y axis.
        """
        return self._atts[1]

    @yatt.setter
    def yatt(self, value):
        self._atts[1] = value

    @property
    def zatt(self):
        """
        The data attribute on the z axis.
        """
        return self._atts[2]

    @zatt.setter
    def zatt(self, value):
        self._atts[2] = value

    def copy(self):
        result = RoiSubsetState3d()
        result.xatt = self.xatt
        result.yatt = self.yatt
        result.zatt = self.zatt
        result.roi = self.roi
        result.pretransform = self.pretransform
        return result

    def __gluestate__(self, context):
        return dict(xatt=context.id(self.xatt),
                    yatt=context.id(self.yatt),
                    zatt=context.id(self.zatt),
                    roi=context.id(self.roi),
                    pretransform=context.id(self.pretransform))

    @classmethod
    def __setgluestate__(cls, rec, context):
        pretrans = rec['pretransform'] if 'pretransform' in rec else None
        return RoiSubsetState3d(context.object(rec['xatt']),
                                context.object(rec['yatt']),
                                context.object(rec['zatt']),
                                context.object(rec['roi']),
                                context.object(pretrans))


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


def roi_to_subset_state(roi, x_att=None, y_att=None, x_categories=None, y_categories=None, use_pretransform=False):
    """
    Given a 2D ROI and attributes on the x and y axis, determine the
    corresponding subset state.
    """

    if isinstance(roi, RangeROI) and not use_pretransform:

        if roi.ori == 'x':
            att = x_att
            categories = x_categories
        else:
            att = y_att
            categories = y_categories

        if categories is not None:
            return CategoricalROISubsetState.from_range(categories, att, roi.min, roi.max)
        else:
            return RangeSubsetState(roi.min, roi.max, att)

    elif x_categories is not None or y_categories is not None:

        if isinstance(roi, RectangularROI):

            # In this specific case, we can decompose the rectangular ROI into
            # two RangeROIs that are combined with an 'and' logical operation.

            range1 = XRangeROI(roi.xmin, roi.xmax)
            range2 = YRangeROI(roi.ymin, roi.ymax)

            subset1 = roi_to_subset_state(range1, x_att=x_att, x_categories=x_categories)
            subset2 = roi_to_subset_state(range2, y_att=y_att, y_categories=y_categories)

            return AndState(subset1, subset2)

        elif isinstance(roi, CategoricalROI):

            # The selection is categorical itself. We assume this is along the x axis

            return CategoricalROISubsetState(roi=roi, att=x_att)

        else:

            # The selection is polygon-like, which requires special care.

            if x_categories is not None and y_categories is not None:

                # For each category, we check which categories along the other
                # axis fall inside the polygon:

                selection = {}

                for code, label in enumerate(x_categories):

                    # Determine the coordinates of the points to check
                    n_other = len(y_categories)
                    y = np.arange(n_other)
                    x = np.repeat(code, n_other)

                    # Determine which points are in the polygon, and which
                    # categories these correspond to
                    in_poly = roi.contains(x, y)
                    categories = y_categories[in_poly]

                    if len(categories) > 0:
                        selection[label] = set(categories)

                return CategoricalROISubsetState2D(selection, x_att, y_att)

            else:

                # If one of the components is not categorical, we treat this as
                # if each categorical component was mapped to a numerical value,
                # and at each value, we keep track of the polygon intersection
                # with the component. This will result in zero, one, or multiple
                # separate numerical ranges for each categorical value.

                # TODO: if we ever allow the category order to be changed, we
                # need to figure out how to update this!

                # We loop over each category and for each one we find the
                # numerical ranges

                selection = {}

                if x_categories is not None:
                    categories = x_categories
                    cat_att = x_att
                    num_att = y_att
                    x, y = roi.to_polygon()
                else:
                    categories = y_categories
                    cat_att = y_att
                    num_att = x_att
                    y, x = roi.to_polygon()

                for code, label in enumerate(categories):

                    # We determine all the numerical segments that represent the
                    # ensemble of points in y that fall in the polygon
                    # TODO: profile the following function
                    segments = polygon_line_intersections(x, y, xval=code)

                    if len(segments) > 0:
                        selection[label] = segments

                return CategoricalMultiRangeSubsetState(selection, cat_att=cat_att, num_att=num_att)

    else:

        # The selection is polygon-like or requires a pretransform and components are numerical

        if not isinstance(roi, (PolygonalROI, RectangularROI, CircularROI, EllipticalROI, RangeROI)):
            roi = PolygonalROI(*roi.to_polygon())

        subset_state = RoiSubsetState()
        subset_state.xatt = x_att
        subset_state.yatt = y_att
        subset_state.roi = roi

        return subset_state
