from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import uuid
import numpy as np
import pandas as pd

from glue.external import six
from glue.core.message import (DataUpdateMessage, DataRemoveComponentMessage,
                               DataAddComponentMessage, NumericalDataChangedMessage,
                               SubsetCreateMessage, ComponentsChangedMessage,
                               ComponentReplacedMessage, DataReorderComponentMessage,
                               ExternallyDerivableComponentsChangedMessage,
                               PixelAlignedDataChangedMessage)
from glue.core.decorators import clear_cache
from glue.core.util import split_component_view
from glue.core.hub import Hub
from glue.core.subset import Subset, SubsetState
from glue.core.component_id import ComponentIDList
from glue.core.component_link import ComponentLink, CoordinateComponentLink
from glue.core.exceptions import IncompatibleAttribute
from glue.core.visual import VisualAttributes
from glue.core.coordinates import Coordinates
from glue.core.contracts import contract
from glue.config import settings


# Note: leave all the following imports for component and component_id since
# they are here for backward-compatibility (the code used to live in this
# file)
from glue.core.component import Component, CoordinateComponent, DerivedComponent
from glue.core.component_id import ComponentID, ComponentIDDict, PixelComponentID

__all__ = ['Data']


class Data(object):

    """The basic data container in Glue.

    The data object stores data as a collection of
    :class:`~glue.core.component.Component` objects.  Each component stored in a
    dataset must have the same shape.

    Catalog data sets are stored such that each column is a distinct
    1-dimensional :class:`~glue.core.component.Component`.

    There are several ways to extract the actual numerical data stored in a
    :class:`~glue.core.data.Data` object::

       data = Data(x=[1, 2, 3], label='data')
       xid = data.id['x']

       data[xid]
       data.get_component(xid).data
       data['x']  # if 'x' is a unique component name

    Likewise, datasets support :ref:`fancy indexing <numpy:basics.indexing>`::

        data[xid, 0:2]
        data[xid, [True, False, True]]

    See also: :ref:`data_tutorial`
    """

    def __init__(self, label="", coords=None, **kwargs):
        """

        :param label: label for data
        :type label: str

        Extra array-like keywords are extracted into components
        """

        self._shape = ()

        # Components
        self._components = OrderedDict()
        self._externally_derivable_components = OrderedDict()
        self._pixel_aligned_data = OrderedDict()
        self._pixel_component_ids = ComponentIDList()
        self._world_component_ids = ComponentIDList()

        # Coordinate conversion object
        self.coords = coords or Coordinates()

        self.id = ComponentIDDict(self)

        # Metadata
        self.meta = OrderedDict()

        # Subsets of the data
        self._subsets = []

        # Hub that the data is attached to
        self.hub = None

        self.style = VisualAttributes(parent=self)

        self._coordinate_links = []

        self.data = self
        self.label = label

        self.edit_subset = None

        for lbl, data in sorted(kwargs.items()):
            self.add_component(data, lbl)

        self._key_joins = {}

        # To avoid circular references when saving objects with references to
        # the data, we make sure that all Data objects have a UUID that can
        # uniquely identify them.
        self.uuid = str(uuid.uuid4())

    @property
    def coords(self):
        """
        The coordinates object for the data.
        """
        return self._coords

    @coords.setter
    def coords(self, value):
        if (hasattr(self, '_coords') and self._coords != value) or not hasattr(self, '_coords'):
            self._coords = value
            if len(self.components) > 0:
                self._update_world_components(self.ndim)

    @property
    def subsets(self):
        """
        Tuple of subsets attached to this dataset
        """
        return tuple(self._subsets)

    @property
    def ndim(self):
        """
        Dimensionality of the dataset
        """
        return len(self.shape)

    @property
    def shape(self):
        """
        Tuple of array dimensions, like :attr:`numpy.ndarray.shape`
        """
        return self._shape

    @property
    def label(self):
        """ Convenience access to data set's label """
        return self._label

    @label.setter
    def label(self, value):
        """ Set the label to value
        """
        if getattr(self, '_label', None) != value:
            self._label = value
            self.broadcast(attribute='label')
        elif value is None:
            self._label = value

    @property
    def size(self):
        """
        Total number of elements in the dataset.
        """
        return np.product(self.shape)

    @contract(component=Component)
    def _check_can_add(self, component):
        if isinstance(component, DerivedComponent):
            return component._data is self
        else:
            if len(self._components) == 0:
                return True
            else:
                if all(comp.shape == () for comp in self._components.values()):
                    return True
                else:
                    return component.shape == self.shape

    @contract(cid=ComponentID, returns=np.dtype)
    def dtype(self, cid):
        """Lookup the dtype for the data associated with a ComponentID"""

        # grab a small piece of data
        ind = tuple([slice(0, 1)] * self.ndim)
        arr = self[cid, ind]
        return arr.dtype

    @contract(component_id=ComponentID)
    def remove_component(self, component_id):
        """ Remove a component from a data set

        :param component_id: the component to remove
        :type component_id: :class:`~glue.core.component_id.ComponentID`
        """
        # TODO: avoid too many messages when removing a component triggers
        # the removal of derived components.
        if component_id in self._components:
            self._components.pop(component_id)
            self._removed_derived_that_depend_on(component_id)
            if self.hub:
                msg = DataRemoveComponentMessage(self, component_id)
                self.hub.broadcast(msg)
                msg = ComponentsChangedMessage(self)
                self.hub.broadcast(msg)

    def _removed_derived_that_depend_on(self, component_id):
        """
        Remove internal derived components that can no longer be derived.
        """
        remove = []
        for cid in self.derived_components:
            comp = self.get_component(cid)
            if component_id in comp.link.get_from_ids():
                remove.append(cid)
        for cid in remove:
            self.remove_component(cid)

    @contract(other='isinstance(Data)',
              cid='cid_like',
              cid_other='cid_like')
    def join_on_key(self, other, cid, cid_other):
        """
        Create an *element* mapping to another dataset, by joining on values of
        ComponentIDs in both datasets.

        This join allows any subsets defined on `other` to be propagated to
        self. The different ways to call this method are described in the
        **Examples** section below.

        Parameters
        ----------
        other : :class:`~glue.core.data.Data`
            Data object to join with
        cid : str or :class:`~glue.core.component_id.ComponentID` or iterable
            Component(s) in this dataset to use as a key
        cid_other : str or :class:`~glue.core.component_id.ComponentID` or iterable
            Component(s) in the other dataset to use as a key

        Examples
        --------

        There are several ways to use this function, depending on how many
        components are passed to ``cid`` and ``cid_other``.

        **Joining on single components**

        First, one can specify a single component ID for both ``cid`` and
        ``cid_other``: this is the standard mode, and joins one component from
        one dataset to the other:

            >>> d1 = Data(x=[1, 2, 3, 4, 5], k1=[0, 0, 1, 1, 2], label='d1')
            >>> d2 = Data(y=[2, 4, 5, 8, 4], k2=[1, 3, 1, 2, 3], label='d2')
            >>> d2.join_on_key(d1, 'k2', 'k1')

        Selecting all values in ``d1`` where x is greater than 2 returns
        the last three items as expected:

            >>> s = d1.new_subset()
            >>> s.subset_state = d1.id['x'] > 2
            >>> s.to_mask()
            array([False, False,  True,  True,  True], dtype=bool)

        The linking was done between k1 and k2, and the values of
        k1 for the last three items are 1 and 2 - this means that the
        first, third, and fourth item in ``d2`` will then get selected,
        since k2 has a value of either 1 or 2 for these items.

            >>> s = d2.new_subset()
            >>> s.subset_state = d1.id['x'] > 2
            >>> s.to_mask()
            array([ True, False,  True,  True, False], dtype=bool)

        **Joining on multiple components**

        .. note:: This mode is currently slow, and will be optimized
                  significantly in future.

        Next, one can specify several components for each dataset: in this
        case, the number of components given should match for both datasets.
        This causes items in both datasets to be linked when (and only when)
        the set of keys match between the two datasets:

            >>> d1 = Data(x=[1, 2, 3, 5, 5],
            ...           y=[0, 0, 1, 1, 2], label='d1')
            >>> d2 = Data(a=[2, 5, 5, 8, 4],
            ...           b=[1, 3, 2, 2, 3], label='d2')
            >>> d2.join_on_key(d1, ('a', 'b'), ('x', 'y'))

        Selecting all items where x is 5 in ``d1`` in which x is a
        component works as expected and selects the two last items::

            >>> s = d1.new_subset()
            >>> s.subset_state = d1.id['x'] == 5
            >>> s.to_mask()
            array([False, False, False,  True,  True], dtype=bool)

        If we apply this selection to ``d2``, only items where a is 5
        and b is 2 will be selected:

            >>> s = d2.new_subset()
            >>> s.subset_state = d1.id['x'] == 5
            >>> s.to_mask()
            array([False, False,  True, False, False], dtype=bool)

        and in particular, the second item (where a is 5 and b is 3) is not
        selected.

        **One-to-many and many-to-one joining**

        Finally, you can specify one component in one dataset and multiple ones
        in the other. In the case where one component is specified for this
        dataset and multiple ones for the other dataset, then when an item
        is selected in the other dataset, it will cause any item in the present
        dataset which matches any of the keys in the other data to be selected:

            >>> d1 = Data(x=[1, 2, 3], label='d1')
            >>> d2 = Data(a=[1, 1, 2],
            ...           b=[2, 3, 3], label='d2')
            >>> d1.join_on_key(d2, 'x', ('a', 'b'))

        In this case, if we select all items in ``d2`` where a is 2, this
        will select the third item:

            >>> s = d2.new_subset()
            >>> s.subset_state = d2.id['a'] == 2
            >>> s.to_mask()
            array([False, False,  True], dtype=bool)

        Since we have joined the datasets using both a and b, we select
        all items in ``d1`` where x is either the value or a or b
        (2 or 3) which means we select the second and third item:

            >>> s = d1.new_subset()
            >>> s.subset_state = d2.id['a'] == 2
            >>> s.to_mask()
            array([False,  True,  True], dtype=bool)

        We can also join the datasets the other way around:

            >>> d1 = Data(x=[1, 2, 3], label='d1')
            >>> d2 = Data(a=[1, 1, 2],
            ...           b=[2, 3, 3], label='d2')
            >>> d2.join_on_key(d1, ('a', 'b'), 'x')

        In this case, selecting items in ``d1`` where x is 1 selects the
        first item, as expected:

            >>> s = d1.new_subset()
            >>> s.subset_state = d1.id['x'] == 1
            >>> s.to_mask()
            array([ True, False, False], dtype=bool)

        This then causes any item in ``d2`` where either a or b are 1
        to be selected, i.e. the first two items:

            >>> s = d2.new_subset()
            >>> s.subset_state = d1.id['x'] == 1
            >>> s.to_mask()
            array([ True,  True, False], dtype=bool)
        """

        # To make things easier, we transform all component inputs to a tuple
        if isinstance(cid, six.string_types) or isinstance(cid, ComponentID):
            cid = (cid,)
        if isinstance(cid_other, six.string_types) or isinstance(cid_other, ComponentID):
            cid_other = (cid_other,)

        if len(cid) > 1 and len(cid_other) > 1 and len(cid) != len(cid_other):
            raise Exception("Either the number of components in the key join "
                            "sets should match, or one of the component sets "
                            "should contain a single component.")

        def get_component_id(data, name):
            if isinstance(name, ComponentID):
                return name
            else:
                cid = data.find_component_id(name)
                if cid is None:
                    raise ValueError("ComponentID not found in %s: %s" %
                                     (data.label, name))
                return cid

        cid = tuple(get_component_id(self, name) for name in cid)
        cid_other = tuple(get_component_id(other, name) for name in cid_other)

        self._key_joins[other] = (cid, cid_other)
        other._key_joins[self] = (cid_other, cid)

    @contract(component='component_like', label='cid_like')
    def add_component(self, component, label):
        """ Add a new component to this data set.

        :param component: object to add. Can be a Component,
                          array-like object, or ComponentLink

        :param label:
              The label. If this is a string,
              a new :class:`glue.core.component_id.ComponentID` with this label will be
              created and associated with the Component

        :type component: :class:`~glue.core.component.Component` or
                         array-like
        :type label: :class:`str` or :class:`~glue.core.component_id.ComponentID`

        :raises:

           TypeError, if label is invalid
           ValueError if the component has an incompatible shape

        :returns:

           The ComponentID associated with the newly-added component
        """

        if isinstance(component, ComponentLink):
            return self.add_component_link(component, label=label)

        if not isinstance(component, Component):
            component = Component.autotyped(component)

        if isinstance(component, DerivedComponent):
            if len(self._components) == 0:
                raise TypeError("Cannot add a derived component as a first component")
            component.set_parent(self)

        if not(self._check_can_add(component)):
            raise ValueError("The dimensions of component %s are "
                             "incompatible with the dimensions of this data: "
                             "%r vs %r" % (label, component.shape, self.shape))

        if isinstance(label, ComponentID):
            component_id = label
            if component_id.parent is None:
                component_id.parent = self
        else:
            component_id = ComponentID(label, parent=self)

        if len(self._components) == 0:
            # TODO: make sure the following doesn't raise a componentsraised message
            self._create_pixel_and_world_components(ndim=component.ndim)

        # In some cases, such as when loading a session, we actually disable the
        # auto-creation of pixel and world coordinates, so the first component
        # may be a coordinate component with no shape. Therefore we only set the
        # shape once a component has a valid shape rather than strictly on the
        # first component.
        if self._shape == () and component.shape != ():
            self._shape = component.shape

        is_present = component_id in self._components
        self._components[component_id] = component

        if self.hub and not is_present:
            msg = DataAddComponentMessage(self, component_id)
            self.hub.broadcast(msg)
            msg = ComponentsChangedMessage(self)
            self.hub.broadcast(msg)

        return component_id

    def _set_externally_derivable_components(self, derivable_components):
        """
        Externally deriable components are components identified by component
        IDs from other datasets.

        This method is meant for internal use only and is called by the link
        manager. The ``derivable_components`` argument should be set to a
        dictionary where the keys are the derivable component IDs, and the
        values are DerivedComponent instances which can be used to get the
        data.
        """

        if len(self._externally_derivable_components) == 0 and len(derivable_components) == 0:

            return

        elif len(self._externally_derivable_components) == len(derivable_components):

            for key in derivable_components:
                if key in self._externally_derivable_components:
                    if self._externally_derivable_components[key].link is not derivable_components[key].link:
                        break
                else:
                    break
            else:
                return  # Unchanged!

        self._externally_derivable_components = derivable_components

        if self.hub:
            msg = ExternallyDerivableComponentsChangedMessage(self)
            self.hub.broadcast(msg)

    def _set_pixel_aligned_data(self, pixel_aligned_data):
        """
        Pixel-aligned data are datasets that contain pixel component IDs
        that are equivalent (identically, not transformed) with all pixel
        component IDs in the present dataset.

        Note that the other datasets may have more but not fewer dimensions, so
        this information may not be symmetric between datasets with differing
        numbers of dimensions.
        """

        # First check if anything has changed, as if not then we should just
        # leave things as-is and avoid emitting a message.
        if len(self._pixel_aligned_data) == len(pixel_aligned_data):
            for data in self._pixel_aligned_data:
                if data not in pixel_aligned_data or pixel_aligned_data[data] != self._pixel_aligned_data[data]:
                    break
            else:
                return

        self._pixel_aligned_data = pixel_aligned_data
        if self.hub:
            msg = PixelAlignedDataChangedMessage(self)
            self.hub.broadcast(msg)

    @property
    def pixel_aligned_data(self):
        """
        Information about other datasets in the same data collection that have
        matching or a subset of pixel component IDs.

        This is returned as a dictionary where each key is a dataset with
        matching pixel component IDs, and the value is the order in which the
        pixel component IDs of the other dataset can be found in the current
        one.
        """
        return self._pixel_aligned_data

    @contract(link=ComponentLink,
              label='cid_like|None',
              returns=DerivedComponent)
    def add_component_link(self, link, label=None):
        """
        Shortcut method for generating a new
        :class:`~glue.core.component.DerivedComponent` from a ComponentLink
        object, and adding it to a data set.

        Parameters
        ----------
        link : :class:`~glue.core.component_link.ComponentLink`
            The link to use to generate a new component
        label : :class:`~glue.core.component_id.ComponentID` or str
            The ComponentID or label to attach to.

        Returns
        -------
        component : :class:`~glue.core.component.DerivedComponent`
            The component that was added
        """
        if label is not None:
            if not isinstance(label, ComponentID):
                label = ComponentID(label, parent=self)
            link.set_to_id(label)

        if link.get_to_id() is None:
            raise TypeError("Cannot add component_link: "
                            "has no 'to' ComponentID")

        for cid in link.get_from_ids():
            if cid not in self.components:
                raise ValueError("Can only add internal links with add_component_link "
                                 "- use DataCollection.add_link to add inter-data links")

        dc = DerivedComponent(self, link)
        to_ = link.get_to_id()
        self.add_component(dc, label=to_)
        return dc

    def _create_pixel_and_world_components(self, ndim):
        self._update_pixel_components(ndim)
        self._update_world_components(ndim)

    def _update_pixel_components(self, ndim):
        for i in range(ndim):
            comp = CoordinateComponent(self, i)
            label = pixel_label(i, ndim)
            cid = PixelComponentID(i, "Pixel Axis %s" % label, parent=self)
            self.add_component(comp, cid)
            self._pixel_component_ids.append(cid)

    def _update_world_components(self, ndim):
        for cid in self._world_component_ids[:]:
            self.remove_component(cid)
            self._world_component_ids.remove(cid)
        if self.coords:
            for i in range(ndim):
                comp = CoordinateComponent(self, i, world=True)
                label = self.coords.axis_label(i)
                cid = self.add_component(comp, label)
                self._world_component_ids.append(cid)
            self._set_up_coordinate_component_links(ndim)

    def _set_up_coordinate_component_links(self, ndim):

        def make_toworld_func(i):
            def pix2world(*args):
                return self.coords.pixel2world_single_axis(*args[::-1], axis=ndim - 1 - i)
            return pix2world

        def make_topixel_func(i):
            def world2pix(*args):
                return self.coords.world2pixel_single_axis(*args[::-1], axis=ndim - 1 - i)
            return world2pix

        result = []
        for i in range(ndim):
            link = CoordinateComponentLink(self._pixel_component_ids,
                                           self._world_component_ids[i],
                                           self.coords, i)
            result.append(link)
            link = CoordinateComponentLink(self._world_component_ids,
                                           self._pixel_component_ids[i],
                                           self.coords, i, pixel2world=False)
            result.append(link)

        self._coordinate_links = result
        return result

    @property
    def components(self):
        """All :class:`ComponentIDs <glue.core.component_id.ComponentID>` in the Data

        :rtype: list
        """
        return list(self._components.keys())

    @property
    def externally_derivable_components(self):
        return list(self._externally_derivable_components.keys())

    @property
    def visible_components(self):
        """All :class:`ComponentIDs <glue.core.component_id.ComponentID>` in the Data that aren't coordinate.

        :rtype: list
        """
        return [cid for cid, comp in self._components.items()
                if not isinstance(comp, CoordinateComponent) and cid.parent is self]

    @property
    def coordinate_components(self):
        """The ComponentIDs associated with a :class:`~glue.core.component.CoordinateComponent`

        :rtype: list
        """
        return [c for c in self.component_ids() if
                isinstance(self._components[c], CoordinateComponent)]

    @property
    def main_components(self):
        return [c for c in self.component_ids() if
                not isinstance(self._components[c], (DerivedComponent, CoordinateComponent))]

    @property
    def primary_components(self):
        """The ComponentIDs not associated with a :class:`~glue.core.component.DerivedComponent`

        :rtype: list
        """
        return [c for c in self.component_ids() if
                not isinstance(self._components[c], DerivedComponent)]

    @property
    def derived_components(self):
        """The ComponentIDs for each :class:`~glue.core.component.DerivedComponent`

        :rtype: list
        """
        return [c for c in self.component_ids() if
                isinstance(self._components[c], DerivedComponent)]

    @property
    def pixel_component_ids(self):
        """
        The :class:`ComponentIDs <glue.core.component_id.ComponentID>` for each pixel coordinate.
        """
        return self._pixel_component_ids

    @property
    def world_component_ids(self):
        """
        The :class:`ComponentIDs <glue.core.component_id.ComponentID>` for each world coordinate.
        """
        return self._world_component_ids

    @contract(label='cid_like', returns='inst($ComponentID)|None')
    def find_component_id(self, label):
        """ Retrieve component_ids associated by label name.

        :param label: ComponentID or string to search for

        :returns:
            The associated ComponentID if label is found and unique, else None.
            First, this checks whether the component ID is present and unique in
            the primary (non-derived) components of the data, and if not then
            the derived components are checked. If there is one instance of the
            label in the primary and one in the derived components, the primary
            one takes precedence.
        """

        for cid_set in (self.primary_components, self.derived_components, self.coordinate_components, list(self._externally_derivable_components)):

            result = []
            for cid in cid_set:
                if isinstance(label, ComponentID):
                    if cid is label:
                        result.append(cid)
                else:
                    if cid.label == label:
                        result.append(cid)

            if len(result) == 1:
                return result[0]
            elif len(result) > 1:
                return None
        return None

    @property
    def links(self):
        """
        A list of all the links internal to the dataset.
        """
        return self.coordinate_links + self.derived_links

    @property
    def coordinate_links(self):
        """
        A list of the ComponentLinks that connect pixel and world. If no
        coordinate transformation object is present, return an empty list.
        """
        return self._coordinate_links

    @property
    def derived_links(self):
        """
        A list of the links present inside all of the DerivedComponent objects
        in this dataset.
        """
        return [self.get_component(cid).link for cid in self.derived_components]

    @contract(axis=int, returns=ComponentID)
    def get_pixel_component_id(self, axis):
        """Return the pixel :class:`glue.core.component_id.ComponentID` associated with a given axis
        """
        return self._pixel_component_ids[axis]

    @contract(axis=int, returns=ComponentID)
    def get_world_component_id(self, axis):
        """Return the world :class:`glue.core.component_id.ComponentID` associated with a given axis
        """
        return self._world_component_ids[axis]

    @contract(returns='list(inst($ComponentID))')
    def component_ids(self):
        """
        Equivalent to :attr:`Data.components`
        """
        return ComponentIDList(self._components.keys())

    @contract(subset='isinstance(Subset)|None',
              color='color|None',
              label='string|None',
              returns=Subset)
    def new_subset(self, subset=None, color=None, label=None, **kwargs):
        """
        Create a new subset, and attach to self.

        .. note:: The preferred way for creating subsets is via
            :meth:`~glue.core.data_collection.DataCollection.new_subset_group`.
            Manually-instantiated subsets will **not** be
            represented properly by the UI

        :param subset: optional, reference subset or subset state.
                       If provided, the new subset will copy the logic of
                       this subset.

        :returns: The new subset object
        """
        nsub = len(self.subsets)
        color = color or settings.SUBSET_COLORS[nsub % len(settings.SUBSET_COLORS)]
        label = label or "%s.%i" % (self.label, nsub + 1)
        new_subset = Subset(self, color=color, label=label, **kwargs)
        if subset is not None:
            new_subset.subset_state = subset.subset_state.copy()

        self.add_subset(new_subset)
        return new_subset

    @contract(subset='inst($Subset, $SubsetState)')
    def add_subset(self, subset):
        """Assign a pre-existing subset to this data object.

        :param subset: A :class:`~glue.core.subset.Subset` or
                       :class:`~glue.core.subset.SubsetState` object

        If input is a :class:`~glue.core.subset.SubsetState`,
        it will be wrapped in a new Subset automatically

        .. note:: The preferred way for creating subsets is via
            :meth:`~glue.core.data_collection.DataCollection.new_subset_group`.
            Manually-instantiated subsets will **not** be
            represented properly by the UI
        """

        if subset in self.subsets:
            return  # prevents infinite recursion
        if isinstance(subset, SubsetState):
            # auto-wrap state in subset
            state = subset
            subset = Subset(None)
            subset.subset_state = state

        self._subsets.append(subset)

        if subset.data is not self:
            subset.do_broadcast(False)
            subset.data = self
            subset.label = subset.label  # hacky. disambiguates name if needed

        if self.hub is not None:
            msg = SubsetCreateMessage(subset)
            self.hub.broadcast(msg)

        subset.do_broadcast(True)

    @contract(hub=Hub)
    def register_to_hub(self, hub):
        """ Connect to a hub.

        This method usually doesn't have to be called directly, as
        DataCollections manage the registration of data objects
        """
        if not isinstance(hub, Hub):
            raise TypeError("input is not a Hub object: %s" % type(hub))
        self.hub = hub

    @contract(attribute='string')
    def broadcast(self, attribute):
        """
        Send a :class:`~glue.core.message.DataUpdateMessage` to the hub

        :param attribute: Name of an attribute that has changed (or None)
        :type attribute: str
        """
        if not self.hub:
            return
        msg = DataUpdateMessage(self, attribute=attribute)
        self.hub.broadcast(msg)

    @contract(old=ComponentID, new=ComponentID)
    def update_id(self, old, new):
        """
        Reassign a component to a different :class:`glue.core.component_id.ComponentID`

        Parameters
        ----------
        old : :class:`glue.core.component_id.ComponentID`
            The old component ID
        new : :class:`glue.core.component_id.ComponentID`
            The new component ID
        """

        if new is old:
            return

        if new.parent is None:
            new.parent = self

        changed = False
        if old in self._components:

            # We want to keep the original order, so we can't just do:
            #   self._components[new] = self._components[old]
            # which will put the new component ID at the end, but instead
            # we need to do:
            self._components = OrderedDict((new, value) if key is old else (key, value)
                                           for key, value in self._components.items())
            changed = True

        try:
            index = self._pixel_component_ids.index(old)
            self._pixel_component_ids[index] = new
            changed = True
        except ValueError:
            pass
        try:
            index = self._world_component_ids.index(old)
            self._world_component_ids[index] = new
            changed = True
        except ValueError:
            pass

        if changed and self.hub is not None:

            # remove old component and broadcast the change
            # see #508 for discussion of this
            msg = ComponentReplacedMessage(self, old, new)
            self.hub.broadcast(msg)

    def __str__(self):
        s = "Data Set: %s\n" % self.label
        s += "Number of dimensions: %i\n" % self.ndim
        s += "Shape: %s\n" % ' x '.join([str(x) for x in self.shape])
        categories = [('Main', self.main_components),
                      ('Derived', self.derived_components),
                      ('Coordinate', self.coordinate_components)]
        for category, components in categories:
            if len(components) > 0:
                s += category + " components:\n"
                for cid in components:
                    comp = self.get_component(cid)
                    if comp.units is None or comp.units == '':
                        s += " - {0}\n".format(cid)
                    else:
                        s += " - {0} [{1}]\n".format(cid, comp.units)
        return s[:-1]

    def __repr__(self):
        return 'Data (label: %s)' % self.label

    def __setattr__(self, name, value):
        if name == "hub" and hasattr(self, 'hub') \
                and self.hub is not value and self.hub is not None:
            raise AttributeError("Data has already been assigned "
                                 "to a different hub")
        object.__setattr__(self, name, value)

    def __getitem__(self, key):
        """ Shortcut syntax to access the numerical data in a component.
        Equivalent to:

        ``component = data.get_component(component_id).data``

        :param key:
          The component to fetch data from

        :type key: :class:`~glue.core.component_id.ComponentID`

        :returns: :class:`~numpy.ndarray`
        """

        key, view = split_component_view(key)
        if isinstance(key, six.string_types):
            _k = key
            key = self.find_component_id(key)
            if key is None:
                raise IncompatibleAttribute(_k)

        if isinstance(key, ComponentLink):
            return key.compute(self, view)

        if key in self._components:
            comp = self._components[key]
        elif key in self._externally_derivable_components:
            comp = self._externally_derivable_components[key]
        else:
            raise IncompatibleAttribute(key)

        if view is not None:
            result = comp[view]
        else:
            if comp.categorical:
                result = comp.codes
            else:
                result = comp.data

        return result

    def __setitem__(self, key, value):
        """
        Wrapper for data.add_component()
        """
        self.add_component(value, key)

    @contract(component_id='cid_like|None', returns=Component)
    def get_component(self, component_id):
        """Fetch the component corresponding to component_id.

        :param component_id: the component_id to retrieve
        """
        if component_id is None:
            raise IncompatibleAttribute()

        if isinstance(component_id, six.string_types):
            component_id = self.id[component_id]

        if component_id in self._components:
            return self._components[component_id]
        elif component_id in self._externally_derivable_components:
            return self._externally_derivable_components[component_id]
        else:
            raise IncompatibleAttribute(component_id)

    def to_dataframe(self, index=None):
        """ Convert the Data object into a pandas.DataFrame object

        :param index: Any 'index-like' object that can be passed to the pandas.Series constructor

        :return: pandas.DataFrame
        """

        h = lambda comp: self.get_component(comp).to_series(index=index)
        df = pd.DataFrame(dict((comp.label, h(comp)) for comp in self.components))
        order = [comp.label for comp in self.components]
        return df[order]

    def reorder_components(self, component_ids):
        """
        Reorder the components using a list of component IDs. The new set
        of component IDs has to match the existing set (though order may differ).
        """

        # We need to be careful because component IDs overload == so we can't
        # use the normal ways to test whether the component IDs are the same
        # as self.components - instead we need to explicitly use id

        if len(component_ids) != len(self.components):
            raise ValueError("Number of component in component_ids does not "
                             "match existing number of components")

        if set(id(c) for c in self.components) != set(id(c) for c in component_ids):
            raise ValueError("specified component_ids should match existing components")

        existing = self.components
        for idx in range(len(component_ids)):
            if component_ids[idx] is not existing[idx]:
                break
        else:
            # If we get here then the suggested order is the same as the existing one
            return

        # PY3: once we drop support for Python 2 we could sort in-place using
        # the move_to_end method on OrderedDict
        self._components = OrderedDict((key, self._components[key]) for key in component_ids)

        if self.hub:
            msg = DataReorderComponentMessage(self, list(self._components))
            self.hub.broadcast(msg)

    @contract(mapping="dict(inst($Component, $ComponentID):array_like)")
    def update_components(self, mapping):
        """
        Change the numerical data associated with some of the Components
        in this Data object.

        All changes to component numerical data should use this method,
        which broadcasts the state change to the appropriate places.

        :param mapping: A dict mapping Components or ComponenIDs to arrays.

        This method has the following restrictions:
          - New compoments must have the same shape as old compoments
          - Component subclasses cannot be updated.
        """
        for comp, data in mapping.items():
            if isinstance(comp, ComponentID):
                comp = self.get_component(comp)
            data = np.asarray(data)
            if data.shape != self.shape:
                raise ValueError("Cannot change shape of data")

            comp._data = data

        # alert hub of the change
        if self.hub is not None:
            msg = NumericalDataChangedMessage(self)
            self.hub.broadcast(msg)

        for subset in self.subsets:
            clear_cache(subset.subset_state.to_mask)

    def update_values_from_data(self, data):
        """
        Replace numerical values in data to match values from another dataset.

        Notes
        -----

        This method drops components that aren't present in the new data, and
        adds components that are in the new data that were not in the original
        data. The matching is done by component label, and components are
        resized if needed. This means that for components with matching labels
        in the original and new data, the
        :class:`~glue.core.component_id.ComponentID` are preserved, and
        existing plots and selections will be updated to reflect the new
        values. Note that the coordinates are also copied, but the style is
        **not** copied.
        """

        old_labels = [cid.label for cid in self.components]
        new_labels = [cid.label for cid in data.components]

        if len(old_labels) == len(set(old_labels)):
            old_labels = set(old_labels)
        else:
            raise ValueError("Non-unique component labels in original data")

        if len(new_labels) == len(set(new_labels)):
            new_labels = set(new_labels)
        else:
            raise ValueError("Non-unique component labels in new data")

        # Remove components that don't have a match in new data
        for cname in old_labels - new_labels:
            cid = self.find_component_id(cname)
            self.remove_component(cid)

        # Update shape
        self._shape = data._shape

        # Update components that exist in both. Note that we can't just loop
        # over old_labels & new_labels since we need to make sure we preserve
        # the order of the components, and sets don't preserve order.
        for cid in self.components:
            cname = cid.label
            if cname in old_labels & new_labels:
                comp_old = self.get_component(cname)
                comp_new = data.get_component(cname)
                comp_old._data = comp_new._data

        # Add components that didn't exist in original one. As above, we try
        # and preserve the order of components as much as possible.
        for cid in data.components:
            cname = cid.label
            if cname in new_labels - old_labels:
                cid = data.find_component_id(cname)
                comp_new = data.get_component(cname)
                self.add_component(comp_new, cid.label)

        # Update data label
        self.label = data.label

        # Update data coordinates
        self.coords = data.coords

        # alert hub of the change
        if self.hub is not None:
            msg = NumericalDataChangedMessage(self)
            self.hub.broadcast(msg)

        for subset in self.subsets:
            clear_cache(subset.subset_state.to_mask)


@contract(i=int, ndim=int)
def pixel_label(i, ndim):
    label = "{0}".format(i)
    if 1 <= ndim <= 3:
        label += " [{0}]".format('xyz'[ndim - 1 - i])
    return label
