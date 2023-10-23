import numpy as np
import shapely

from glue.core.data import Data
from glue.core.contracts import contract

from glue.core.component import Component, ExtendedComponent


__all__ = ['RegionData']


class RegionData(Data):
    """
    A glue Data object for storing data that is associated with a region.

    This object can be used when a dataset describes 2D regions or 1D ranges. It
    contains exactly one :class:`~glue.core.component.ExtendedComponent` object
    which contains the boundaries of the regions, and must also contain
    one or two components that give the center of the regions in whatever data
    coordinates the regions are described in. Links in glue are not made
    directly on the :class:`~glue.core.component.ExtendedComponent`, but instead
    on the center components.

    Thus, a subset that includes the center of a region will include that region,
    but a subset that includes just a little part of the region will not include
    that region. These center components are not the same pixel components. For
    example, a dataset that is a table of 2D regions will have a single
    :class:`~glue.core.component.CoordinateComponent`, but must have two of these center
    components.

    A typical use case for this object is to store the properties of geographic
    regions, where the boundaries of the regions are stored in an
    :class:`~glue.core.component.ExtendedComponent` and the centers of the
    regions are stored in two components, one for the longitude and one for the
    latitude. Additional components may describe arbitrary properties of these
    geographic regions (e.g. population, area, etc).

    This class is mostly a convenience class. By using this class, Data
    Loaders can create RegionData directly from an iterable of geometries,
    since this class deals with creating representative points. Viewers
    can assume that when adding a RegionData object they are probably
    being asked to visualize the ExtendedComponent, and this class provides
    convenience methods for ascertaining whether the components currently
    visualized in a Viewer are the correct ones to enable display of the
    ExtendedComponent and to transform the regions through the glue
    linking architecture into the correct coordinates for display.

    Parameters
    ----------
    label : `str`, optional
        The label of the data.
    coords : :class:`~glue.core.coordinates.Coordinates`, optional
        The coordinates associated with the data.
    **kwargs
        All other keyword arguments are passed to the :class:`~glue.core.data.Data`
        constructor.

    Attributes
    ----------
    extended_component_id : :class:`~glue.core.component_id.ComponentID`
        The ID of the :class:`~glue.core.component.ExtendedComponent` object
        that contains the boundaries of the regions.
    center_x_id : :class:`~glue.core.component_id.ComponentID`
        The ID of the Component object that contains the x-coordinate of the
        center of the regions. This is actually stored in the component
        with the extended_component_id, but it is convenient to have it here.
    center_y_id : :class:`~glue.core.component_id.ComponentID`
        The ID of the Component object that contains the y-coordinate of the
        center of the regions. This is actually stored in the component
        with the extended_component_id, but it is convenient to have it here.

    Examples
    --------

    There are two main options for initializing a :class:`~glue.core.data_region.RegionData`
    object. The first is to simply pass in a list of ``Shapely.Geometry`` objects
    with dimesionality N, from which we will create N+1 components: one
    :class:`~glue.core.component.ExtendedComponent` with the boundaries, and N
    regular Component(s) with the center coordinates computed from the Shapley
    method :meth:`~shapely.GeometryCollection.representative_point`:

        >>> geometries = [shapely.geometry.Point(0, 0).buffer(1), shapely.geometry.Point(1, 1).buffer(1)]
        >>> my_region_data = RegionData(label='My Regions', boundary=geometries)

    This will create a :class:`~glue.core.data_region.RegionData` object with three
    components: one :class:`~glue.core.component.ExtendedComponent` with label
    "geo" and two regular Components with labels "Center [x] for boundary"
    and "Center [y] for boundary".

    The second is to explicitly create an :class:`~glue.core.component.ExtendedComponent`
    (which requires passing in the ComponentIDs for the center coordinates) and
    then use `add_component` to add this component to a :class:`~glue.core.data_region.RegionData`
    object. You might use this approach if your dataset already contains points that
    represent the centers of your regions and you want to avoid re-calculating them. For example:

        >>> center_x = [0, 1]
        >>> center_y = [0, 1]
        >>> geometries = [shapely.geometry.Point(0, 0).buffer(1), shapely.geometry.Point(1, 1).buffer(1)]

        >>> my_region_data = RegionData(label='My Regions')
        >>> # Region IDs are created and returned when we add a Component to a Data object
        >>> cen_x_id = my_region_data.add_component(center_x, label='Center [x]')
        >>> cen_y_id = my_region_data.add_component(center_y, label='Center [y]')
        >>> extended_comp = ExtendedComponent(geometries, center_comp_ids=[cen_x_id, cen_y_id])
        >>> my_region_data.add_component(extended_comp, label='boundaries')

    """

    def __init__(self, label="", coords=None, **kwargs):
        self._extended_component_id = None
        # __init__ calls add_component which deals with ExtendedComponent logic
        super().__init__(label=label, coords=coords, **kwargs)

    def __repr__(self):
        return f'RegionData (label: {self.label} | extended_component: {self.extended_component_id})'

    @property
    def center_x_id(self):
        return self.get_component(self.extended_component_id).x

    @property
    def center_y_id(self):
        return self.get_component(self.extended_component_id).y

    @property
    def extended_component_id(self):
        return self._extended_component_id

    @contract(component='component_like', label='cid_like')
    def add_component(self, component, label):
        """ Add a new component to this data set, allowing only one :class:`~glue.core.component.ExtendedComponent`

        If component is an array of Shapely objects then we use
        :meth:`~shapely.GeometryCollection.representative_point`: to
        create two new components for the center coordinates of the regions and
        add them to the :class:`~glue.core.data_region.RegionData` object as well.

        If component is an :class:`~glue.core.component.ExtendedComponent`,
        then we simply add it to the :class:`~glue.core.data_region.RegionData` object.

        We do this here instead of extending ``Component.autotyped`` because
        we only want to use :class:`~glue.core.component.ExtendedComponent` objects
        in the context of a :class:`~glue.core.data_region.RegionData` object.

        Parameters
        ----------
        component : :class:`~glue.core.component.Component` or array-like
            Object to add. If this is an array of Shapely objects, then we
            create two new components for the center coordinates of the regions
            as well.
        label : `str` or :class:`~glue.core.component_id.ComponentID`
              The label. If this is a string, a new
              :class:`glue.core.component_id.ComponentID`
              with this label will be created and associated with the Component.

        Raises
        ------
           `ValueError`, if the :class:`~glue.core.data_region.RegionData` already has an extended component
        """

        if not isinstance(component, Component):
            if all(isinstance(s, shapely.Geometry) for s in component):
                center_x = []
                center_y = []
                for s in component:
                    rep = s.representative_point()
                    center_x.append(rep.x)
                    center_y.append(rep.y)
                cen_x_id = super().add_component(np.asarray(center_x), f"Center [x] for {label}")
                cen_y_id = super().add_component(np.asarray(center_y), f"Center [y] for {label}")
                ext_component = ExtendedComponent(np.asarray(component), center_comp_ids=[cen_x_id, cen_y_id])
                self._extended_component_id = super().add_component(ext_component, label)
                return self._extended_component_id

        if isinstance(component, ExtendedComponent):
            if self.extended_component_id is not None:
                raise ValueError(f"Cannot add another ExtendedComponent; existing extended component: {self.extended_component_id}")
            else:
                self._extended_component_id = super().add_component(component, label)
                return self._extended_component_id
        else:
            return super().add_component(component, label)

    def _get_trans_to_cids(self, cen_cids, other_cids):
        """
        Use recursion to traverse links and build up a list of functions
        to convert cen_ids to other_cids.

        Parameters
        ----------
        cen_cids : list of :class:`~glue.core.component_id.ComponentID`
            The ComponentIDs that are the inputs to the transformation
            function.
        other_cids : list of :class:`~glue.core.component_id.ComponentID`
            The ComponentIDs that are the outputs of the transformation
            function.

        Raises
        ------
        ValueError
            If the links imply a transformation that cannot be done by Shapely.
        """
        if len(other_cids) != 2:
            raise ValueError("Can only deal with 2D transformations")
        linkx = self._get_external_link(other_cids[0])
        linky = self._get_external_link(other_cids[1])

        funcx = linkx.get_using()
        funcy = linky.get_using()

        if len(linkx.get_from_ids()) > 2 or len(linky.get_from_ids()) > 2:
            raise ValueError("Can only display regions if links depend on 2 or fewer other components.")

        def conv_function(x, y=None):
            if len(linkx.get_from_ids()) == 1 and len(linky.get_from_ids()) == 1:
                return [funcx(x), funcy(y)]
            else:
                return [funcx(x, y), funcy(x, y)]

        self.list_of_functions.append(conv_function)
        if len(linkx.get_from_ids()) == 2:
            other_cids = linkx.get_from_ids()
        else:
            other_cids = linkx.get_from_ids() + linky.get_from_ids()
        if cen_cids[0] in other_cids or cen_cids[1] in other_cids:
            if set([cen_cids[0]]+[cen_cids[1]]) == set(other_cids):
                return
            else:
                raise ValueError("Cannot display regions if links depend on other components.")
        else:
            self._get_trans_to_cids(cen_cids, other_cids)

    def get_transform_to_cids(self, other_cids):
        """
        Return the function that converts the center components to other_cids.

        We can use this to get the transformation from the x,y coordinates
        that the ExtendedComponent are in to x and y attributes that are
        visualized in a Viewer so that we can translate the geometries
        in the ExtendedComponent to the new coordinates before displaying them.

        Can be called in viewers as:

            >>> tfunc = region_data.get_transform_to_cids([viewer_x_att, viewer_y_att])

        And the function can be used to transform the geometries as:

            >>> from shapely.ops import transform
            >>> new_geoms = [transform(tfunc, g) for g in old_geoms]

        TODO: This is currently hard-coded to work with 2D transformations,
              but could be extended to work with 1D viewers as well. Our region
              geometries are limited to be 2D (or 1D ranges), so links
              that require more than 2 input components do not admit to
              a valid transformation.

        Parameters
        ----------
        other_cid : list of :class:`~glue.core.component.ComponentID`
            The other ComponentIDs (typically the ones that are
            visualized in a Viewer).

        Returns
        -------
        func : `callable`
            The function that converts center_x_id and center_y_id to
            other_cids which can then be used to transform the
            geometries before display. Returns None if there is no
            such valid transformation.
        """

        self.list_of_functions = []
        self._get_trans_to_cids([self.center_x_id, self.center_y_id], other_cids)
        if not self.list_of_functions:
            return None
        elif len(self.list_of_functions) == 1:
            return self.list_of_functions[0]
        else:
            def conv_function(*args):
                # Our list of functions is built up in reverse order
                for f in self.list_of_functions[::-1]:
                    args = f(*args)
                return args
            return conv_function

    def linked_to_center_comp(self, target_cid):
        """
        Check if target_cid can be mapped to one of the center components.

        This is used to see if we can display the ExtendedComponent in a Viewer.

        It is not sufficient to simply see if we can retrieve data from
        target_cid like is commonly done in Viewers:

                >>> _ = self[target_cid]

        Because if target_cid is mapped to a Component that is not one of the
        center components, then we cannot display the regions.

        Parameters
        ----------
        target_cid : :class:`~glue.core.component.ComponentID`
            The ComponentID (typically displayed in a Viewer) which we
            want to check if it is one of the special center components.

        Returns
        -------
        bool
            True if target_cid can be mapped to one of the center components, False otherwise.
        """
        from glue.core.link_manager import is_equivalent_cid  # avoid circular import

        center_cids = [self.center_x_id, self.center_y_id]
        for center_cid in center_cids:
            if is_equivalent_cid(self, center_cid, target_cid):
                return True

        link = self._get_external_link(target_cid)
        if not link:
            return False
        for center_cid in center_cids:
            if center_cid in link:
                return True
            else:
                return any([self.linked_to_center_comp(x) for x in link.get_from_ids()])
