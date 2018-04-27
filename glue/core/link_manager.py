"""
The LinkManager class is responsible for maintaining the conistency
of the "web of links" in a DataCollection. It discovers how to
combine ComponentLinks together to discover all of the ComponentIDs
that a Data object can derive,

As a trivial example, imagine a chain of 2 ComponentLinks linking
ComponentIDs across 3 datasets:

Data:        D1        D2       D3
ComponentID: x         y        z
Link:        <---x2y---><--y2z-->

The LinkManager autocreates a link from D1.id['x'] to D3.id['z']
by chaining x2y and y2z.
"""

from __future__ import absolute_import, division, print_function

import logging

from glue.external import six
from glue.core.hub import HubListener
from glue.core.message import DataCollectionDeleteMessage, DataRemoveComponentMessage
from glue.core.contracts import contract
from glue.core.link_helpers import LinkCollection
from glue.core.component_link import ComponentLink
from glue.core.data import Data
from glue.core.component import DerivedComponent
from glue.core.exceptions import IncompatibleAttribute
from glue.core.subset import Subset

__all__ = ['accessible_links', 'discover_links', 'find_dependents',
           'LinkManager', 'is_equivalent_cid']


def accessible_links(cids, links):
    """ Calculate all ComponentLink objects in a list
    that can be calculated from a collection of componentIds

    :param cids: Collection of ComponentID objects
    :param links: Iterable of ComponentLink objects

    :rtype: list
    A list of all links that can be evaluated
    given the input ComponentIDs
    """
    cids = set(cids)
    return [l for l in links if
            set(l.get_from_ids()) <= cids]


def discover_links(data, links):
    """
    Discover all links to components that can be derived
    based on the current components known to a dataset, and a set
    of ComponentLinks.

    :param Data: Data object to discover new components for
    :param links: Set of ComponentLinks to use

    :rtype: dict
    A dict of componentID -> componentLink
    The ComponentLink that data can use to generate the componentID.
    """

    # TODO: try to add shortest paths first -- should
    # prevent lots of repeated checking

    cids = set(data.primary_components)
    cid_links = {}
    depth = {}
    for cid in cids:
        depth[cid] = 0

    while True:
        for link in accessible_links(cids, links):
            from_ = set(link.get_from_ids())
            to_ = link.get_to_id()
            if len(from_) > 0:
                cost = max([depth[f] for f in from_]) + 1
            else:
                cost = 1
            if to_ in cids and cost >= depth[to_]:
                continue
            depth[to_] = cost
            cids.add(to_)
            cid_links[to_] = link
            break
        else:
            # no more links to add
            break
    return cid_links


def find_dependents(data, link):
    """ Determine which `DerivedComponents` in a data set
    depend (either directly or implicitly) on a given
    `ComponentLink`.

    :param data: The data object to consider
    :param link: The `ComponentLink` object to consider

    :rtype: set
    A `set` of `glue.core.component.DerivedComponent` IDs that cannot be
    calculated without the input `Link`
    """
    dependents = set()
    visited = set()
    while True:
        for derived in data.derived_components:
            derived = data.get_component(derived)
            if derived in visited:
                continue
            to_, from_ = derived.link.get_to_id(), derived.link.get_from_ids()
            if derived.link is link:
                dependents.add(to_)
                visited.add(derived)
                break
            if any(f in dependents for f in from_):
                dependents.add(to_)
                visited.add(derived)
                break
        else:
            break  # nothing more to remove
    return dependents


class LinkManager(HubListener):

    """A helper class to generate and store ComponentLinks,
    and compute which components are accesible from which data sets
    """

    def __init__(self, data_collection=None):
        self._external_links = set()
        self.hub = None
        self.trigger = False
        self.data_collection = data_collection

    def register_to_hub(self, hub):
        self.hub = hub
        self.hub.subscribe(self, DataRemoveComponentMessage,
                           handler=self._component_removed)
        self.hub.subscribe(self, DataCollectionDeleteMessage,
                           handler=self._data_removed)

    def _component_removed(self, msg):
        remove = []
        remove_cid = msg.component_id
        for link in self._external_links:
            ids = set(link.get_from_ids()) | set([link.get_to_id()])
            for cid in ids:
                if cid is remove_cid:
                    remove.append(link)
                    break
        for link in remove:
            self.remove_link(link)

    def _data_removed(self, msg):
        remove = []
        for link in self._external_links:
            ids = set(link.get_from_ids()) | set([link.get_to_id()])
            for cid in ids:
                if cid.parent is msg.data:
                    remove.append(link)
                    break
        for link in remove:
            self.remove_link(link)

    def clear_links(self):
        self._external_links.clear()

    def add_link(self, link, update_external=True):
        """
        Ingest one or more ComponentLinks to the manager

        Parameters
        ----------
        link : ComponentLink, LinkCollection, or list thereof
           The link(s) to ingest
        """
        if isinstance(link, (LinkCollection, list)):
            for l in link:
                self.add_link(l, update_external=False)
            if update_external:
                self.update_externally_derivable_components()
        else:
            if link.inverse not in self._external_links:
                self._external_links.add(link)
                if update_external:
                    self.update_externally_derivable_components()

    @contract(link=ComponentLink)
    def remove_link(self, link, update_external=True):
        if isinstance(link, (LinkCollection, list)):
            for l in link:
                self.remove_link(l, update_external=False)
            if update_external:
                self.update_externally_derivable_components()
        else:
            logging.getLogger(__name__).debug('removing link %s', link)
            self._external_links.remove(link)
            if update_external:
                self.update_externally_derivable_components()

    @contract(data=Data)
    def update_externally_derivable_components(self, data=None):
        """
        Update all the externally derived components in all data objects, based
        on all the Components deriveable based on the links in self.

        This overrides any ComponentLinks stored in the
        DerivedComponents of the data itself -- any components which
        depend on a link not tracked by the LinkManager will be
        deleted.

        Parameters
        -----------
        data : Data object

        Behavior
        --------
        DerivedComponents will be replaced / added into
        the data object
        """

        if self.data_collection is None:
            if data is None:
                return
            else:
                data_collection = [data]
        else:
            data_collection = self.data_collection

        for data in data_collection:
            links = discover_links(data, self._links | self._inverse_links)
            comps = {}
            for cid, link in six.iteritems(links):
                d = DerivedComponent(data, link)
                comps[cid] = d
            data._set_externally_derivable_components(comps)

        # Now update information about pixel-aligned data
        for data1 in data_collection:
            equivalent = {}
            for data2 in data_collection:
                if data1 is not data2:
                    order = equivalent_pixel_cids(data2, data1)
                    if order is not None:
                        equivalent[data2] = order
            data1._set_pixel_aligned_data(equivalent)

    @property
    def _links(self):
        if self.data_collection is None:
            data_links = set()
        else:
            data_links = set(link for data in self.data_collection for link in data.links)
        return data_links | self._external_links

    @property
    def _inverse_links(self):
        return set(link.inverse for link in self._links if link.inverse is not None)

    @property
    def links(self):
        return list(self._links)

    @property
    def external_links(self):
        return list(self._external_links)

    def clear(self):
        self._external_links.clear()

    def __contains__(self, item):
        return item in self._links


def _find_identical_reference_cid(data, cid):
    """
    Given a dataset and a component ID, return the equivalent component ID that
    truly belongs to the dataset (not via a link). Returns None if there is
    no strictly identical component in the dataset.
    """
    try:
        target_comp = data.get_component(cid)
    except IncompatibleAttribute:
        return None
    if isinstance(target_comp, DerivedComponent):
        if target_comp.link.identity:
            updated_cid = target_comp.link.get_from_ids()[0]
            return _find_identical_reference_cid(data, updated_cid)
        else:
            return None
    else:
        return cid


def is_equivalent_cid(data, cid1, cid2):
    """
    Convenience function to determine if two component IDs in a dataset are
    equivalent.

    Parameters
    ----------
    data : `~glue.core.Data`
        The data object in which to check for the component IDs
    cid1, cid2 : `~glue.core.ComponentID`
        The two component IDs to compare
    """

    # Dereference the component IDs to find base component ID
    cid1 = _find_identical_reference_cid(data, cid1)
    cid2 = _find_identical_reference_cid(data, cid2)

    return cid1 is cid2


def is_convertible_to_single_pixel_cid(data, cid):
    """
    Given a dataset and a component ID, determine whether a pixel component
    exists in data such that the component ID can be derived solely from the
    pixel component. Returns `None` if no such pixel component ID can be found
    and returns the pixel component ID if one exists.

    Parameters
    ----------
    data : `~glue.core.Data`
        The data in which to check for pixel components IDs
    cid : `~glue.core.ComponentID`
        The component ID to search for
    """
    if isinstance(data, Subset):
        data = data.data
    if cid in data.pixel_component_ids:
        return cid
    else:
        try:
            target_comp = data.get_component(cid)
        except IncompatibleAttribute:
            return None
        if cid in data.world_component_ids:
            if len(data.coords.dependent_axes(target_comp.axis)) == 1:
                return data.pixel_component_ids[target_comp.axis]
            else:
                return None
        else:
            if isinstance(target_comp, DerivedComponent):
                from_ids = [is_convertible_to_single_pixel_cid(data, c)
                            for c in target_comp.link.get_from_ids()]
                if None in from_ids:
                    return None
                else:
                    # Use set to get rid of duplicates
                    from_ids = list(set(from_ids))
                    if len(from_ids) == 1:
                        return is_convertible_to_single_pixel_cid(data, from_ids[0])


def equivalent_pixel_cids(reference, target):
    """
    Given two datasets with potentially linked pixel coordinates, determine
    whether the two datasets have an equivalent set of pixel component IDs.

    Note that the target can have fewer pixel components than the reference.

    This returns either the order of the reference pixel component IDs in the
    target dataset, or `None` if there is no match
    """

    # Shortcut - if target has more dimensions than the reference, we know
    # it's impossible for all pixel components in target to be contained in the
    # reference
    if target.ndim > reference.ndim:
        return

    # Shortcut, if target is reference, we can just return the axis order
    if target is reference:
        return list(range(reference.ndim))

    order = []
    for tar_cid in target.pixel_component_ids:
        for iref, ref_cid in enumerate(reference.pixel_component_ids):
            if is_equivalent_cid(reference, ref_cid, tar_cid):
                order.append(iref)
                break
        else:
            return None
    return order
