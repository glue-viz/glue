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
from glue.core.message import DataCollectionDeleteMessage
from glue.core.contracts import contract
from glue.core.link_helpers import LinkCollection
from glue.core.component_link import ComponentLink
from glue.core.data import Data
from glue.core.component import DerivedComponent
from glue.core.exceptions import IncompatibleAttribute


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
    """ Discover all links to components that can be derived
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
            cost = max([depth[f] for f in from_]) + 1
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

    def __init__(self):
        self._links = set()
        self.hub = None

    def register_to_hub(self, hub):
        self.hub = hub
        self.hub.subscribe(self, DataCollectionDeleteMessage,
                           handler=self._data_removed)

    def _data_removed(self, msg):
        remove = []
        for link in self._links:
            ids = set(link.get_from_ids()) | set([link.get_to_id()])
            for cid in ids:
                if cid.parent is msg.data:
                    remove.append(link)
                    break
        for link in remove:
            self.remove_link(link)

    def add_link(self, link):
        """
        Ingest one or more ComponentLinks to the manager

        Parameters
        ----------
        link : ComponentLink, LinkCollection, or list thereof
           The link(s) to ingest
        """
        if isinstance(link, (LinkCollection, list)):
            for l in link:
                self.add_link(l)
        else:
            if link.inverse not in self._links:
                self._links.add(link)

    @contract(link=ComponentLink)
    def remove_link(self, link):
        if isinstance(link, (LinkCollection, list)):
            for l in link:
                self.remove_link(l)
        else:
            logging.getLogger(__name__).debug('removing link %s', link)
            self._links.remove(link)

    @contract(data=Data)
    def update_data_components(self, data):
        """Update all the DerivedComponents in a data object, based on
        all the Components deriveable based on the links in self.

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
        self._remove_underiveable_components(data)
        self._add_deriveable_components(data)

    @property
    def _inverse_links(self):
        return set(link.inverse for link in self._links if link.inverse is not None)

    def _remove_underiveable_components(self, data):
        """ Find and remove any DerivedComponent in the data
        which requires a ComponentLink not tracked by this LinkManager
        """
        data_links = set(data.get_component(dc).link
                         for dc in data.derived_components)
        missing_links = data_links - self._links - self._inverse_links
        to_remove = []
        for m in missing_links:
            to_remove.extend(find_dependents(data, m))

        if getattr(data, 'hub', None) is None:
            for r in to_remove:
                data.remove_component(r)
        else:
            with data.hub.delay_callbacks():
                for r in to_remove:
                    data.remove_component(r)

    def _add_deriveable_components(self, data):
        """Find and add any DerivedComponents that a data object can
        calculate given the ComponentLinks tracked by this
        LinkManager

        """
        links = discover_links(data, self._links | self._inverse_links)
        for cid, link in six.iteritems(links):
            d = DerivedComponent(data, link)
            if cid not in data.components:
                data.add_component(d, cid)

    @property
    def links(self):
        return list(self._links)

    def clear(self):
        self._links.clear()

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
