from collections import defaultdict

from glue.component_link import ComponentLink
from glue.data import DerivedComponent

def accessible_links(cids, links):
    """ Calculate all ComponentLink objects in a list
    that can be calculated from a collection of componentIds

    Parameters
    ----------
    cids : Collection of ComponentID objects
    links : Iterable of ComponentLink objects

    Returns
    -------
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

    ------
    Data : Data object to discover new components for
    links : Set of ComponentLinks to use

    Output:
    -------
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

    Parameters
    ----------
    data : The data object to consider
    link_to_remove: The `ComponentLink` object to consider

    Returns
    -------
    A `set` of `DerivedComponent` IDs that cannot be
    calculated without the input `Link`
    """
    dependents = set()
    visited = set()
    while True:
        for derived in data.derived_components:
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
            break # nothing more to remove
    return dependents


class LinkManager(object):
    """A helper class to generate and store ComponentLinks,
    and compute which components are accesible from which data sets
    """
    def __init__(self):
        self._links = set()
        self._components = set()

    def add_link(self, link):
        self._links.add(link)

    def remove_link(self, link):
        raise NotImplemented

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

    def _remove_underiveable_components(self, data):
        """ Find and remove any DerivedComponent in the data
        which requires a ComponentLink not tracked by this LinkManager
        """
        data_links = set(dc.link for dc in data.derived_components)
        missing_links = data_links - self._links
        to_remove = []
        for m in missing_links:
            to_remove.extend(find_dependents(data, m))

        for r in to_remove:
            data.remove_component(r)

    def _add_deriveable_components(self, data):
        """Find and add any DerivedComponents that a data object can
        calculate given the ComponentLInks tracked by this
        LinkManager

        """
        links = discover_links(data, self._links)
        for cid, link in links.iteritems():
            d = DerivedComponent(data, link)
            data.add_component(d, cid)

    @property
    def links(self):
        return list(self._links)