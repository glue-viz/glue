from collections import defaultdict

from glue.component_link import ComponentLink


def accessible_links(cids, links):
    """ Yield each link in a collection of links
    whose from_components are all contained in a set of cids
    """
    for link in links:
        from_ = set(link.get_from_ids())
        if from_ <= cids:
            yield link


def virtual_components(data, links):
    """ Finds and returns component_links for
    all components derivable from the primary components
    of a given data set.


    Input:
    ------
    Data : Data object
    links : Set of ComponentLinks

    Output:
    -------
    A dict keyed by component_ids. The value
    of each entry is a ComponentLink that can be
    used by data to calculate the to_component from the link
    """

    # try to add shortest paths first -- should
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


class LinkManager(object):
    """A helper class to generate ComponentLinks, and compute which
    components are accesible from which data sets
    """
    def __init__(self):
        self._links = set()
        self._components = set()

    def make_link(self, from_ids, to_id, using=None):
        for f in from_ids:
            self._components.add(f)
        self._components.add(to_id)
        result = ComponentLink(from_ids, to_id, using)
        self._links.add(result)
        return result

    def virtual_components(self, data):
        return virtual_components(data, self._links)

    @property
    def links(self):
        return list(self._links)