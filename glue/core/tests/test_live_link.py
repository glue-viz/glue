from mock import MagicMock
import pytest

from ..live_link import LiveLink, LiveLinkManager
from ..data import Data
from ..data_collection import DataCollection
from ..hub import Hub
from ..subset import Subset, SubsetState

class CopySubsetState(SubsetState):
    # a bit of a hack -- easier to use is than == when checking sync
    # live linker calls copy
    def copy(self):
        return self

def get_subset():
    result = Subset(None)
    result.subset_state = CopySubsetState()
    return result


class TestLiveLinkIntegrated(object):
    """ LiveLink integrates properly with Hub, intercepting messages """
    def setup_method(self, method):
        d1 = Data()
        d2 = Data()
        dc = DataCollection()
        dc.append(d1)
        dc.append(d2)
        link = LiveLink([d1.edit_subset, d2.edit_subset])
        self.hub = Hub(dc, d1, d2, link)
        self.s1 = d1.edit_subset
        self.s2 = d2.edit_subset

    def test_synced_on_change(self):
        """ Subset editing should trigger syncing """
        self.s1.style.color = "blue123"
        assert self.s2.style.color == "blue123"

def assert_synced(subsets):
    state = subsets[0].subset_state
    style = subsets[0].style
    for s in subsets:
        assert s.subset_state is state
        assert s.style is style

class TestLiveLInk(object):
    """ Sync method properly syncs both state and style """

    def run_test(self, subsets):
        self.link = LiveLink(subsets)
        self.link.sync(subsets[0])
        assert_synced(subsets)

    def test_single_subset(self):
        s1 = get_subset()
        self.run_test([s1])

    def test_two_subsets(self):
        s1 = get_subset()
        s2 = get_subset()
        self.run_test([s1, s2])

    def test_three_subsets(self):
        s1 = get_subset()
        s2 = get_subset()
        s3 = get_subset()
        self.run_test([s1, s2, s3])

    def test_repeated_subset(self):
        s1 = get_subset()
        s2 = get_subset()
        self.run_test([s1, s2])


class TestLiveLinkManager(object):
    def setup_method(self, method):
        data = Data()
        self.sub1 = data.new_subset()
        self.sub2 = data.new_subset()
        self.hub = MagicMock(spec_set=Hub)
        self.mgr = LiveLinkManager(self.hub)

    def test_add_link(self):
        self.mgr.add_link_between(self.sub1, self.sub2)
        links = self.mgr.links
        assert len(links) == 1
        assert self.sub1 in links[0].subsets
        assert self.sub2 in links[0].subsets
        assert self.hub.broadcast.call_count == 1

    def test_linked_on_init(self):
        self.mgr.add_link_between(self.sub1, self.sub2)
        assert self.sub1.style is self.sub2.style

    def test_remove_link(self):
        self.mgr.add_link_between(self.sub1, self.sub2)
        link = self.mgr.links[0]
        self.mgr.remove_links_from(self.sub1)
        assert len(self.mgr.links) == 0
        self.hub.unsubscribe_all.assert_called_once_with(link)
        assert self.hub.broadcast.call_count == 2

    def test_requires_hub(self):
        with pytest.raises(TypeError) as exc:
            LiveLinkManager().add_link_between(self.sub1, self.sub2)

        with pytest.raises(TypeError) as exc:
            LiveLinkManager().remove_links_from(self.sub1, self.sub2)

    def test_remove_not_found(self):
        """ Remove links shouldn't crash if it can't find a subset """
        self.mgr.remove_links_from(self.sub1)

