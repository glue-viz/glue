from ..live_link import LiveLink
from ..data import Data
from ..data_collection import DataCollection
from ..hub import Hub
from ..subset import Subset

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
        self.s1.style.color = "blue"
        assert self.s2.style.color == "blue"

class TestLiveLInk(object):
    def setup_method(self, method):
        self.s1 = Subset(None)
        self.s2 = Subset(None)
        self.link = LiveLink([self.s1, self.s2])

    def test_sync_style(self):
        """ Syncing should sync styles"""
        self.link.sync(self.s1)
        assert self.s2.style is self.s1.style

    def test_sync_state(self):
        """ Syncing should sync state """
        self.link.sync(self.s2)
        assert self.s1.subset_state is self.s2.subset_state
