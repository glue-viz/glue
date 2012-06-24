import unittest

from glue.live_link import LiveLink
import glue

class TestLiveLinkIntegrated(unittest.TestCase):
    """ LiveLink integrates properly with Hub, intercepting messages """
    def setUp(self):
        d1 = glue.Data()
        d2 = glue.Data()
        dc = glue.DataCollection()
        dc.append(d1)
        dc.append(d2)
        link = LiveLink([d1.edit_subset, d2.edit_subset])
        hub = glue.Hub(dc, d1, d2, link)
        self.s1 = d1.edit_subset
        self.s2 = d2.edit_subset

    def test_synced_on_change(self):
        """ Subset editing should trigger syncing """
        self.s1.style.color = "blue"
        self.assertIs(self.s2.style.color, "blue")

class TestLiveLInk(unittest.TestCase):
    def setUp(self):
        self.s1 = glue.Subset(None)
        self.s2 = glue.Subset(None)
        self.link = LiveLink([self.s1, self.s2])

    def test_sync_style(self):
        """ Syncing should sync styles"""
        self.link.sync(self.s1)
        self.assertIs(self.s2.style, self.s1.style)

    def test_sync_state(self):
        """ Syncing should sync state """
        self.link.sync(self.s2)
        self.assertIs(self.s1.subset_state, self.s2.subset_state)


if __name__ == "__main__":
    unittest.main()