from glue.config import link_helper
from glue.core.link_helpers import LinkCollection


@link_helper(category="Join")
class Join_Link(LinkCollection):
    cid_independent = False

    display = "Join on ID"
    description = "Join two datasets on a common indetifier or index. \
This is similar to a database join in that subsets defined \
on one dataset can always propogate through joins."

    labels1 = ["Identifier in dataset 1"]
    labels2 = ["Identifier in dataset 2"]

    def __init__(self, *args, cids1=None, cids2=None, data1=None, data2=None):
        # only support linking by one value now, even though link_by_value supports multiple
        assert len(cids1) == 1
        assert len(cids2) == 1

        self.data1 = data1
        self.data2 = data2
        self.cids1 = cids1
        self.cids2 = cids2

        self._links = []

    def __str__(self):
        return '%s >< %s' % (self.cids1, self.cids2)

    def __repr__(self):
        return "<Join_Link: %s>" % self

    # Define __eq__ and __ne__ to facilitate removing
    # these kinds of links from the link_manager
    def __eq__(self, other):
        same = ((self.data1 == other.data1) and
                (self.data2 == other.data2) and
                (self.cids1 == other.cids1) and
                (self.cids2 == other.cids2))
        flip = ((self.data1 == other.data2) and
                (self.data2 == other.data1) and
                (self.cids1 == other.cids2) and
                (self.cids2 == other.cids1))
        return same or flip

    def __ne__(self, other):
        same = ((self.data1 != other.data1) or
                (self.data2 != other.data2) or
                (self.cids1 != other.cids1) or
                (self.cids2 != other.cids2))
        flip = ((self.data1 != other.data2) or
                (self.data2 != other.data1) or
                (self.cids1 != other.cids2) or
                (self.cids2 != other.cids1))
        return same and flip
