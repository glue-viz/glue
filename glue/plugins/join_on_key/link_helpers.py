from glue.config import link_helper
from glue.core.link_helpers import LinkCollection
from glue.core.component_link import JoinLink


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

        data1.join_on_key(data2, cids1[0], cids2[0])

        self._links = [JoinLink()]
