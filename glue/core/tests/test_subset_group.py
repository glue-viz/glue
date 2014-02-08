from mock import MagicMock

from .. import DataCollection, Data, SubsetGroup
from ..subset import SubsetState


class TestSubsetGroup(object):

    def setup_method(self, method):
        x = Data(label='x', x=[1, 2, 3])
        y = Data(label='y', y=[2, 4, 8])
        self.dc = DataCollection([x, y])
        self.sg = SubsetGroup()

    def test_creation(self):
        self.sg.register(self.dc)
        sg = self.sg
        for subset, data in zip(sg.subsets, self.dc):
            assert subset is data.subsets[0]

    def test_attributes_matched_to_group(self):
        self.sg.register(self.dc)
        sg = self.sg
        for subset in sg.subsets:
            assert subset.subset_state is sg.subset_state
            assert subset.label is sg.label

    def test_attributes_synced_to_group(self):
        self.sg.register(self.dc)
        sg = self.sg
        sg.subsets[0].subset_state = SubsetState()
        sg.subsets[0].label = 'testing'
        for subset in sg.subsets:
            assert subset.subset_state is sg.subset_state
            assert subset.label is sg.label

    def test_set_style_overrides(self):
        self.sg.register(self.dc)
        sg = self.sg
        sg.subsets[0].style.color = 'blue'
        for s in sg.subsets[1:]:
            assert s.style.color != 'blue'

    def test_set_group_style_clears_override(self):
        sg = self.dc.new_subset_group()
        sg.subsets[0].style.color = 'blue'
        sg.style.color = 'red'
        assert sg.subsets[0].style.color == 'red'

    def test_new_data_creates_subset(self):
        sg = self.dc.new_subset_group()
        d = Data(label='z', z=[10, 20, 30])
        self.dc.append(d)
        assert d.subsets[0] in sg.subsets

    def test_remove_data_deletes_subset(self):
        sg = self.dc.new_subset_group()
        sub = self.dc[0].subsets[0]
        self.dc.remove(self.dc[0])
        assert sub not in sg.subsets

    def test_subsets_given_data_reference(self):
        sg = self.dc.new_subset_group()
        assert sg.subsets[0].data is self.dc[0]

    def test_data_collection_subset(self):
        sg = self.dc.new_subset_group()
        assert tuple(self.dc.subset_groups) == (sg,)
        sg2 = self.dc.new_subset_group()
        assert tuple(self.dc.subset_groups) == (sg, sg2)

    def test_remove_subset(self):
        sg = self.dc.new_subset_group()
        n = len(self.dc[0].subsets)
        self.dc.remove_subset_group(sg)
        assert len(self.dc[0].subsets) == n - 1

    def test_edit_broadcasts(self):
        sg = self.dc.new_subset_group()
        bcast = MagicMock()
        sg.subsets[0].broadcast = bcast
        bcast.reset_mock()
        sg.subsets[0].style.color = 'red'
        assert bcast.call_count == 1

    def test_braodcast(self):
        sg = self.dc.new_subset_group()
        bcast = MagicMock()
        sg.subsets[0].broadcast = bcast
        bcast.reset_mock()

        sg.subset_state = SubsetState()
        assert bcast.call_count == 1

        sg.style.color = 'red'
        assert bcast.call_count == 2

        sg.label = 'new label'
        assert bcast.call_count == 3

    def test_auto_labeled(self):
        sg = self.dc.new_subset_group()
        assert sg.label is not None

    def test_label_color_cycle(self):
        sg1 = self.dc.new_subset_group()
        sg2 = self.dc.new_subset_group()

        assert sg1.label != sg2.label
        assert sg1.style.color != sg2.style.color
