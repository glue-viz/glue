from mock import MagicMock

from .. import DataCollection, Data, SubsetGroup
from ..subset import SubsetState


class TestSubsetGroup(object):

    def setup_method(self, method):
        x = Data(label='x', x=[1, 2, 3])
        y = Data(label='y', y=[2, 4, 8])
        self.dc = DataCollection([x, y])

    def test_creation(self):
        sg = SubsetGroup(self.dc)
        for subset, data in zip(sg.subsets, self.dc):
            assert subset is data.subsets[0]

    def test_attributes_matched_to_group(self):
        sg = SubsetGroup(self.dc)
        for subset in sg.subsets:
            assert subset.subset_state is sg.subset_state
            assert subset.label is sg.label

    def test_attributes_synced_to_group(self):
        sg = SubsetGroup(self.dc)
        sg.subsets[0].subset_state = SubsetState()
        sg.subsets[0].label = 'testing'
        sg.subsets[1].style.color = 'blue'
        for subset in sg.subsets:
            assert subset.subset_state is sg.subset_state
            assert subset.label is sg.label
            assert subset.style.color == 'blue'

    def test_new_data_creates_subset(self):
        sg = self.dc.new_subset_group()
        d = Data(label='z', z=[10, 20, 30])
        self.dc.append(d)
        assert d.subsets[0] in sg.subsets

    def test_remove_data_deletes_subset(self):
        sg = self.dc.new_subset_group()
        sub = self.dc[0].subsets[0]
        self.dc.remove(self.dc[0])
        print sg.subsets, sub, sub in sg.subsets
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
        self.dc.remove_subset_group(sg)
        assert len(self.dc[0].subsets) == 0

    def test_edit_broadcasts(self):
        sg = self.dc.new_subset_group()
        bcast = MagicMock()
        sg.subsets[0].broadcast = bcast
        sg.subsets[0].style.color = 'red'
        assert bcast.call_count == 1

    def test_style_override(self):
        sg = self.dc.new_subset_group()
        sg.subsets[0].override_style('color', 'black')
        assert sg.subsets[0].style.color == 'black'
        assert sg.subsets[1].style.color != 'black'
        sg.subsets[0].clear_override_style()
        assert sg.subsets[0].style is sg.subsets[1].style
