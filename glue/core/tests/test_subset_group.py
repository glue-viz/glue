from mock import MagicMock, patch
import numpy as np

from .. import DataCollection, Data, SubsetGroup
from ..subset import SubsetState
from ..subset_group import coerce_subset_groups
from .test_state import clone


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
        assert sg.subsets[0].style.color == 'blue'

    def test_new_subset_group_syncs_style(self):
        sg = self.dc.new_subset_group()
        for s in sg.subsets:
            assert s.style == sg.style

    def test_set_group_style_clears_override(self):
        sg = self.dc.new_subset_group()
        style = sg.style.copy()
        style.parent = sg.subsets[0]
        sg.subsets[0].style = style
        style.color = 'blue'
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

        sg.style.color = '#123456'
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


class TestSerialze(TestSubsetGroup):

    def test_save_group(self):
        sg = self.dc.new_subset_group()
        sg2 = clone(sg)

        assert sg.style == sg2.style
        assert sg.label == sg2.label

    def test_save_subset(self):
        sg = self.dc.new_subset_group()
        sg.subset_state = self.dc[0].id['x'] > 1

        sub = sg.subsets[0]
        dc = clone(self.dc)

        sub2 = dc[0].subsets[0]

        np.testing.assert_array_equal(sub2.to_mask(), [False, True, True])
        assert sub2.style == sg.style
        assert sub2.label == sg.label

    def test_save_override(self):
        sg = self.dc.new_subset_group()
        sg.subsets[0].style.color = 'blue'

        dc = clone(self.dc)

        assert dc.subset_groups[0].style == sg.style
        assert dc.subset_groups[0].subsets[0].style.color == 'blue'


class TestCoerce(object):

    def setup_method(self, method):
        self.x = Data(label='x', x=[1, 2, 3])
        self.y = Data(label='y', y=[1, 2, 3])
        self.dc = DataCollection([self.x, self.y])

    def test_noop_on_good_setup(self):
        with patch('glue.core.subset_group.warn') as warn:
            coerce_subset_groups(self.dc)
        assert warn.call_count == 0

    def test_reassign_non_grouped_subsets(self):
        s = self.x.new_subset()
        dc = self.dc
        with patch('glue.core.subset_group.warn') as warn:
            coerce_subset_groups(dc)

        assert len(dc.subset_groups) == 1
        assert dc.subset_groups[0].subset_state is s.subset_state
        assert dc.subset_groups[0].style == s.style
        assert dc.subset_groups[0].label == s.label
        assert warn.call_count == 1
