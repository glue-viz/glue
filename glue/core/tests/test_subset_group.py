import numpy as np
from unittest.mock import MagicMock, patch

from ...config import settings
from .. import DataCollection, Data, SubsetGroup
from .. import subset
from ..subset import SubsetState
from ..subset_group import coerce_subset_groups
from .test_state import clone

from ...utils.matplotlib import MATPLOTLIB_GE_36


def restore_settings(func):
    def wrapper(*args, **kwargs):
        settings.reset_defaults()
        results = func(*args, **kwargs)
        settings.reset_defaults()
        return results
    return wrapper


class TestSubsetGroup(object):

    def setup_method(self, method):
        x = Data(label='x', x=[1, 2, 3])
        y = Data(label='y', y=[2, 4, 8])
        self.dc = DataCollection([x, y])
        self.sg = SubsetGroup()

    def test_creation(self):
        self.sg.register(self.dc)
        sg = self.sg
        for sub, data in zip(sg.subsets, self.dc):
            assert sub is data.subsets[0]

    def test_attributes_matched_to_group(self):
        self.sg.register(self.dc)
        sg = self.sg
        for sub in sg.subsets:
            assert sub.subset_state is sg.subset_state
            assert sub.label is sg.label

    def test_attributes_synced_to_group(self):
        self.sg.register(self.dc)
        sg = self.sg
        sg.subsets[0].subset_state = SubsetState()
        sg.subsets[0].label = 'testing'
        for sub in sg.subsets:
            assert sub.subset_state is sg.subset_state
            assert sub.label is sg.label

    def test_new_subset_group_syncs_style(self):
        sg = self.dc.new_subset_group()
        for sub in sg.subsets:
            assert sub.style == sg.style

    def test_changing_subset_style_changes_group(self):

        # Test to make sure that if a subset's visual properties are changed,
        # the visual properties of all subsets in the same subset group are changed

        d1 = Data(x=[1, 2, 3], label='d1')
        d2 = Data(y=[2, 3, 4], label='d2')
        d3 = Data(y=[2, 3, 4], label='d3')

        dc = DataCollection([d1, d2, d3])

        sg = dc.new_subset_group(subset_state=d1.id['x'] > 1, label='A')

        # Changing d1 subset properties changes group and other subsets

        d1.subsets[0].style.color = '#c0b4a1'
        assert sg.style.color == '#c0b4a1'
        assert d2.subsets[0].style.color == '#c0b4a1'
        assert d3.subsets[0].style.color == '#c0b4a1'

        d2.subsets[0].style.alpha = 0.2
        assert sg.style.alpha == 0.2
        assert d1.subsets[0].style.alpha == 0.2
        assert d3.subsets[0].style.alpha == 0.2

        d3.subsets[0].style.markersize = 16
        assert sg.style.markersize == 16
        assert d1.subsets[0].style.markersize == 16
        assert d2.subsets[0].style.markersize == 16

        # Changing subset group changes subsets

        sg.style.color = '#abcdef'
        assert d1.subsets[0].style.color == '#abcdef'
        assert d2.subsets[0].style.color == '#abcdef'
        assert d3.subsets[0].style.color == '#abcdef'

        sg.style.linewidth = 12
        assert d1.subsets[0].style.linewidth == 12
        assert d2.subsets[0].style.linewidth == 12
        assert d3.subsets[0].style.linewidth == 12

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

    def test_new_label(self):
        sg = self.dc.new_subset_group(label='test')
        assert sg.label == 'test'

    def test_new_state(self):
        state = SubsetState()
        sg = self.dc.new_subset_group(subset_state=state)
        assert sg.subset_state is state

    def test_deleted_subsets_dont_respawn(self):
        # regression test
        sg1 = self.dc.new_subset_group()
        self.dc.remove_subset_group(sg1)
        d = Data(label='z', z=[1, 2, 3])
        self.dc.append(d)
        assert len(d.subsets) == 0

    def test_visual_attributes_default(self):
        num_sg = len(self.dc.subset_groups)
        sg = self.dc.new_subset_group()
        expected_color = settings.SUBSET_COLORS[num_sg % len(settings.SUBSET_COLORS)]

        # VisualAttributes stores color with lowercase letters
        # but the values in settings.SUBSET_COLORS use uppercase
        expected_color = expected_color.lower()
        assert sg.style.color == expected_color
        assert sg.style.alpha == 0.5
        assert sg.style.linewidth == 2.5
        assert sg.style.markersize == 7
        assert sg.style.linestyle == 'solid'
        assert sg.style.marker == 'o'
        assert sg.style.preferred_cmap is None

    def test_visual_attributes(self):
        visual_attributes = dict(
            color="#ff7f00",
            alpha=0.3,
            linewidth=4,
            markersize=10,
            marker='x',
            linestyle='dashed',
            preferred_cmap='viridis'
        )
        sg = self.dc.new_subset_group(**visual_attributes)
        for attr, value in visual_attributes.items():
            if attr == 'preferred_cmap':
                if MATPLOTLIB_GE_36:
                    from matplotlib import colormaps
                    va_cmap = colormaps[visual_attributes[attr]]
                else:
                    from matplotlib.cm import get_cmap
                    va_cmap = get_cmap(visual_attributes[attr])
                assert sg.style.preferred_cmap == va_cmap
            else:
                assert getattr(sg.style, attr, None) == value


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


class TestCombination(object):

    def check_type_and_children(self, s1, s2, s3, statetype):
        assert isinstance(s3, statetype)
        assert s3.state1 is s1.subset_state
        assert s3.state2 is s2.subset_state

    def test_and(self):
        s1, s2 = SubsetGroup(), SubsetGroup()
        assert isinstance(s1 & s2, subset.AndState)

    def test_or(self):
        s1, s2 = SubsetGroup(), SubsetGroup()
        assert isinstance(s1 | s2, subset.OrState)

    def test_xor(self):
        s1, s2 = SubsetGroup(), SubsetGroup()
        assert isinstance(s1 ^ s2, subset.XorState)

    def test_invert(self):
        s1 = SubsetGroup()
        assert isinstance(~s1, subset.InvertState)


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
