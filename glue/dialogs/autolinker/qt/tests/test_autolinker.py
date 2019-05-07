import pytest

from glue.core import Data, DataCollection
from glue.core.link_helpers import identity
from glue.dialogs.autolinker.qt.autolinker import AutoLinkPreview
from glue.core.component_link import ComponentLink

# NOTE: the autolinker re-uses the main widget from the link editor so we don't
# need to test all the functionality - just things that are specific to the
# autolink preview.


class TestLinkEditor:

    def setup_method(self, method):

        self.data1 = Data(x=[1, 2, 3], y=[2, 3, 4], z=[6, 5, 4], label='data1')
        self.data2 = Data(a=[2, 3, 4], b=[4, 5, 4], c=[3, 4, 1], label='data2')
        self.data3 = Data(i=[5, 4, 3], j=[2, 2, 1], label='data3')

        self.data_collection = DataCollection([self.data1, self.data2, self.data3])

    @pytest.mark.parametrize('accept', [False, True])
    def test_basic(self, accept):

        # Set up an existing link
        link1 = ComponentLink([self.data1.id['x']], self.data2.id['c'])
        self.data_collection.add_link(link1)

        # Set up two suggested links

        def add(x, y):
            return x + y

        def double(x):
            return x * 2

        def halve(x):
            return x / 2

        link2 = ComponentLink([self.data2.id['a'], self.data2.id['b']], self.data3.id['j'], using=add)
        link3 = ComponentLink([self.data3.id['i']], self.data2.id['c'], using=double, inverse=halve)

        suggested_links = [link2, link3]

        dialog = AutoLinkPreview('test autolinker', self.data_collection, suggested_links)
        dialog.show()
        link_widget = dialog.link_widget

        link_widget.state.data1 = self.data1
        link_widget.state.data2 = self.data2

        assert link_widget.listsel_current_link.count() == 1

        link_widget.state.data1 = self.data3

        assert link_widget.listsel_current_link.count() == 2

        if accept:

            dialog.accept()

            links = self.data_collection.external_links

            assert len(links) == 3

            assert isinstance(links[0], ComponentLink)
            assert links[0].get_from_ids()[0] is self.data1.id['x']
            assert links[0].get_to_id() is self.data2.id['c']
            assert links[0].get_using() is identity

            assert isinstance(links[1], ComponentLink)
            assert links[1].get_from_ids()[0] is self.data2.id['a']
            assert links[1].get_from_ids()[1] is self.data2.id['b']
            assert links[1].get_to_id() is self.data3.id['j']
            assert links[1].get_using() is add

            assert isinstance(links[2], ComponentLink)
            assert links[2].get_from_ids()[0] is self.data3.id['i']
            assert links[2].get_to_id() is self.data2.id['c']
            assert links[2].get_using() is double
            assert links[2].get_inverse() is halve

        else:

            dialog.reject()

            links = self.data_collection.external_links

            assert len(links) == 1

            assert isinstance(links[0], ComponentLink)
            assert links[0].get_from_ids()[0] is self.data1.id['x']
            assert links[0].get_to_id() is self.data2.id['c']
            assert links[0].get_using() is identity
