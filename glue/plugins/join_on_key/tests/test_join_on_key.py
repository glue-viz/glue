from glue.core import Data, DataCollection
from glue.dialogs.link_editor.qt import LinkEditor
from glue.core.exceptions import IncompatibleAttribute
from glue.utils.qt import process_events

from glue.plugins.join_on_key.link_helpers import Join_Link

from qtpy import QtWidgets
from numpy.testing import assert_array_equal
import pytest


def test_remove_and_add_again():
    d1 = Data(x=[1, 2, 3, 4, 5], k1=[0, 0, 1, 1, 2], label='d1')
    d2 = Data(y=[2, 4, 5, 8, 4], k2=[1, 3, 1, 2, 3], label='d2')
    dc = DataCollection([d1, d2])

    mylink = Join_Link(cids1=[d1.id['k1']], cids2=[d2.id['k2']], data1=d1, data2=d2)
    dc.add_link(mylink)

    dc.remove_link(mylink)
    s = d1.new_subset()
    s.subset_state = d1.id['x'] > 2
    assert_array_equal(s.to_mask(), [False, False, True, True, True])
    s = d2.new_subset()
    s.subset_state = d1.id['x'] > 2
    with pytest.raises(IncompatibleAttribute):
        assert_array_equal(s.to_mask(), [True, False, True, True, False])
    mylink = Join_Link(cids1=[d1.id['k1']], cids2=[d2.id['k2']], data1=d1, data2=d2)
    dc.add_link(mylink)
    s = d2.new_subset()
    s.subset_state = d1.id['x'] > 2
    assert_array_equal(s.to_mask(), [True, False, True, True, False])


def test_remove_is_clean():
    d1 = Data(x=[1, 2, 3, 4, 5], k1=[0, 0, 1, 1, 2], label='d1')
    d2 = Data(y=[2, 4, 5, 8, 4], k2=[1, 3, 1, 2, 3], label='d2')
    dc = DataCollection([d1, d2])

    mylink = Join_Link(cids1=[d1.id['k1']], cids2=[d2.id['k2']], data1=d1, data2=d2)
    dc.add_link(mylink)

    dc.remove_link(mylink)
    s = d1.new_subset()
    s.subset_state = d1.id['x'] > 2
    assert_array_equal(s.to_mask(), [False, False, True, True, True])
    s = d2.new_subset()
    s.subset_state = d1.id['x'] > 2
    with pytest.raises(IncompatibleAttribute):
        assert_array_equal(s.to_mask(), [True, False, True, True, False])


def test_remove():
    d1 = Data(x=[1, 2, 3, 4, 5], k1=[0, 0, 1, 1, 2], label='d1')
    d2 = Data(y=[2, 4, 5, 8, 4], k2=[1, 3, 1, 2, 3], label='d2')
    dc = DataCollection([d1, d2])

    assert len(dc._link_manager._external_links) == 0
    assert len(dc.links) == 0
    assert d1._key_joins == {}
    assert d2._key_joins == {}

    mylink = Join_Link(cids1=[d1.id['k1']], cids2=[d2.id['k2']], data1=d1, data2=d2)
    dc.add_link(mylink)
    assert len(dc._link_manager._external_links) == 1  # The link manager tracks all links
    assert len(dc.links) == 0  # dc.links just keeps component links so joins do not show up here
    dc.remove_link(mylink)
    assert len(dc._link_manager._external_links) == 0
    assert len(dc.links) == 0

    assert d1._key_joins == {}
    assert d2._key_joins == {}


def test_using_link_index():
    d1 = Data(x=[1, 2, 3, 4, 5], k1=[0, 0, 1, 1, 2], label='d1')
    d2 = Data(y=[2, 4, 5, 8, 4], k2=[1, 3, 1, 2, 3], label='d2')
    dc = DataCollection([d1, d2])

    assert len(dc._link_manager._external_links) == 0
    assert len(dc.links) == 0
    dc.add_link(Join_Link(cids1=[d1.id['k1']], cids2=[d2.id['k2']], data1=d1, data2=d2))
    assert len(dc.links) == 0
    assert len(dc._link_manager._external_links) == 1

    s = d1.new_subset()
    s.subset_state = d1.id['x'] > 2
    assert_array_equal(s.to_mask(), [False, False, True, True, True])
    s = d2.new_subset()
    s.subset_state = d1.id['x'] > 2
    assert_array_equal(s.to_mask(), [True, False, True, True, False])


def test_basic_join_on_key():
    d1 = Data(x=[1, 2, 3, 4, 5], k1=[0, 0, 1, 1, 2], label='d1')
    d2 = Data(y=[2, 4, 5, 8, 4], k2=[1, 3, 1, 2, 3], label='d2')
    d2.join_on_key(d1, 'k2', 'k1')

    s = d1.new_subset()
    s.subset_state = d1.id['x'] > 2
    assert_array_equal(s.to_mask(), [False, False, True, True, True])
    s = d2.new_subset()
    s.subset_state = d1.id['x'] > 2
    assert_array_equal(s.to_mask(), [True, False, True, True, False])


def test_eq_logic():
    d1 = Data(x=[1, 2, 3, 4, 5], k1=[0, 0, 1, 1, 2], label='d1')
    d2 = Data(y=[2, 4, 5, 8, 4], k2=[1, 3, 1, 2, 3], label='d2')
    d3 = Data(y=[2, 4, 5, 8, 4], k2=[1, 3, 1, 2, 3], label='d3')

    dc = DataCollection([d1, d2, d3])
    a = Join_Link(cids1=[d1.id['k1']], cids2=[d2.id['k2']], data1=d1, data2=d2)
    b = Join_Link(cids1=[d2.id['k2']], cids2=[d1.id['k1']], data1=d2, data2=d1)
    assert a == a
    assert a == b
    c = Join_Link(cids1=[d3.id['k2']], cids2=[d1.id['k1']], data1=d3, data2=d1)
    assert c != b
    assert c != a


# TODO: import this from test_link_editor
def get_action(link_widget, text):
    for submenu in link_widget._menu.children():
        if isinstance(submenu, QtWidgets.QMenu):
            for action in submenu.actions():
                if action.text() == text:
                    return action
    raise ValueError("Action '{0}' not found".format(text))


class TestLinkEditorForJoins:

    def setup_method(self, method):

        self.data1 = Data(x=['101', '102', '105'], y=[2, 3, 4], z=[6, 5, 4], label='data1')
        self.data2 = Data(a=['102', '104', '105'], b=[4, 5, 4], c=[3, 4, 1], label='data2')

        self.data_collection = DataCollection([self.data1, self.data2])

    def test_make_and_delete_link(self):
        # Make sure the dialog opens and closes and check default settings.
        dialog = LinkEditor(self.data_collection)
        dialog.show()
        link_widget = dialog.link_widget
        link_widget.state.data1 = self.data1
        link_widget.state.data2 = self.data2
        add_join_link = get_action(link_widget, 'Join on ID')

        add_join_link.trigger()
        # Ensure that all events get processed
        # key_joins only happen on dialog.accept()
        process_events()
        dialog.accept()

        assert len(self.data_collection.links) == 0
        assert len(self.data_collection._link_manager._external_links) == 1

        assert self.data1._key_joins != {}
        assert self.data2._key_joins != {}

        dialog.show()
        link_widget = dialog.link_widget

        # With two datasets this will select the current link
        assert link_widget.listsel_current_link.count() == 1
        assert link_widget.link_details.text().startswith('Join two datasets')
        link_widget.state.current_link.data1 = self.data1
        link_widget.state.current_link.data2 = self.data2

        link_widget.state.current_link.join_link = True  # Not sure why we need to set this in the test

        assert link_widget.state.current_link.link in self.data_collection._link_manager._external_links
        assert link_widget.button_remove_link.isEnabled()

        link_widget.button_remove_link.click()
        process_events()

        dialog.accept()

        assert len(self.data_collection.links) == 0
        assert len(self.data_collection._link_manager._external_links) == 0
        assert self.data1._key_joins == {}
        assert self.data2._key_joins == {}
