from unittest.mock import patch

from qtpy import QtWidgets
from glue.utils.qt import process_events
from glue.core import Data, DataCollection
from glue.dialogs.link_editor.qt import LinkEditor
from glue.core.component_link import ComponentLink
from glue.plugins.coordinate_helpers.link_helpers import Galactic_to_FK5, ICRS_to_Galactic
from glue.core.link_helpers import identity, functional_link_collection, LinkSame


def non_empty_rows_count(layout):
    """
    Determine how many rows of the QGridLayout are not empty
    """
    count = 0
    for row in range(layout.rowCount()):
        for col in range(layout.columnCount()):
            item = layout.itemAtPosition(row, col)
            if item is not None and item.widget() is not None and item.widget().isVisible():
                count += 1
                break
    return count


def get_action(link_widget, text):
    for submenu in link_widget._menu.children():
        if isinstance(submenu, QtWidgets.QMenu):
            for action in submenu.actions():
                if action.text() == text:
                    return action
    raise ValueError("Action '{0}' not found".format(text))


class TestLinkEditor:

    def setup_method(self, method):

        self.data1 = Data(x=[1, 2, 3], y=[2, 3, 4], z=[6, 5, 4], label='data1')
        self.data2 = Data(a=[2, 3, 4], b=[4, 5, 4], c=[3, 4, 1], label='data2')
        self.data3 = Data(i=[5, 4, 3], j=[2, 2, 1], label='data3')

        self.data_collection = DataCollection([self.data1, self.data2, self.data3])

    def test_defaults(self):
        # Make sure the dialog opens and closes and check default settings.
        dialog = LinkEditor(self.data_collection)
        dialog.show()
        link_widget = dialog.link_widget

        assert link_widget.state.data1 is None
        assert link_widget.state.data2 is None
        assert not link_widget.button_add_link.isEnabled()
        assert not link_widget.button_remove_link.isEnabled()

        link_widget.state.data1 = self.data2

        assert not link_widget.button_add_link.isEnabled()
        assert not link_widget.button_remove_link.isEnabled()

        link_widget.state.data2 = self.data1

        assert link_widget.button_add_link.isEnabled()
        assert not link_widget.button_remove_link.isEnabled()

        dialog.accept()

        assert len(self.data_collection.external_links) == 0

    def test_defaults_two(self):
        # Make sure the dialog opens and closes and check default settings. With
        # two datasets, the datasets should be selected by default.
        self.data_collection.remove(self.data3)
        dialog = LinkEditor(self.data_collection)
        dialog.show()
        link_widget = dialog.link_widget
        assert link_widget.state.data1 is self.data1
        assert link_widget.state.data2 is self.data2
        assert link_widget.button_add_link.isEnabled()
        assert not link_widget.button_remove_link.isEnabled()
        dialog.accept()
        assert len(self.data_collection.external_links) == 0

    def test_ui_behavior(self):

        # This is a bit more detailed test that checks that things update
        # correctly as we change various settings

        dialog = LinkEditor(self.data_collection)
        dialog.show()
        link_widget = dialog.link_widget

        link_widget.state.data1 = self.data1
        link_widget.state.data2 = self.data2

        add_identity_link = get_action(link_widget, 'identity')
        add_lengths_volume_link = get_action(link_widget, 'lengths_to_volume')

        # At this point, there should be no links in the main list widget
        # and nothing on the right.
        assert link_widget.listsel_current_link.count() == 0
        assert link_widget.link_details.text() == ''
        assert non_empty_rows_count(link_widget.combos1) == 0
        assert non_empty_rows_count(link_widget.combos2) == 0

        # Let's add an identity link
        add_identity_link.trigger()

        # Ensure that all events get processed
        process_events()

        # Now there should be one link in the main list and content in the
        # right hand panel.
        assert link_widget.listsel_current_link.count() == 1
        assert link_widget.link_details.text() == 'Link conceptually identical components'
        assert non_empty_rows_count(link_widget.combos1) == 1
        assert non_empty_rows_count(link_widget.combos2) == 1
        assert link_widget.combos1.itemAtPosition(0, 1).widget().currentText() == 'x'
        assert link_widget.combos2.itemAtPosition(0, 1).widget().currentText() == 'a'

        # Let's change the current components for the link
        link_widget.state.current_link.x = self.data1.id['y']
        link_widget.state.current_link.y = self.data2.id['b']

        # and make sure the UI gets updated
        assert link_widget.combos1.itemAtPosition(0, 1).widget().currentText() == 'y'
        assert link_widget.combos2.itemAtPosition(0, 1).widget().currentText() == 'b'

        # We now add another link of a different type
        add_lengths_volume_link.trigger()

        # Ensure that all events get processed
        process_events()

        # and make sure the UI has updated
        assert link_widget.listsel_current_link.count() == 2
        assert link_widget.link_details.text() == 'Convert between linear measurements and volume'
        assert non_empty_rows_count(link_widget.combos1) == 3
        assert non_empty_rows_count(link_widget.combos2) == 1
        assert link_widget.combos1.itemAtPosition(0, 1).widget().currentText() == 'x'
        assert link_widget.combos1.itemAtPosition(1, 1).widget().currentText() == 'y'
        assert link_widget.combos1.itemAtPosition(2, 1).widget().currentText() == 'z'
        assert link_widget.combos2.itemAtPosition(0, 1).widget().currentText() == 'a'

        # Try swapping the order of the data, the current link should stay the same
        link_widget.state.flip_data()
        assert link_widget.link_details.text() == 'Convert between linear measurements and volume'

        # And flip it back
        link_widget.state.flip_data()
        assert link_widget.link_details.text() == 'Convert between linear measurements and volume'

        # Now switch back to the first link
        link_widget.state.current_link = type(link_widget.state).current_link.get_choices(link_widget.state)[0]

        # and make sure the UI updates and has preserved the correct settings
        assert link_widget.listsel_current_link.count() == 2
        assert link_widget.link_details.text() == 'Link conceptually identical components'
        assert non_empty_rows_count(link_widget.combos1) == 1
        assert non_empty_rows_count(link_widget.combos2) == 1
        assert link_widget.combos1.itemAtPosition(0, 1).widget().currentText() == 'y'
        assert link_widget.combos2.itemAtPosition(0, 1).widget().currentText() == 'b'

        # Next up, we try changing the data

        link_widget.state.data1 = self.data3

        # At this point there should be no links in the list

        assert link_widget.listsel_current_link.count() == 0
        assert link_widget.link_details.text() == ''
        assert non_empty_rows_count(link_widget.combos1) == 0
        assert non_empty_rows_count(link_widget.combos2) == 0

        # Add another identity link
        add_identity_link.trigger()

        # Ensure that all events get processed
        process_events()

        # Now there should be one link in the main list
        assert link_widget.listsel_current_link.count() == 1
        assert link_widget.link_details.text() == 'Link conceptually identical components'
        assert non_empty_rows_count(link_widget.combos1) == 1
        assert non_empty_rows_count(link_widget.combos2) == 1
        assert link_widget.combos1.itemAtPosition(0, 1).widget().currentText() == 'i'
        assert link_widget.combos2.itemAtPosition(0, 1).widget().currentText() == 'a'

        # Switch back to the original data
        link_widget.state.data1 = self.data1

        # And check the output is as before
        assert link_widget.listsel_current_link.count() == 2
        assert link_widget.link_details.text() == 'Link conceptually identical components'
        assert non_empty_rows_count(link_widget.combos1) == 1
        assert non_empty_rows_count(link_widget.combos2) == 1
        assert link_widget.combos1.itemAtPosition(0, 1).widget().currentText() == 'y'
        assert link_widget.combos2.itemAtPosition(0, 1).widget().currentText() == 'b'

        # Let's now remove this link
        link_widget.button_remove_link.click()

        # Ensure that all events get processed
        process_events()

        # We should now see the lengths/volume link
        assert link_widget.listsel_current_link.count() == 1
        assert link_widget.link_details.text() == 'Convert between linear measurements and volume'
        assert non_empty_rows_count(link_widget.combos1) == 3
        assert non_empty_rows_count(link_widget.combos2) == 1
        assert link_widget.combos1.itemAtPosition(0, 1).widget().currentText() == 'x'
        assert link_widget.combos1.itemAtPosition(1, 1).widget().currentText() == 'y'
        assert link_widget.combos1.itemAtPosition(2, 1).widget().currentText() == 'z'
        assert link_widget.combos2.itemAtPosition(0, 1).widget().currentText() == 'a'

        dialog.accept()

        links = self.data_collection.external_links

        assert len(links) == 2

        assert isinstance(links[0], ComponentLink)
        assert links[0].get_from_ids()[0] is self.data1.id['x']
        assert links[0].get_from_ids()[1] is self.data1.id['y']
        assert links[0].get_from_ids()[2] is self.data1.id['z']
        assert links[0].get_to_id() is self.data2.id['a']

        assert isinstance(links[1], ComponentLink)
        assert links[1].get_from_ids()[0] is self.data3.id['i']
        assert links[1].get_to_id() is self.data2.id['a']

    def test_graph(self):

        dialog = LinkEditor(self.data_collection)
        dialog.show()
        link_widget = dialog.link_widget

        add_identity_link = get_action(link_widget, 'identity')

        graph = link_widget.graph_widget

        def click(node_or_edge):
            # We now simulate a selection - since we can't deterministically
            # figure out the exact pixel coordinates to use, we patch
            # 'find_object' to return the object we want to select.
            with patch.object(graph, 'find_object', return_value=node_or_edge):
                graph.mousePressEvent(None)

        def hover(node_or_edge):
            # Same as for select, we patch find_object
            with patch.object(graph, 'find_object', return_value=node_or_edge):
                graph.mouseMoveEvent(None)

        # To start with, no data should be selected
        assert link_widget.state.data1 is None
        assert link_widget.state.data2 is None

        # and the graph should have three nodes and no edges
        assert len(graph.nodes) == 3
        assert len(graph.edges) == 0

        click(graph.nodes[0])

        # Check that this has caused one dataset to be selected
        assert link_widget.state.data1 is self.data1
        assert link_widget.state.data2 is None

        # Click on the same node again and this should deselect the data
        # (but only once we move off from the node)

        click(graph.nodes[0])

        assert link_widget.state.data1 is self.data1
        assert link_widget.state.data2 is None

        hover(None)

        assert link_widget.state.data1 is None
        assert link_widget.state.data2 is None

        # Select it again
        click(graph.nodes[0])

        # and now select another node too
        click(graph.nodes[1])

        assert link_widget.state.data1 is self.data1
        assert link_widget.state.data2 is self.data2

        assert len(graph.nodes) == 3
        assert len(graph.edges) == 0

        add_identity_link.trigger()

        assert len(graph.nodes) == 3
        assert len(graph.edges) == 1

        # Unselect and select another node
        click(graph.nodes[1])
        click(graph.nodes[2])

        # and check the data selections have been updated
        assert link_widget.state.data1 is self.data1
        assert link_widget.state.data2 is self.data3

        # Deselect it and move off
        click(graph.nodes[2])
        hover(None)

        # and the second dataset should now once again be None
        assert link_widget.state.data1 is self.data1
        assert link_widget.state.data2 is None

        # Now change the data manually
        link_widget.state.data1 = self.data2
        link_widget.state.data2 = self.data3

        # and check that if we select the edge the datasets change back
        click(graph.edges[0])

        assert link_widget.state.data1 is self.data1
        assert link_widget.state.data2 is self.data2

        # Unselect and hover over nothing
        click(graph.edges[0])
        hover(None)
        assert link_widget.state.data1 is None
        assert link_widget.state.data2 is None

        # Hover over the edge and the datasets should change back
        hover(graph.edges[0])
        assert link_widget.state.data1 is self.data1
        assert link_widget.state.data2 is self.data2

        # And check that clicking outside of nodes/edges deselects everything
        click(None)
        assert link_widget.state.data1 is None
        assert link_widget.state.data2 is None

        # Select a node, select another, then make sure that selecting a third
        # one will deselect the two original ones
        click(graph.nodes[0])
        click(graph.nodes[1])
        click(graph.nodes[2])
        assert link_widget.state.data1 is self.data3
        assert link_widget.state.data2 is None

        dialog.accept()

    def test_preexisting_links(self):

        # Check that things work properly if there are pre-existing links

        link1 = ComponentLink([self.data1.id['x']], self.data2.id['c'])

        def add(x, y):
            return x + y

        def double(x):
            return x * 2

        def halve(x):
            return x / 2

        link2 = ComponentLink([self.data2.id['a'], self.data2.id['b']], self.data3.id['j'], using=add)
        link3 = ComponentLink([self.data3.id['i']], self.data2.id['c'], using=double, inverse=halve)

        # Test using a LinkHelper link since that caused a bug earlier
        link4 = LinkSame(self.data1.id['z'], self.data2.id['c'])

        self.data_collection.add_link(link1)
        self.data_collection.add_link(link2)
        self.data_collection.add_link(link3)
        self.data_collection.add_link(link4)

        dialog = LinkEditor(self.data_collection)
        dialog.show()
        link_widget = dialog.link_widget

        link_widget.state.data1 = self.data1
        link_widget.state.data2 = self.data2

        assert link_widget.listsel_current_link.count() == 2
        assert link_widget.link_details.text() == ''
        assert non_empty_rows_count(link_widget.combos1) == 1
        assert non_empty_rows_count(link_widget.combos2) == 1
        assert link_widget.combos1.itemAtPosition(0, 1).widget().currentText() == 'x'
        assert link_widget.combos2.itemAtPosition(0, 1).widget().currentText() == 'c'

        link_widget.state.current_link = type(link_widget.state).current_link.get_choices(link_widget.state)[1]
        assert link_widget.link_details.text() == ''
        assert non_empty_rows_count(link_widget.combos1) == 1
        assert non_empty_rows_count(link_widget.combos2) == 1
        assert link_widget.combos1.itemAtPosition(0, 1).widget().currentText() == 'z'
        assert link_widget.combos2.itemAtPosition(0, 1).widget().currentText() == 'c'

        link_widget.state.data1 = self.data3

        assert link_widget.listsel_current_link.count() == 2
        assert link_widget.link_details.text() == ''
        assert non_empty_rows_count(link_widget.combos1) == 1
        assert non_empty_rows_count(link_widget.combos2) == 2
        assert link_widget.combos1.itemAtPosition(0, 1).widget().currentText() == 'j'
        assert link_widget.combos2.itemAtPosition(0, 1).widget().currentText() == 'a'
        assert link_widget.combos2.itemAtPosition(1, 1).widget().currentText() == 'b'

        link_widget.state.current_link = type(link_widget.state).current_link.get_choices(link_widget.state)[1]

        assert link_widget.listsel_current_link.count() == 2
        assert link_widget.link_details.text() == ''
        assert non_empty_rows_count(link_widget.combos1) == 1
        assert non_empty_rows_count(link_widget.combos2) == 1
        assert link_widget.combos1.itemAtPosition(0, 1).widget().currentText() == 'i'
        assert link_widget.combos2.itemAtPosition(0, 1).widget().currentText() == 'c'

        dialog.accept()

        links = self.data_collection.external_links

        assert len(links) == 4

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

        assert isinstance(links[3], LinkSame)
        assert len(links[3].cids1) == 1
        assert links[3].cids1[0] is self.data1.id['z']
        assert len(links[3].cids2) == 1
        assert links[3].cids2[0] is self.data2.id['c']
        assert links[3].forwards is identity

    def test_add_helper(self):

        dialog = LinkEditor(self.data_collection)
        dialog.show()
        link_widget = dialog.link_widget

        link_widget.state.data1 = self.data1
        link_widget.state.data2 = self.data2

        add_coordinate_link = get_action(link_widget, 'ICRS <-> Galactic')

        # Add a coordinate link
        add_coordinate_link.trigger()

        # Ensure that all events get processed
        process_events()

        assert link_widget.listsel_current_link.count() == 1
        assert link_widget.link_details.text() == 'Link ICRS and Galactic coordinates'
        assert non_empty_rows_count(link_widget.combos1) == 2
        assert non_empty_rows_count(link_widget.combos2) == 2
        assert link_widget.combos1.itemAtPosition(0, 1).widget().currentText() == 'x'
        assert link_widget.combos1.itemAtPosition(1, 1).widget().currentText() == 'y'
        assert link_widget.combos2.itemAtPosition(0, 1).widget().currentText() == 'a'
        assert link_widget.combos2.itemAtPosition(1, 1).widget().currentText() == 'b'

        dialog.accept()

        links = self.data_collection.external_links

        assert len(links) == 1

        assert isinstance(links[0], ICRS_to_Galactic)
        assert links[0].cids1[0] is self.data1.id['x']
        assert links[0].cids1[1] is self.data1.id['y']
        assert links[0].cids2[0] is self.data2.id['a']
        assert links[0].cids2[1] is self.data2.id['b']

    def test_preexisting_helper(self):

        link1 = Galactic_to_FK5(cids1=[self.data1.id['x'], self.data1.id['y']],
                                cids2=[self.data2.id['c'], self.data2.id['b']])

        self.data_collection.add_link(link1)

        dialog = LinkEditor(self.data_collection)
        dialog.show()
        link_widget = dialog.link_widget

        assert link_widget.listsel_current_link.count() == 0

        link_widget.state.data1 = self.data1
        link_widget.state.data2 = self.data2

        assert link_widget.listsel_current_link.count() == 1
        assert link_widget.link_details.text() == 'Link Galactic and FK5 (J2000) Equatorial coordinates'
        assert non_empty_rows_count(link_widget.combos1) == 2
        assert non_empty_rows_count(link_widget.combos2) == 2
        assert link_widget.combos1.itemAtPosition(0, 1).widget().currentText() == 'x'
        assert link_widget.combos1.itemAtPosition(1, 1).widget().currentText() == 'y'
        assert link_widget.combos2.itemAtPosition(0, 1).widget().currentText() == 'c'
        assert link_widget.combos2.itemAtPosition(1, 1).widget().currentText() == 'b'

        dialog.accept()

        links = self.data_collection.external_links

        assert len(links) == 1

        assert isinstance(links[0], Galactic_to_FK5)
        assert links[0].cids1[0] is self.data1.id['x']
        assert links[0].cids1[1] is self.data1.id['y']
        assert links[0].cids2[0] is self.data2.id['c']
        assert links[0].cids2[1] is self.data2.id['b']

    def test_cancel(self):

        # Make sure that changes aren't saved if dialog is cancelled
        # This is a bit more detailed test that checks that things update
        # correctly as we change various settings

        link1 = ComponentLink([self.data1.id['x']], self.data2.id['c'])

        self.data_collection.add_link(link1)

        dialog = LinkEditor(self.data_collection)
        dialog.show()
        link_widget = dialog.link_widget

        link_widget.state.data1 = self.data1
        link_widget.state.data2 = self.data2

        link_widget.state.current_link.x = self.data1.id['y']

        assert link_widget.combos1.itemAtPosition(0, 1).widget().currentText() == 'y'

        add_identity_link = get_action(link_widget, 'identity')
        add_identity_link.trigger()

        assert link_widget.listsel_current_link.count() == 2

        dialog.reject()

        links = self.data_collection.external_links

        assert len(links) == 1

        assert isinstance(links[0], ComponentLink)
        assert links[0].get_from_ids()[0] is self.data1.id['x']
        assert links[0].get_to_id() is self.data2.id['c']

    def test_functional_link_collection(self):

        # Test that if we use a @link_helper in 'legacy' mode, i.e. with only
        # input labels, both datasets are available from the combos in the
        # link editor dialog. Also test the new-style @link_helper.

        def deg_arcsec(degree, arcsecond):
            return [ComponentLink([degree], arcsecond, using=lambda d: d * 3600),
                    ComponentLink([arcsecond], degree, using=lambda a: a / 3600)]

        # Old-style link helper

        helper1 = functional_link_collection(deg_arcsec, description='Legacy link',
                                             labels1=['deg', 'arcsec'], labels2=[])

        link1 = helper1(cids1=[self.data1.id['x'], self.data2.id['c']])

        self.data_collection.add_link(link1)

        # New-style link helper

        helper2 = functional_link_collection(deg_arcsec, description='New-style link',
                                             labels1=['deg'], labels2=['arcsec'])

        link2 = helper2(cids1=[self.data1.id['x']], cids2=[self.data2.id['c']])

        self.data_collection.add_link(link2)

        dialog = LinkEditor(self.data_collection)
        dialog.show()
        link_widget = dialog.link_widget

        assert link_widget.listsel_current_link.count() == 0

        link_widget.state.data1 = self.data1
        link_widget.state.data2 = self.data2

        assert link_widget.listsel_current_link.count() == 2

        assert not link_widget.combos1_header.isVisible()
        assert not link_widget.combos2_header.isVisible()
        assert link_widget.link_details.text() == 'Legacy link'
        assert non_empty_rows_count(link_widget.combos1) == 2
        assert non_empty_rows_count(link_widget.combos2) == 0
        assert link_widget.combos1.itemAtPosition(0, 1).widget().currentText() == 'x'
        assert link_widget.combos1.itemAtPosition(1, 1).widget().currentText() == 'c'

        link_widget.state.current_link = type(link_widget.state).current_link.get_choices(link_widget.state)[1]

        assert link_widget.combos1_header.isVisible()
        assert link_widget.combos2_header.isVisible()
        assert link_widget.link_details.text() == 'New-style link'
        assert non_empty_rows_count(link_widget.combos1) == 1
        assert non_empty_rows_count(link_widget.combos2) == 1
        assert link_widget.combos1.itemAtPosition(0, 1).widget().currentText() == 'x'
        assert link_widget.combos2.itemAtPosition(0, 1).widget().currentText() == 'c'

        dialog.accept()

        links = self.data_collection.external_links

        assert len(links) == 2

        assert isinstance(links[0], helper1)
        assert links[0].cids1[0] is self.data1.id['x']
        assert links[0].cids1[1] is self.data2.id['c']

        assert isinstance(links[1], helper2)
        assert links[1].cids1[0] is self.data1.id['x']
        assert links[1].cids2[0] is self.data2.id['c']

    def test_same_data(self):

        # Test that we can't set the same data twice

        dialog = LinkEditor(self.data_collection)
        dialog.show()
        link_widget = dialog.link_widget

        link_widget.state.data1 = self.data1
        link_widget.state.data2 = self.data2

        assert link_widget.state.data1 == self.data1
        assert link_widget.state.data2 == self.data2

        link_widget.state.data1 = self.data2

        assert link_widget.state.data1 == self.data2
        assert link_widget.state.data2 == self.data1

        link_widget.state.data2 = self.data2

        assert link_widget.state.data1 == self.data1
        assert link_widget.state.data2 == self.data2

        dialog.accept()

    def test_preexisting_links_twodata(self):

        # Regression test for an issue that occurred specifically if there were
        # exactly two datasets and pre-existing links (since this means that
        # the window opens with a current_link selected by default)

        data1 = Data(x=[1, 2, 3], y=[2, 3, 4], z=[6, 5, 4], label='data1')
        data2 = Data(a=[2, 3, 4], b=[4, 5, 4], c=[3, 4, 1], label='data2')

        data_collection = DataCollection([data1, data2])

        link1 = ComponentLink([data1.id['x']], data2.id['c'])
        data_collection.add_link(link1)

        dialog = LinkEditor(data_collection)
        dialog.show()

        dialog.accept()


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
        add_JoinLink = get_action(link_widget, 'Join on ID')

        add_JoinLink.trigger()
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

        link_widget.state.current_link.link_type = 'join'  # Not sure why we need to set this in the test

        assert link_widget.state.current_link.link in self.data_collection._link_manager._external_links
        assert link_widget.button_remove_link.isEnabled()

        link_widget.button_remove_link.click()
        process_events()

        dialog.accept()
        assert len(self.data_collection.links) == 0
        assert len(self.data_collection._link_manager._external_links) == 0
        assert self.data1._key_joins == {}
        assert self.data2._key_joins == {}
