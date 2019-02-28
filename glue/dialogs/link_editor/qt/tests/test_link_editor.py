from glue.core import Data, DataCollection
from glue.dialogs.link_editor.qt import LinkEditor


def non_empty_rows_count(layout):
    """
    Determine how many rows of the QGridLayout are not empty
    """
    count = 0
    for row in range(layout.rowCount()):
        for col in range(layout.columnCount()):
            if layout.itemAtPosition(row, col) is not None:
                if layout.itemAtPosition(row, col).widget() is not None:
                    count += 1
                    break
    return count


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

        assert dialog.state.data1 is None
        assert dialog.state.data2 is None
        assert not dialog._ui.button_add_link.isEnabled()
        assert not dialog._ui.button_remove_link.isEnabled()

        dialog.state.data1 = self.data2

        assert not dialog._ui.button_add_link.isEnabled()
        assert not dialog._ui.button_remove_link.isEnabled()

        dialog.state.data2 = self.data1

        assert dialog._ui.button_add_link.isEnabled()
        assert dialog._ui.button_remove_link.isEnabled()

        dialog.accept()

    def test_defaults_two(self):
        # Make sure the dialog opens and closes and check default settings. With
        # two datasets, the datasets should be selected by default.
        self.data_collection.remove(self.data3)
        dialog = LinkEditor(self.data_collection)
        dialog.show()
        assert dialog.state.data1 is self.data1
        assert dialog.state.data2 is self.data2
        assert dialog._ui.button_add_link.isEnabled()
        assert dialog._ui.button_remove_link.isEnabled()
        dialog.accept()

    def test_ui_behavior(self):

        # This is a bit more detailed test that checks that things update
        # correctly as we change various settings

        dialog = LinkEditor(self.data_collection)
        dialog.show()
        dialog.state.data1 = self.data1
        dialog.state.data2 = self.data2

        # TODO: We should probably provide a way to get from a helper/function
        # to the action to avoid having to do this.
        add_identity_link = dialog._menu.children()[1].actions()[0]
        add_lengths_volume_link = dialog._menu.children()[1].actions()[1]

        # At this point, there should be no links in the main list widget
        # and nothing on the right.
        assert dialog._ui.listsel_links.count() == 0
        assert dialog._ui.link_details.count() == 0
        assert dialog._ui.link_io.count() == 0

        # Let's add an identity link
        add_identity_link.trigger()

        # Now there should be one link in the main list and content in the
        # right hand panel.
        assert dialog._ui.listsel_links.count() == 1
        assert dialog._ui.link_details.count() == 0
        assert non_empty_rows_count(dialog._ui.link_io) == 5
        assert dialog._ui.link_io.itemAtPosition(1, 1).widget().currentText() == 'x'
        assert dialog._ui.link_io.itemAtPosition(4, 1).widget().currentText() == 'a'

        # Let's change the current components for the link
        dialog.state.links.x = self.data1.id['y']
        dialog.state.links.output = self.data2.id['b']

        # and make sure the UI gets updated
        assert dialog._ui.link_io.itemAtPosition(1, 1).widget().currentText() == 'y'
        assert dialog._ui.link_io.itemAtPosition(4, 1).widget().currentText() == 'b'

        # We now add another link of a different type
        add_lengths_volume_link.trigger()

        # and make sure the UI has updated
        assert dialog._ui.listsel_links.count() == 2
        assert dialog._ui.link_details.count() == 0
        assert non_empty_rows_count(dialog._ui.link_io) == 7
        assert dialog._ui.link_io.itemAtPosition(1, 1).widget().currentText() == 'x'
        assert dialog._ui.link_io.itemAtPosition(2, 1).widget().currentText() == 'y'
        assert dialog._ui.link_io.itemAtPosition(3, 1).widget().currentText() == 'z'
        assert dialog._ui.link_io.itemAtPosition(6, 1).widget().currentText() == 'a'

        # Now switch back to the first link
        dialog.state.links = type(dialog.state).links.get_choices(dialog.state)[0]

        # and make sure the UI updates and has preserved the correct settings
        assert dialog._ui.listsel_links.count() == 2
        assert dialog._ui.link_details.count() == 0
        assert non_empty_rows_count(dialog._ui.link_io) == 5
        assert dialog._ui.link_io.itemAtPosition(1, 1).widget().currentText() == 'y'
        assert dialog._ui.link_io.itemAtPosition(4, 1).widget().currentText() == 'b'

        # Next up, we try changing the data

        dialog.state.data1 = self.data3

        # At this point there should be no links in the list

        assert dialog._ui.listsel_links.count() == 0
        assert dialog._ui.link_details.count() == 0
        assert non_empty_rows_count(dialog._ui.link_io) == 0

        # Add another identity link
        add_identity_link.trigger()

        # Now there should be one link in the main list
        assert dialog._ui.listsel_links.count() == 1
        assert dialog._ui.link_details.count() == 0
        assert non_empty_rows_count(dialog._ui.link_io) == 5
        assert dialog._ui.link_io.itemAtPosition(1, 1).widget().currentText() == 'i'
        assert dialog._ui.link_io.itemAtPosition(4, 1).widget().currentText() == 'a'

        # Switch back to the original data
        dialog.state.data1 = self.data1

        # And check the output is as before
        assert dialog._ui.listsel_links.count() == 2
        assert dialog._ui.link_details.count() == 0
        assert non_empty_rows_count(dialog._ui.link_io) == 5
        assert dialog._ui.link_io.itemAtPosition(1, 1).widget().currentText() == 'y'
        assert dialog._ui.link_io.itemAtPosition(4, 1).widget().currentText() == 'b'

        # Let's now remove this link
        dialog._ui.button_remove_link.click()

        # We should now see the lengths/volume link
        assert dialog._ui.listsel_links.count() == 1
        assert dialog._ui.link_details.count() == 0
        assert non_empty_rows_count(dialog._ui.link_io) == 7
        assert dialog._ui.link_io.itemAtPosition(1, 1).widget().currentText() == 'x'
        assert dialog._ui.link_io.itemAtPosition(2, 1).widget().currentText() == 'y'
        assert dialog._ui.link_io.itemAtPosition(3, 1).widget().currentText() == 'z'
        assert dialog._ui.link_io.itemAtPosition(6, 1).widget().currentText() == 'a'

        dialog.accept()
