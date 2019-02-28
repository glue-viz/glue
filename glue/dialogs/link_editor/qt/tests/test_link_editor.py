from glue.core import Data, DataCollection
from glue.dialogs.link_editor.qt import LinkEditor


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
