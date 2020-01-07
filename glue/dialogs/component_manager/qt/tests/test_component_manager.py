from numpy.testing import assert_equal

from glue.core import Data, DataCollection, ComponentID
from glue.core.component_link import ComponentLink
from glue.core.parse import ParsedCommand, ParsedComponentLink

from ..component_manager import ComponentManagerWidget
from glue.dialogs.component_arithmetic.qt.tests.test_component_arithmetic import ChangeListener


class TestComponentManagerWidget:

    def setup_method(self):

        self.data1 = Data(x=[1, 2, 3], y=[3.5, 4.5, -1.0], z=['a', 'r', 'w'])
        self.data2 = Data(a=[3, 4, 1], b=[1.5, -2.0, 3.5], c=['y', 'e', 'r'])

        # Add a derived component so that we can test how we deal with existing ones
        components = dict((cid.label, cid) for cid in self.data2.components)
        pc = ParsedCommand('{a}', components)
        link = ParsedComponentLink(ComponentID('d'), pc)
        self.data2.add_component_link(link)

        self.data_collection = DataCollection([self.data1, self.data2])

        link = ComponentLink([self.data1.id['x']], self.data2.id['a'])
        self.data_collection.add_link(link)

        self.listener1 = ChangeListener(self.data1)
        self.listener2 = ChangeListener(self.data2)

    def test_nochanges(self):
        self.manager = ComponentManagerWidget(self.data_collection)
        self.manager.show()
        self.manager.button_ok.click()
        self.listener1.assert_exact_changes()
        self.listener2.assert_exact_changes()

    def test_remove(self):
        x_cid = self.data1.id['x']
        self.manager = ComponentManagerWidget(self.data_collection)
        self.manager.show()
        item = list(self.manager.list)[0]
        self.manager.list.select_item(item)
        self.manager.button_remove_main.click()
        self.manager.button_ok.click()
        self.listener1.assert_exact_changes(removed=[x_cid])
        self.listener2.assert_exact_changes()

    def test_rename_valid(self):
        x_cid = self.data1.id['x']
        self.manager = ComponentManagerWidget(self.data_collection)
        self.manager.show()
        item = list(self.manager.list)[0]
        item.setText(0, 'newname')
        self.manager.button_ok.click()
        assert self.manager.result() == 1
        self.listener1.assert_exact_changes(renamed=[x_cid])
        self.listener2.assert_exact_changes()
        assert x_cid.label == 'newname'
        assert_equal(self.data1['newname'], [1, 2, 3])

    def test_rename_invalid(self):
        x_cid = self.data1.id['x']
        self.manager = ComponentManagerWidget(self.data_collection)
        self.manager.show()
        item = list(self.manager.list)[0]
        item.setText(0, 'y')
        assert not self.manager.button_ok.isEnabled()
        assert self.manager.ui.label_status.text() == 'Error: some components have duplicate names'
        item = list(self.manager.list)[0]
        item.setText(0, 'a')
        assert self.manager.button_ok.isEnabled()
        assert self.manager.ui.label_status.text() == ''
        self.manager.button_ok.click()
        self.listener1.assert_exact_changes(renamed=[x_cid])
        self.listener2.assert_exact_changes()
        assert x_cid.label == 'a'
        assert_equal(self.data1['a'], [1, 2, 3])
