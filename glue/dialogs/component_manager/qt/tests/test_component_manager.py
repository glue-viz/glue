from __future__ import absolute_import, division, print_function

from mock import patch

from numpy.testing import assert_equal

from glue.core import Data, DataCollection, HubListener, ComponentID
from glue.core import message as msg
from glue.utils.qt import get_qapp
from glue.core.parse import ParsedCommand, ParsedComponentLink

from ..component_manager import ComponentManagerWidget, EquationEditorDialog


def auto_accept(equation):
    def exec_replacement(self):
        self.ui.expression.clear()
        self.ui.expression.insertPlainText(equation)
        self.accept()
    return exec_replacement


def auto_reject():
    def exec_replacement(self):
        self.ui.expression.clear()
        self.reject()
    return exec_replacement


class ChangeListener(HubListener):

    def __init__(self, data, *args, **kwargs):

        super(ChangeListener, self).__init__(*args, **kwargs)

        self.data = data
        self.removed = []
        self.added = []
        self.renamed = []
        self.reordered = []
        self.numerical = False

        self.register_to_hub(data.hub)

    def _has_data(self, message):
        return message.sender is self.data

    def register_to_hub(self, hub):

        hub.subscribe(self, msg.DataAddComponentMessage,
                      handler=self._component_added,
                      filter=self._has_data)

        hub.subscribe(self, msg.DataRemoveComponentMessage,
                      handler=self._component_removed,
                      filter=self._has_data)

        hub.subscribe(self, msg.DataRenameComponentMessage,
                      handler=self._component_renamed,
                      filter=self._has_data)

        hub.subscribe(self, msg.DataReorderComponentMessage,
                      handler=self._components_reordered,
                      filter=self._has_data)

        hub.subscribe(self, msg.NumericalDataChangedMessage,
                      handler=self._numerical_changed,
                      filter=self._has_data)

    def _component_added(self, message):
        self.added.append(message.component_id)

    def _component_removed(self, message):
        self.removed.append(message.component_id)

    def _component_renamed(self, message):
        self.renamed.append(message.component_id)

    def _components_reordered(self, message):
        self.reordered = message.component_ids

    def _numerical_changed(self, message):
        self.numerical = True

    def assert_exact_changes(self, added=[], removed=[], renamed=[], reordered=[], numerical=False):
        assert set(added) == set(self.added)
        assert set(removed) == set(self.removed)
        assert set(renamed) == set(self.renamed)
        assert set(reordered) == set(self.reordered)
        assert numerical is self.numerical


class TestComponentManagerWidget:

    def setup_method(self):
        self.app = get_qapp()
        self.data1 = Data(x=[1, 2, 3], y=[3.5, 4.5, -1.0], z=['a', 'r', 'w'])
        self.data2 = Data(a=[3, 4, 1], b=[1.5, -2.0, 3.5], c=['y', 'e', 'r'])

        # Add a derived component so that we can test how we deal with existing ones
        components = dict((cid.label, cid) for cid in self.data2.components)
        pc = ParsedCommand('{a}', components)
        link = ParsedComponentLink(ComponentID('d'), pc)
        self.data2.add_component_link(link)

        self.data_collection = DataCollection([self.data1, self.data2])
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
        item = list(self.manager.list['main'])[0]
        self.manager.list['main'].select_item(item)
        self.manager.button_remove_main.click()
        self.manager.button_ok.click()
        self.listener1.assert_exact_changes(removed=[x_cid])
        self.listener2.assert_exact_changes()

    def test_rename_valid(self):
        x_cid = self.data1.id['x']
        self.manager = ComponentManagerWidget(self.data_collection)
        self.manager.show()
        item = list(self.manager.list['main'])[0]
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
        item = list(self.manager.list['main'])[0]
        item.setText(0, 'y')
        assert not self.manager.button_ok.isEnabled()
        assert self.manager.ui.label_status.text() == 'Error: some components have duplicate names'
        item = list(self.manager.list['main'])[0]
        item.setText(0, 'a')
        assert self.manager.button_ok.isEnabled()
        assert self.manager.ui.label_status.text() == ''
        self.manager.button_ok.click()
        self.listener1.assert_exact_changes(renamed=[x_cid])
        self.listener2.assert_exact_changes()
        assert x_cid.label == 'a'
        assert_equal(self.data1['a'], [1, 2, 3])

    def test_add_derived_and_rename(self):
        self.manager = ComponentManagerWidget(self.data_collection)
        self.manager.show()
        with patch.object(EquationEditorDialog, 'exec_', auto_accept('{x} + {y}')):
            self.manager.button_add_derived.click()
        item = list(self.manager.list['derived'])[0]
        item.setText(0, 'new')
        self.manager.button_ok.click()
        self.listener1.assert_exact_changes(added=[self.data1.id['new']])
        self.listener2.assert_exact_changes()
        assert_equal(self.data1['new'], [4.5, 6.5, 2.0])

    def test_add_derived_and_cancel(self):
        self.manager = ComponentManagerWidget(self.data_collection)
        self.manager.show()
        with patch.object(EquationEditorDialog, 'exec_', auto_reject()):
            self.manager.button_add_derived.click()
        assert len(self.manager.list['derived']) == 0

    def test_edit_existing_equation(self):
        assert_equal(self.data2['d'], [3, 4, 1])
        self.manager = ComponentManagerWidget(self.data_collection)
        self.manager.show()
        assert len(self.manager.list['derived']) == 0
        self.manager.combosel_data.setCurrentIndex(1)
        assert len(self.manager.list['derived']) == 1
        self.manager.list['derived'].select_cid(self.data2.id['d'])
        with patch.object(EquationEditorDialog, 'exec_', auto_accept('{a} + {b}')):
            self.manager.button_edit_derived.click()
        self.manager.button_ok.click()
        self.listener1.assert_exact_changes()
        self.listener2.assert_exact_changes(numerical=True)
        assert_equal(self.data2['d'], [4.5, 2.0, 4.5])

    def test_edit_equation_after_rename(self):
        self.manager = ComponentManagerWidget(self.data_collection)
        self.manager.show()
        self.manager.combosel_data.setCurrentIndex(1)
        self.manager.list['main'].select_cid(self.data2.id['a'])
        self.manager.list['main'].selected_item.setText(0, 'renamed')
        self.manager.list['derived'].select_cid(self.data2.id['d'])
        with patch.object(EquationEditorDialog, 'exec_', auto_accept('{renamed} + 1')):
            self.manager.button_edit_derived.click()
        self.manager.button_ok.click()
        self.listener1.assert_exact_changes()
        self.listener2.assert_exact_changes(renamed=[self.data2.id['renamed']], numerical=True)
        assert_equal(self.data2['d'], [4, 5, 2])
