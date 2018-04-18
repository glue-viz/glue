from __future__ import absolute_import, division, print_function

from mock import patch

from numpy.testing import assert_equal

from glue.core import Data, DataCollection, HubListener, ComponentID
from glue.core import message as msg
from glue.core.component_link import ComponentLink
from glue.core.parse import ParsedCommand, ParsedComponentLink

from ..component_arithmetic import ArithmeticEditorWidget, EquationEditorDialog


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
        assert reordered == self.reordered
        assert numerical is self.numerical


class TestArithmeticEditorWidget:

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
        editor = ArithmeticEditorWidget(self.data_collection)
        editor.show()
        editor.button_ok.click()
        self.listener1.assert_exact_changes()
        self.listener2.assert_exact_changes()
        editor.close()

    def test_add_derived_and_rename(self):
        editor = ArithmeticEditorWidget(self.data_collection)
        editor.show()
        with patch.object(EquationEditorDialog, 'exec_', auto_accept('{x} + {y}')):
            editor.button_add_derived.click()
        item = list(editor.list)[0]
        item.setText(0, 'new')
        editor.button_ok.click()
        self.listener1.assert_exact_changes(added=[self.data1.id['new']])
        self.listener2.assert_exact_changes()
        assert_equal(self.data1['new'], [4.5, 6.5, 2.0])
        editor.close()

    def test_add_derived_and_cancel(self):
        editor = ArithmeticEditorWidget(self.data_collection)
        editor.show()
        with patch.object(EquationEditorDialog, 'exec_', auto_reject()):
            editor.button_add_derived.click()
        assert len(editor.list) == 0
        editor.close()

    def test_edit_existing_equation(self):
        assert_equal(self.data2['d'], [3, 4, 1])
        editor = ArithmeticEditorWidget(self.data_collection)
        editor.show()
        assert len(editor.list) == 0
        editor.combosel_data.setCurrentIndex(1)
        assert len(editor.list) == 1
        editor.list.select_cid(self.data2.id['d'])
        with patch.object(EquationEditorDialog, 'exec_', auto_accept('{a} + {b}')):
            editor.button_edit_derived.click()
        editor.button_ok.click()
        self.listener1.assert_exact_changes()
        self.listener2.assert_exact_changes(numerical=True)
        assert_equal(self.data2['d'], [4.5, 2.0, 4.5])
        editor.close()

    # TODO: add an updated version of the following test back once we add
    # support for using derived components in derived components.
    #
    # def test_edit_equation_after_rename(self):
    #     editor = ArithmeticEditorWidget(self.data_collection)
    #     editor.show()
    #     editor.combosel_data.setCurrentIndex(1)
    #     editor.list['main'].select_cid(self.data2.id['a'])
    #     editor.list['main'].selected_item.setText(0, 'renamed')
    #     editor.list.select_cid(self.data2.id['d'])
    #     with patch.object(EquationEditorDialog, 'exec_', auto_accept('{renamed} + 1')):
    #         editor.button_edit_derived.click()
    #     editor.button_ok.click()
    #     self.listener1.assert_exact_changes()
    #     self.listener2.assert_exact_changes(renamed=[self.data2.id['renamed']], numerical=True)
    #     assert_equal(self.data2['d'], [4, 5, 2])
