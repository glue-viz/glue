from glue.core.data import Data

from ..application import GlueApplication


def combo_labels(combo):
    return combo.currentText(), [combo.itemText(idx) for idx in range(combo.count())]


def test_edit_subset_mode_toolbar():

    ga = GlueApplication()
    dc = ga.data_collection

    tbar = ga._mode_toolbar
    edit = ga.session.edit_subset_mode

    subset_combo = tbar.subset_combo
    mode_label = tbar._label_subset_mode

    dc.append(Data(x=[1, 2, 3]))

    assert combo_labels(subset_combo) == ('None/Create New', ['None/Create New'])
    assert mode_label.text() == '(the next selection will create a subset)'

    sg1 = dc.new_subset_group(subset_state=dc[0].id['x'] > 1, label='Subset 1')

    assert combo_labels(subset_combo) == ('Subset 1', ['Subset 1', 'None/Create New'])
    assert mode_label.text() == 'Mode:'

    sg2 = dc.new_subset_group(subset_state=dc[0].id['x'] < 1, label='Subset 2')

    assert combo_labels(subset_combo) == ('Subset 2', ['Subset 1', 'Subset 2', 'None/Create New'])
    assert mode_label.text() == 'Mode:'

    edit.edit_subset = [sg1, sg2]

    assert combo_labels(subset_combo) == ('Multiple subsets', ['Multiple subsets', 'Subset 1', 'Subset 2', 'None/Create New'])
    assert mode_label.text() == 'Mode:'

    edit.edit_subset = [sg1]

    assert combo_labels(subset_combo) == ('Subset 1', ['Subset 1', 'Subset 2', 'None/Create New'])
    assert mode_label.text() == 'Mode:'

    edit.edit_subset = []

    assert combo_labels(subset_combo) == ('None/Create New', ['Subset 1', 'Subset 2', 'None/Create New'])
    assert mode_label.text() == '(the next selection will create a subset)'

    edit.edit_subset = [sg2]

    assert combo_labels(subset_combo) == ('Subset 2', ['Subset 1', 'Subset 2', 'None/Create New'])
    assert mode_label.text() == 'Mode:'
