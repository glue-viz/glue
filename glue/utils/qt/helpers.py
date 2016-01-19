from __future__ import absolute_import, division, print_function

def update_combobox(combo, labeldata):
    """
    Redefine the items in a QComboBox

    Parameters
    ----------
    widget : QComboBox
       The widget to update
    labeldata : sequence of N (label, data) tuples
       The combobox will contain N items with the appropriate
       labels, and data set as the userData

    Returns
    -------
    combo : QComboBox
        The updated input

    Notes
    -----

    If the current userData in the combo box matches
    any of labeldata, that selection will be retained.
    Otherwise, the first item will be selected.

    Signals are disabled while the combo box is updated

    The QComboBox is modified inplace
    """
    
    combo.blockSignals(True)
    idx = combo.currentIndex()
    if idx >= 0:
        current = combo.itemData(idx)
    else:
        current = None

    combo.clear()
    index = 0
    for i, (label, data) in enumerate(labeldata):
        combo.addItem(label, userData=data)
        if data is current:
            index = i
    combo.blockSignals(False)
    combo.setCurrentIndex(index)

    # We need to force emit this, otherwise if the index happens to be the
    # same as before, even if the data is different, callbacks won't be
    # called.
    if idx == index or idx == -1:
        combo.currentIndexChanged.emit(index)
