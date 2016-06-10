from __future__ import absolute_import, division, print_function

from qtpy import QtWidgets

__all__ = ['pick_item', 'pick_class', 'get_text']

# TODO: update docstrings


def pick_item(items, labels, title="Pick an item", label="Pick an item",
              default=None):
    """ Prompt the user to choose an item

    :param items: List of items to choose
    :param labels: List of strings to label items
    :param title: Optional widget title
    :param label: Optional prompt

    Returns the selected item, or None
    """

    if default in items:
        current = items.index(default)
    else:
        current = 0

    choice, isok = QtWidgets.QInputDialog.getItem(None, title, label,
                                        labels, current=current,
                                        editable=False)
    if isok:
        index = labels.index(str(choice))
        return items[index]


def pick_class(classes, sort=False, **kwargs):
    """Prompt the user to pick from a list of classes using QT

    :param classes: list of class objects
    :param title: string of the prompt

    Returns:
       The class that was selected, or None
    """
    def _label(c):
        try:
            return c.LABEL
        except AttributeError:
            return c.__name__

    if sort:
        classes = sorted(classes, key=lambda x: _label(x))
    choices = [_label(c) for c in classes]
    return pick_item(classes, choices, **kwargs)


def get_text(title='Enter a label'):
    """Prompt the user to enter text using QT

    :param title: Name of the prompt

    *Returns*
       The text the user typed, or None
    """
    result, isok = QtWidgets.QInputDialog.getText(None, title, title)
    if isok:
        return str(result)
