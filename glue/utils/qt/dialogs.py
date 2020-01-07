from qtpy import QtWidgets

__all__ = ['pick_item', 'pick_class', 'get_text', 'CenteredDialog']


def pick_item(items, labels, title="Pick an item", label="Pick an item",
              default=None):
    """
    Prompt the user to choose an item

    Returns the selected item, or `None`

    Parameters
    ----------
    items : iterable
        Items to choose (can be any Python object)
    labels : iterables
        Labels for the items to choose
    title : str, optional
        The title of the dialog
    label : str, optional
        The prompt message
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
    """
    Prompt the user to pick from a list of classes using Qt

    This is the same as `pick_item`, but the labels are automatically determined
    from the classes using the LABEL attribute, and if not set, then the
    __name__.

    Returns the class that was selected, or `None`

    Parameters
    ----------
    classes : iterable
        The classes to choose from
    title : str, optional
        The title of the dialog
    label : str, optional
        The prompt message
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


def get_text(title='Enter a label', default=None):
    """
    Prompt the user to enter text using Qt

    Returns the text the user typed, or `None`

    Parameters
    ----------
    title : str
        The prompt message and widget title.
    default : str
        The default text to show in the prompt.
    """
    result, isok = QtWidgets.QInputDialog.getText(None, title, title, text=default)
    if isok:
        return str(result)


class CenteredDialog(QtWidgets.QDialog):
    """
    A dialog that is centered on the screen.
    """

    def center(self):
        # Adapted from StackOverflow
        # https://stackoverflow.com/questions/20243637/pyqt4-center-window-on-active-screen
        frameGm = self.frameGeometry()
        screen = QtWidgets.QApplication.desktop().screenNumber(QtWidgets.QApplication.desktop().cursor().pos())
        centerPoint = QtWidgets.QApplication.desktop().screenGeometry(screen).center()
        frameGm.moveCenter(centerPoint)
        self.move(frameGm.topLeft())
