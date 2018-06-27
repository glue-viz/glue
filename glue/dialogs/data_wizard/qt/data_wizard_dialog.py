from __future__ import absolute_import, division, print_function

from qtpy.QtCore import Qt
from qtpy import QtWidgets
from glue.utils.qt import set_cursor_cm

__all__ = ['data_wizard', 'GlueDataDialog']


def data_wizard():
    """ QT Dialog to load a file into a new data object

    Returns:
       A list of new data objects. Returns an empty list if
       selection is canceled.
    """
    def report_error(error, factory, curfile):
        import traceback
        retry = QtWidgets.QMessageBox.Retry
        cancel = QtWidgets.QMessageBox.Cancel
        buttons = retry | cancel
        detail = traceback.format_exc()
        msg = "\n".join(["Could not load %s (wrong load method?)" % curfile,
                         "File load method: %s" % factory.label])
        detail = "\n\n".join(["Error message: %s" % error, detail])
        mb = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Critical, "Data Load Error", msg)
        mb.setDetailedText(detail)
        mb.setDefaultButton(cancel)
        mb.setStandardButtons(buttons)
        ok = mb.exec_()
        return ok == retry

    while True:
        gdd = GlueDataDialog()
        try:
            result = gdd.load_data()
            break
        except Exception as e:
            decision = report_error(e, gdd.factory(), gdd._curfile)
            if not decision:
                return []
    return result


class GlueDataDialog(object):

    def __init__(self, parent=None):
        self._fd = QtWidgets.QFileDialog(parent)
        from glue.config import data_factory
        self.filters = [(f, self._filter(f))
                        for f in data_factory.members if not f.deprecated]
        self.setNameFilter()
        self._fd.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        self._curfile = ''
        try:
            self._fd.setOption(
                QtWidgets.QFileDialog.Option.HideNameFilterDetails, True)
        except AttributeError:  # HideNameFilterDetails not present
            pass

    def factory(self):
        fltr = self._fd.selectedNameFilter()
        for k, v in self.filters:
            if v.startswith(fltr):
                return k

    def setNameFilter(self):
        fltr = ";;".join([flt for fac, flt in self.filters])
        self._fd.setNameFilter(fltr)

    def _filter(self, factory):
        return "%s (*)" % factory.label

    def paths(self):
        """
        Return all selected paths, as a list of unicode strings
        """
        return self._fd.selectedFiles()

    def _get_paths_and_factory(self):
        """Show dialog to get a file path and data factory

        :rtype: tuple of (list-of-strings, func)
                giving the path and data factory.
                returns ([], None) if user cancels dialog
        """
        result = self._fd.exec_()
        if result == QtWidgets.QDialog.Rejected:
            return [], None
        # path = list(map(str, self.paths()))  # cast out of unicode
        path = list(self.paths())
        factory = self.factory()
        return path, factory

    def load_data(self):
        """Highest level method to interactively load a data set.

        :rtype: A list of constructed data objects
        """
        from glue.core.data_factories import data_label, load_data
        paths, fac = self._get_paths_and_factory()
        result = []

        # Check that the user didn't select a .glu file by mistake
        for path in paths:
            if path.endswith('.glu'):
                mb = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Critical,
                                           "Error loading data",
                                           "It looks like you have selected "
                                           "a .glu session file. You should open "
                                           "this using 'Open Session' under the "
                                           "'File' menu instead")
                mb.exec_()
                return []

        with set_cursor_cm(Qt.WaitCursor):
            for path in paths:
                self._curfile = path
                d = load_data(path, factory=fac.function)
                if not isinstance(d, list):
                    if not d.label:
                        d.label = data_label(path)
                    d = [d]
                result.extend(d)

        return result
