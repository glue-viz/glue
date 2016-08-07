from __future__ import absolute_import, division, print_function

from glue.utils.qt import get_qapp

from .. import QMessageBoxPatched as QMessageBox


def test_main():

    app = get_qapp()

    w = QMessageBox(QMessageBox.Critical, "Error", "An error occurred")
    w.setDetailedText("Spam")
    w.select_all()
    w.copy_detailed()

    assert app.clipboard().text() == "Spam"

    app.quit()
