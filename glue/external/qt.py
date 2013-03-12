""" A Qt API selector that can be used to switch between PyQt and PySide.

This file lovingly borrows from the IPython and python_qt_binding project

https://github.com/ipython/ipython/blob/master/IPython/external/qt.py
https://github.com/ros-visualization/python_qt_binding/


See also this discussion
http://qt-project.org/wiki/Differences_Between_PySide_and_PyQt

Do not use this if you need PyQt with the old QString/QVariant API.
"""

import os
import sys

# Available APIs.
QT_API_PYQT = 'pyqt'
QT_API_PYSIDE = 'pyside'
QT_API = None


def prepare_pyqt4():
    # For PySide compatibility, use the new-style string API that automatically
    # converts QStrings to Unicode Python strings. Also, automatically unpack
    # QVariants to their underlying objects.
    import sip
    sip.setapi('QString', 2)
    sip.setapi('QVariant', 2)


def register_module(module, modlabel):
    """Register an imported module into a
    submodule of glue.external.qt. Enables syntax like
    from glue.qt.QtGui import QMessageBox
    """
    sys.modules[__name__ + '.' + modlabel] = module


def _load_pyqt4():
    prepare_pyqt4()
    from PyQt4 import QtCore, QtGui, QtTest
    if QtCore.PYQT_VERSION_STR < '4.8':
        raise ImportError("Glue Requires PyQt4 >= 4.8")

    QtCore.Signal = QtCore.pyqtSignal
    QtCore.Slot = QtCore.pyqtSlot
    QtCore.Property = QtCore.pyqtProperty

    from PyQt4.QtGui import QFileDialog
    QFileDialog.getOpenFileName = QFileDialog.getOpenFileNameAndFilter
    QFileDialog.getSaveFileName = QFileDialog.getSaveFileNameAndFilter

    register_module(QtCore, 'QtCore')
    register_module(QtGui, 'QtGui')
    register_module(QtTest, 'QtTest')

    global QT_API
    QT_API = QT_API_PYQT


def _load_pyside():
    from PySide import QtCore, QtGui, __version__, QtTest
    if __version__ < '1.0.3':
        # old PySide, fallback on PyQt
        raise ImportError("Glue requires PySide >= 1.0.3")

    register_module(QtCore, 'QtCore')
    register_module(QtGui, 'QtGui')
    register_module(QtTest, 'QtTest')

    def setMargin(self, x):
        self.setContentsMargins(x, x, x, x)
    QtGui.QLayout.setMargin = setMargin

    global QT_API
    QT_API = QT_API_PYSIDE


loaders = [_load_pyqt4, _load_pyside]
if os.environ.get('QT_API') == QT_API_PYSIDE:
    loaders = loaders[::-1]

#acutally do the loading
for loader in loaders:
    msgs = []
    try:
        loader()
        #we set this env var, since IPython also looks for it
        os.environ['QT_API'] = QT_API
        QtCore = sys.modules[__name__ + '.QtCore']
        QtGui = sys.modules[__name__ + '.QtGui']
        break
    except ImportError as e:
        msgs.append(str(e))
        pass
else:
    raise ImportError("Could not find a suitable QT installation."
                      " Encountered the following errors: %s" %
                      '\n'.join(msgs))
