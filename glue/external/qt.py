# qt-helpers - a common front-end to various Qt modules
#
# Copyright (c) 2015, Chris Beaumont and Thomas Robitaille
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the
#    distribution.
#  * Neither the name of the Glue project nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# This file includes code adapted from:
#
#   * IPython, which is released under the modified BSD license
#     (https://github.com/ipython/ipython/blob/master/COPYING.rst)
#
#   * python_qt_binding, which is released under the BSD license
#     (https://pypi.python.org/pypi/python_qt_binding)
#
# See also this discussion
#
# http://qt-project.org/wiki/Differences_Between_PySide_and_PyQt


"""
This module provides a way to import from Python Qt wrappers in a uniform
way, regardless of whether PySide or PyQt is used.

Do not use this if you need PyQt with the old QString/QVariant API.
"""

from __future__ import absolute_import, division, print_function

import os
import sys

__all__ = ['QtCore', 'QtGui', 'is_pyside', 'is_pyqt4', 'is_pyqt5', 'load_ui',
           'QT_API_PYQT4', 'QT_API_PYQT5', 'QT_API_PYSIDE']

# Available APIs.
QT_API_PYQT4 = 'pyqt'
QT_API_PYSIDE = 'pyside'
QT_API_PYQT5 = 'pyqt5'
QT_API = None


def is_pyside():
    return QT_API == QT_API_PYSIDE


def is_pyqt4():
    return QT_API == QT_API_PYQT4


def is_pyqt5():
    return QT_API == QT_API_PYQT5


# Backward-compatibility
is_pyqt = is_pyqt4
QT_API_PYQT = QT_API_PYQT4


class ImportDenier(object):
    """
    Import hook to protect importing of both PySide and PyQt.
    """

    _forbidden = set()

    def find_module(self, mod_name, pth):
        if pth or not mod_name in self._forbidden:
            return
        else:
            return self

    def load_module(self, mod_name):
        raise ImportError("Importing %s forbidden by %s"
                          % (mod_name, __name__))

    def deny_module(self, module):
        self._forbidden.add(module)

_import_hook = ImportDenier()
sys.meta_path.append(_import_hook)


def prepare_pyqt4():
    # For PySide compatibility, use the new-style string API that
    # automatically converts QStrings to Unicode Python strings. Also,
    # automatically unpack QVariants to their underlying objects.
    import sip
    sip.setapi('QString', 2)
    sip.setapi('QVariant', 2)

prepare_pyqt5 = prepare_pyqt4


def register_module(module, modlabel):
    """
    Register an imported module into a submodule of qt_helpers.

    This enables syntax such as:

        >>> from qt_helpers.QtGui import QMessageBox
    """
    sys.modules[__name__ + '.' + modlabel] = module


def _load_pyqt4():

    prepare_pyqt4()

    from PyQt4 import QtCore, QtGui, QtTest
    from distutils.version import LooseVersion

    if LooseVersion(QtCore.PYQT_VERSION_STR) < LooseVersion('4.8'):
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
    QT_API = QT_API_PYQT4

    _import_hook.deny_module('PySide')
    _import_hook.deny_module('PyQt5')


def _load_pyqt5():

    prepare_pyqt5()

    from PyQt5 import QtCore, QtGui, QtTest, QtWidgets
    from distutils.version import LooseVersion

    QtCore.Signal = QtCore.pyqtSignal
    QtCore.Slot = QtCore.pyqtSlot
    QtCore.Property = QtCore.pyqtProperty

    # In PyQt5, some widgets such as QMessageBox have moved from QtGui to
    # QWidgets so we add backward-compatibility hooks here for now
    for widget in dir(QtWidgets):
        if widget.startswith('Q'):
            setattr(QtGui, widget, getattr(QtWidgets, widget))
    QtGui.QItemSelection = QtCore.QItemSelection
    QtGui.QItemSelectionModel = QtCore.QItemSelectionModel
    QtGui.QSortFilterProxyModel = QtCore.QSortFilterProxyModel

    register_module(QtCore, 'QtCore')
    register_module(QtGui, 'QtGui')
    register_module(QtTest, 'QtTest')

    global QT_API
    QT_API = QT_API_PYQT5

    _import_hook.deny_module('PySide')
    _import_hook.deny_module('PyQt4')


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

    _import_hook.deny_module('PyQt4')
    _import_hook.deny_module('PyQt5')


QtCore = None
QtGui = None


def reload_qt():
    """
    Reload the Qt bindings.

    If the QT_API environment variable has been updated, this will load the
    new Qt bindings given by this variable. This should be used instead of
    the build-in ``reload`` function because the latter can in some cases
    cause issues with the ImportDenier (which prevents users from importing
    e.g. PySide if PyQt4 is loaded).
    """

    # Clear any forbidden modules
    _import_hook._forbidden.clear()

    # Quit app if active
    global qapp
    if qapp is not None:
        qapp.quit()
        qapp = None

    global QtCore
    global QtGui

    if os.environ.get('QT_API') == QT_API_PYQT5:
        loaders = [_load_pyqt5]
    elif os.environ.get('QT_API') == QT_API_PYSIDE:
        loaders = [_load_pyside, _load_pyqt4]
    else:
        loaders = [_load_pyqt4, _load_pyside, _load_pyqt5]

    msgs = []

    # acutally do the loading
    for loader in loaders:
        try:
            loader()
            # we set this env var, since IPython also looks for it
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

    # We patch this only now, once QtCore and QtGui are defined
    if is_pyside() or is_pyqt4():
        patch_qcombobox()

    # For PySide, we need to create a loadUi function
    if is_pyside():
        patch_loadui()


def load_ui(path, parent=None, custom_widgets=None):
    if is_pyside():
        return _load_ui_pyside(path, parent, custom_widgets=custom_widgets)
    elif is_pyqt5():
        return _load_ui_pyqt5(path, parent)
    else:
        return _load_ui_pyqt4(path, parent)


def _load_ui_pyside(path, parent, custom_widgets=None):
    from PySide import loadUi
    if custom_widgets is not None:
        custom_widgets = dict((widget.__name__, widget) for widget in custom_widgets)
    return loadUi(path, parent, customWidgets=custom_widgets)


def _load_ui_pyqt4(path, parent):
    from PyQt4.uic import loadUi
    return loadUi(path, parent)


def _load_ui_pyqt5(path, parent):
    from PyQt5.uic import loadUi
    return loadUi(path, parent)


qapp = None

def get_qapp(icon_path=None):
    global qapp
    qapp = QtGui.QApplication.instance()
    if qapp is None:
        qapp = QtGui.QApplication([''])
        qapp.setQuitOnLastWindowClosed(True)
        if icon_path is not None:
            qapp.setWindowIcon(QIcon(icon_path))

    # Make sure we use high resolution icons with PyQt5 for HDPI
    # displays. TODO: check impact on non-HDPI displays.
    if is_pyqt5():
        qapp.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)

    return qapp


def patch_qcombobox():

    # In PySide, using Python objects as userData in QComboBox causes
    # Segmentation faults under certain conditions. Even in cases where it
    # doesn't, findData does not work correctly. Likewise, findData also
    # does not work correctly with Python objects when using PyQt4. On the
    # other hand, PyQt5 deals with this case correctly. We therefore patch
    # QComboBox when using PyQt4 and PySide to avoid issues.

    class userDataWrapper(QtCore.QObject):
        def __init__(self, data, parent=None):
            super(userDataWrapper, self).__init__(parent)
            self.data = data

    _addItem = QtGui.QComboBox.addItem

    def addItem(self, *args, **kwargs):
        if len(args) == 3 or (not isinstance(args[0], QtGui.QIcon)
                              and len(args) == 2):
            args, kwargs['userData'] = args[:-1], args[-1]
        if 'userData' in kwargs:
            kwargs['userData'] = userDataWrapper(kwargs['userData'],
                                                 parent=self)
        _addItem(self, *args, **kwargs)

    _insertItem = QtGui.QComboBox.insertItem

    def insertItem(self, *args, **kwargs):
        if len(args) == 4 or (not isinstance(args[1], QtGui.QIcon)
                              and len(args) == 3):
            args, kwargs['userData'] = args[:-1], args[-1]
        if 'userData' in kwargs:
            kwargs['userData'] = userDataWrapper(kwargs['userData'],
                                                 parent=self)
        _insertItem(self, *args, **kwargs)

    _setItemData = QtGui.QComboBox.setItemData

    def setItemData(self, index, value, role=QtCore.Qt.UserRole):
        value = userDataWrapper(value, parent=self)
        _setItemData(self, index, value, role=role)

    _itemData = QtGui.QComboBox.itemData

    def itemData(self, index, role=QtCore.Qt.UserRole):
        userData = _itemData(self, index, role=role)
        if isinstance(userData, userDataWrapper):
            userData = userData.data
        return userData

    def findData(self, value):
        for i in range(self.count()):
            if self.itemData(i) == value:
                return i
        return -1

    QtGui.QComboBox.addItem = addItem
    QtGui.QComboBox.insertItem = insertItem
    QtGui.QComboBox.setItemData = setItemData
    QtGui.QComboBox.itemData = itemData
    QtGui.QComboBox.findData = findData


def patch_loadui():

    # In PySide, loadUi does not exist, so we define it using QUiLoader, and
    # then make sure we expose that function. This is based on the solution at
    #
    # https://gist.github.com/cpbotha/1b42a20c8f3eb9bb7cb8
    #
    # which was released under the MIT license:
    #
    # Copyright (c) 2011 Sebastian Wiesner <lunaryorn@gmail.com>
    # Modifications by Charl Botha <cpbotha@vxlabs.com>
    #
    # Permission is hereby granted, free of charge, to any person obtaining a
    # copy of this software and associated documentation files (the "Software"),
    # to deal in the Software without restriction, including without limitation
    # the rights to use, copy, modify, merge, publish, distribute, sublicense,
    # and/or sell copies of the Software, and to permit persons to whom the
    # Software is furnished to do so, subject to the following conditions:
    #
    # The above copyright notice and this permission notice shall be included in
    # all copies or substantial portions of the Software.
    #
    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    # THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    # FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    # DEALINGS IN THE SOFTWARE.
    #
    # This version includes further changes.

    from PySide.QtCore import Slot, QMetaObject
    from PySide.QtUiTools import QUiLoader
    from PySide.QtGui import QApplication, QMainWindow, QMessageBox

    class UiLoader(QUiLoader):
        """
        Subclass of :class:`~PySide.QtUiTools.QUiLoader` to create the user
        interface in a base instance.

        Unlike :class:`~PySide.QtUiTools.QUiLoader` itself this class does not
        create a new instance of the top-level widget, but creates the user
        interface in an existing instance of the top-level class if needed.

        This mimics the behaviour of :func:`PyQt4.uic.loadUi`.
        """

        def __init__(self, baseinstance, customWidgets=None):
            """
            Create a loader for the given ``baseinstance``.

            The user interface is created in ``baseinstance``, which must be an
            instance of the top-level class in the user interface to load, or a
            subclass thereof.

            ``customWidgets`` is a dictionary mapping from class name to class
            object for custom widgets. Usually, this should be done by calling
            registerCustomWidget on the QUiLoader, but with PySide 1.1.2 on
            Ubuntu 12.04 x86_64 this causes a segfault.

            ``parent`` is the parent object of this loader.
            """

            QUiLoader.__init__(self, baseinstance)
            self.baseinstance = baseinstance
            self.customWidgets = customWidgets

        def createWidget(self, class_name, parent=None, name=''):
            """
            Function that is called for each widget defined in ui file,
            overridden here to populate baseinstance instead.
            """

            if parent is None and self.baseinstance:
                # supposed to create the top-level widget, return the base
                # instance instead
                return self.baseinstance

            else:

                # For some reason, Line is not in the list of available
                # widgets, but works fine, so we have to special case it here.
                if class_name in self.availableWidgets() or class_name == 'Line':
                    # create a new widget for child widgets
                    widget = QUiLoader.createWidget(self, class_name, parent, name)

                else:
                    # if not in the list of availableWidgets, must be a custom
                    # widget this will raise KeyError if the user has not
                    # supplied the relevant class_name in the dictionary, or
                    # TypeError, if customWidgets is None
                    try:
                        widget = self.customWidgets[class_name](parent)
                    except (TypeError, KeyError) as e:
                        raise Exception('No custom widget ' + class_name + ' '
                                        'found in customWidgets')

                if self.baseinstance:
                    # set an attribute for the new child widget on the base
                    # instance, just like PyQt4.uic.loadUi does.
                    setattr(self.baseinstance, name, widget)

                return widget

    def loadUi(uifile, baseinstance=None, customWidgets=None,
               workingDirectory=None):
        """
        Dynamically load a user interface from the given ``uifile``.

        ``uifile`` is a string containing a file name of the UI file to load.

        If ``baseinstance`` is ``None``, the a new instance of the top-level
        widget will be created. Otherwise, the user interface is created within
        the given ``baseinstance``. In this case ``baseinstance`` must be an
        instance of the top-level widget class in the UI file to load, or a
        subclass thereof. In other words, if you've created a ``QMainWindow``
        interface in the designer, ``baseinstance`` must be a ``QMainWindow``
        or a subclass thereof, too. You cannot load a ``QMainWindow`` UI file
        with a plain :class:`~PySide.QtGui.QWidget` as ``baseinstance``.

        ``customWidgets`` is a dictionary mapping from class name to class
        object for custom widgets. Usually, this should be done by calling
        registerCustomWidget on the QUiLoader, but with PySide 1.1.2 on Ubuntu
        12.04 x86_64 this causes a segfault.

        :method:`~PySide.QtCore.QMetaObject.connectSlotsByName()` is called on
        the created user interface, so you can implemented your slots according
        to its conventions in your widget class.

        Return ``baseinstance``, if ``baseinstance`` is not ``None``. Otherwise
        return the newly created instance of the user interface.
        """

        loader = UiLoader(baseinstance, customWidgets)

        if workingDirectory is not None:
            loader.setWorkingDirectory(workingDirectory)

        widget = loader.load(uifile)
        QMetaObject.connectSlotsByName(widget)
        return widget

    import PySide
    PySide.loadUi = loadUi



# Now load default Qt
reload_qt()
