# This file was originally from qt-helpers, for which the license is below.
# However, it now mainly uses QtPy and provides some additional patches. Once
# these are in QtPy, we can remove this file altogether (and move the remaining)
# functions to ``glue.utils.qt``.
#
# Original license for qt-helpers:
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


from __future__ import absolute_import, division, print_function

__all__ = ['get_qapp', 'load_ui']

from qtpy import QtCore, QtGui, QtWidgets, PYSIDE, PYQT4, PYQT5

def load_ui(path, parent=None):
    from qtpy.uic import loadUi
    return loadUi(path, parent)

qapp = None

def get_qapp(icon_path=None):
    global qapp
    qapp = QtWidgets.QApplication.instance()
    if qapp is None:
        qapp = QtWidgets.QApplication([''])
        qapp.setQuitOnLastWindowClosed(True)
        if icon_path is not None:
            qapp.setWindowIcon(QIcon(icon_path))

    # Make sure we use high resolution icons with PyQt5 for HDPI
    # displays. TODO: check impact on non-HDPI displays.
    if PYQT5:
        qapp.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)

    return qapp
