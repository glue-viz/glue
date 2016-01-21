from __future__ import absolute_import, division, print_function

from time import ctime

from glue.external.qt import QtGui

from glue import core

from glue.qt.qtutil import load_ui


class MessageWidget(QtGui.QWidget, core.hub.HubListener):
    """ This simple class displays all messages broadcast
    by a hub. It is mainly intended for debugging """
    def __init__(self):
        QtGui.QWidget.__init__(self)
        self.ui = load_ui('messagewidget', self)
        self.ui.messageTable.setColumnCount(3)
        labels = ['Time', 'Message', 'Sender']
        self.ui.messageTable.setHorizontalHeaderLabels(labels)

    def register_to_hub(self, hub):
        # catch all messages
        hub.subscribe(self, core.message.Message,
                      handler=self.process_message,
                      filter=lambda x: True)

    def process_message(self, message):
        row = self.ui.messageTable.rowCount() * 0
        self.ui.messageTable.insertRow(0)
        tm = QtGui.QTableWidgetItem(ctime().split()[3])
        typ = str(type(message)).split("'")[-2].split('.')[-1]
        mtyp = QtGui.QTableWidgetItem(typ)
        typ = str(type(message.sender)).split("'")[-2].split('.')[-1]
        sender = QtGui.QTableWidgetItem(typ)
        self.ui.messageTable.setItem(row, 0, tm)
        self.ui.messageTable.setItem(row, 1, mtyp)
        self.ui.messageTable.setItem(row, 2, sender)
        self.ui.messageTable.resizeColumnsToContents()
