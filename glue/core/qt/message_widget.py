from __future__ import absolute_import, division, print_function

import os
from time import ctime

from qtpy import QtWidgets
from glue import core
from glue.utils.qt import load_ui


class MessageWidget(QtWidgets.QWidget, core.hub.HubListener):
    """ This simple class displays all messages broadcast
    by a hub. It is mainly intended for debugging """

    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.ui = load_ui('message_widget.ui', self,
                          directory=os.path.dirname(__file__))
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
        tm = QtWidgets.QTableWidgetItem(ctime().split()[3])
        typ = str(type(message)).split("'")[-2].split('.')[-1]
        mtyp = QtWidgets.QTableWidgetItem(typ)
        typ = str(type(message.sender)).split("'")[-2].split('.')[-1]
        sender = QtWidgets.QTableWidgetItem(typ)
        self.ui.messageTable.setItem(row, 0, tm)
        self.ui.messageTable.setItem(row, 1, mtyp)
        self.ui.messageTable.setItem(row, 2, sender)
        self.ui.messageTable.resizeColumnsToContents()
