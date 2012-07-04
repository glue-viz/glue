from time import ctime

from PyQt4.QtGui import QWidget, QTableWidgetItem

from ... import core

from ..ui.message_widget import Ui_MessageWidget


class MessageWidget(QWidget, core.hub.HubListener):
    """ This simple class displays all messages broadcast
    by a hub. It is mainly intended for debugging """
    def __init__(self):
        QWidget.__init__(self)
        self.ui = Ui_MessageWidget()
        self.ui.setupUi(self)
        self.ui.messageTable.setColumnCount(3)
        labels =['Time', 'Message', 'Sender']
        self.ui.messageTable.setHorizontalHeaderLabels(labels)

    def register_to_hub(self, hub):
        # catch all messages
        hub.subscribe(self, core.message.Message,
                      handler = self.process_message,
                      filter = lambda x:True)

    def process_message(self, message):
        row = self.ui.messageTable.rowCount() * 0
        self.ui.messageTable.insertRow(0)
        tm = QTableWidgetItem(ctime().split()[3])
        typ = str(type(message)).split("'")[-2].split('.')[-1]
        mtyp = QTableWidgetItem(typ)
        typ = str(type(message.sender)).split("'")[-2].split('.')[-1]
        sender = QTableWidgetItem(typ)
        self.ui.messageTable.setItem(row, 0, tm)
        self.ui.messageTable.setItem(row, 1, mtyp)
        self.ui.messageTable.setItem(row, 2, sender)
        self.ui.messageTable.resizeColumnsToContents()

