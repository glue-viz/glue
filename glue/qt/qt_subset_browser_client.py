from time import sleep

from PyQt4.QtCore import SIGNAL
from PyQt4.QtCore import Qt
from PyQt4.QtGui import QMainWindow
from PyQt4.QtGui import QPushButton
from PyQt4.QtGui import QHBoxLayout
from PyQt4.QtGui import QCheckBox
from PyQt4.QtGui import QVBoxLayout
from PyQt4.QtGui import QColorDialog
from PyQt4.QtGui import QWidget
from PyQt4.QtGui import QButtonGroup
from PyQt4.QtGui import QPixmap
from PyQt4.QtGui import QColor
from PyQt4.QtGui import QIcon
from matplotlib.colors import ColorConverter

import glue
import glue.message as msg

def mpl_to_qt4_color(color):
    """ Convert a matplotlib color stirng into a PyQT4 QColor object

    INPUTS:
    -------
    color: String
       A color specification that matplotlib understands

    RETURNS:
    --------
    A QColor object representing color

    """
    cc = ColorConverter()
    r,g,b = cc.to_rgb(color)
    return QColor(r*255, g*255, b*255)

def qt4_to_mpl_color(color):
    """
    Conver a QColor object into a string that matplotlib understands

    Inputs:
    -------
    color: QColor instance

    OUTPUTS:
    --------
    A hex string describing that color
    """

    hex = color.name()
    return str(hex)


class QtSubsetBrowserClient(QMainWindow, glue.Client):
    """ QT class to edit subset colors """

    def __init__(self, data, parent=None):
        glue.Client.__init__(self, data)
        QMainWindow.__init__(self, parent)
        self.setWindowTitle("Subset Selection")
        self.subset_widgets = {}

        self.create_main_frame()

    def register_to_hub(self, hub):
        glue.Client.register_to_hub(self, hub)

        hub.subscribe(self,
                      msg.ActiveSubsetUpdateMessage,
                      handler=self._update_active_subset,
                      filter=lambda x:  \
                          x.sender.is_compatible(self.data))


    def _update_active_subset(self, message):
        subset = message.sender
        widget = self.subset_widgets[subset]

        if not widget['check'].isChecked():
            widget['check'].toggle()

    def _add_subset(self, message):
        """ Add a new subset to the panel """
        s = message.sender
        width = 50
        height = 50
        pm = QPixmap(width, height)

        color = mpl_to_qt4_color(s.style.color)
        pm.fill(color)
        icon = QIcon(pm)

        layer = QHBoxLayout()
        check = QCheckBox("")
        check.setChecked(False)
        self.connect(check, SIGNAL('stateChanged(int)'), self.on_check)
        widget = QPushButton(icon, "")
        self.connect(widget, SIGNAL('clicked()'), self.on_push)
        layer.addWidget(check)
        layer.addWidget(widget)
        self.layout.addLayout(layer)
        self.subset_widgets[s] = {'layer':layer, 'widget':widget,
                                  'check':check, 'pixmap':pm}

        # make sure buttons are exclusive
        self.check_group.addButton(check)


    def _remove_subset(self, message):
        pass

    def _update_all(self, message):
        pass

    def _update_subset(self, message):
        """ When a subset is updated, sync its color on the browser """
        subset = message.sender
        color = mpl_to_qt4_color(subset.style.color)
        pm = self.subset_widgets[subset]['pixmap']
        pm.fill(color)
        icon = QIcon(pm)
        self.subset_widgets[subset]['widget'].setIcon(icon)

    def on_check(self):
        """ When a checkbox is pressed, update the active subset """
        source = self.sender()
        for key in self.subset_widgets:
            check = self.subset_widgets[key]['check']
            if check is source:
                self.data.set_active_subset(key)



    def on_push(self):
        """ When a subset color button is pushed, update the color """
        source = self.sender()
        for key in self.subset_widgets:
            button = self.subset_widgets[key]['widget']
            if button == source:
                dialog = QColorDialog()
                initial = mpl_to_qt4_color(key.style.color)
                color = dialog.getColor(initial = initial)
                key.style.color = qt4_to_mpl_color(color)

    def create_main_frame(self):
        """ Create the main UI """

        self.main_frame = QWidget()
        self.layout = QVBoxLayout()

        for s in self.data.subsets:
            self._add_subset(glue.message.SubsetAddMessage(s))

        self.main_frame.setLayout(self.layout)
        self.setCentralWidget(self.main_frame)
        self.check_group = QButtonGroup()
        self.check_group.setExclusive(True)
