import os

import matplotlib
from matplotlib.backends.backend_qt4 import NavigationToolbar2QT
from PyQt4 import QtCore, QtGui

class GlueToolbar(NavigationToolbar2QT):
    def __init__(self, canvas, frame):
        """ Create a new toolbar object

        Parameters
        ----------
        data_collection : DataCollection instance
         The data collection that this toolbar is meant to edit.
         The toolbar looks to this collection for the available subsets
         to manipulate.
        canvas : Maptloblib canvas instance
         The drawing canvas to interact with
        frame : QWidget
         The QT frame that the canvas is embedded within.
        """
        self.buttons = {}
        self.basedir = None
        self._custom_idMove = None

        NavigationToolbar2QT.__init__(self, canvas, frame)

    def _init_toolbar(self):
        self.basedir = os.path.join(matplotlib.rcParams[ 'datapath' ],'images')

        a = self.addAction(self._icon('home.svg'), 'Home', self.home)
        a.setToolTip('Reset original view')
        a = self.addAction(self._icon('back.svg'), 'Back', self.back)
        a.setToolTip('Back to previous view')
        a = self.addAction(self._icon('forward.svg'), 'Forward', self.forward)
        a.setToolTip('Forward to next view')
        self.addSeparator()

        a = self.addAction(self._icon('move.svg'), 'Pan', self.pan)
        a.setToolTip('Pan axes with left mouse, zoom with right')
        a.setCheckable(True)
        self.buttons['PAN'] = a

        a = self.addAction(self._icon('zoom_to_rect.svg'), 'Zoom', self.zoom)
        a.setToolTip('Zoom to rectangle')
        a.setCheckable(True)
        self.buttons['ZOOM'] = a

        self.addSeparator()
        a = self.addAction(self._icon('subplots.png'), 'Subplots',
                           self.configure_subplots)
        a.setToolTip('Configure subplots')

        a = self.addAction(self._icon("qt4_editor_options.svg"),
                           'Customize', self.edit_parameters)
        a.setToolTip('Edit curves line and axes parameters')

        a = self.addAction(self._icon('filesave.svg'), 'Save',
                           self.save_figure)
        a.setToolTip('Save the figure')


        # Add the x,y location widget at the right side of the toolbar
        # The stretch factor is 1 which means any resizing of the toolbar
        # will resize this label instead of the buttons.
        if self.coordinates:
            self.locLabel = QtGui.QLabel( "", self )
            self.locLabel.setAlignment(
                QtCore.Qt.AlignRight | QtCore.Qt.AlignTop )
            self.locLabel.setSizePolicy(
                QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding,
                                  QtGui.QSizePolicy.Ignored))
            labelAction = self.addWidget(self.locLabel)
            labelAction.setVisible(True)

        # reference holder for subplots_adjust window
        self.adj_window = None

    def zoom(self, *args):
        super(GlueToolbar, self).zoom(self, *args)
        self.update_boxes()

    def pan(self, *args):
        super(GlueToolbar, self).pan(self, *args)
        self.update_boxes()

    def add_mode(self, mode):
        receiver = lambda: self._custom_mode(mode)
        action = self.addAction(mode.icon, mode.action_text, receiver)
        action.setToolTip(mode.tool_tip)
        action.setCheckable(True)
        self.buttons[mode.mode_id] = action

    def _custom_mode(self, mode):
        if self._active == mode.mode_id:
            self._active = None
        else:
            self._active = mode.mode_id

        if self._idPress is not None:
            self._idPress = self.canvas.mpl_disconnect(self._idPress)
            self.mode = ''
        if self._idRelease is not None:
            self._idRelease = self.canvas.mpl_disconnect(self._idRelease)
            self.mode = ''
        if self._custom_idMove is not None:
            self._custom_idMove = self.canvas.mpl_disconnect(
                self._custom_idMove)
            self.mode = ''

        if self._active:
            self._idPress = self.canvas.mpl_connect(
                'button_press_event', mode.press)
            self._custom_idMove = self.canvas.mpl_connect(
                'motion_notify_event', mode.move)
            self._idRelease = self.canvas.mpl_connect(
                'button_release_event', mode.release)
            self.mode = mode.action_text
            self.canvas.widgetlock(self)
        else:
            self.canvas.widgetlock.release(self)

        for a in self.canvas.figure.get_axes():
            a.set_navigate_mode(None)

        self.set_message(self.mode)
        self.update_boxes()

    def update_boxes(self):
        for mode in self.buttons:
            self.buttons[mode].setChecked(self._active == mode)


