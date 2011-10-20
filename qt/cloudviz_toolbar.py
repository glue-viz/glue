import os

import matplotlib
from matplotlib.backends.backend_qt4 import NavigationToolbar2QT
from PyQt4 import QtCore, QtGui, Qt

from cloudviz.roi import MplCircleTool

class CloudvizToolbar(NavigationToolbar2QT):
    def _init_toolbar(self):
        self.basedir = os.path.join(matplotlib.rcParams[ 'datapath' ],'images')
        self.buttons = {}

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
        self.buttons['move'] = a

        a = self.addAction(self._icon('zoom_to_rect.svg'), 'Zoom', self.zoom)
        a.setToolTip('Zoom to rectangle')
        a.setCheckable(True)
        self.buttons['zoom'] = a

        a = self.addAction(QIcon('icons/circle.png'), 'Circle', self.circle)
        a.setToolTip('Modify current subset using circular selection')
        self.setCheckable(True)
        self.buttons['circle'] = a

        a = self.addAction(QIcon('icons/square.png'), 'Box', self.box)
        a.setToolTip('Modify current subset using box selection')
        self.setCheckable(True)
        self.buttons['box'] = a

        a = self.addAction(QIcon('icons/lasso.png'), 'Lasso', self.lasso)
        a.setToolTip('Modify current subset using lasso selection')
        self.setCheckable(True)
        self.buttons['lasso'] = a
        
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
        super(CustomToolbar, self).zoom(self, *args)
        self.update_boxes()

    def pan(self, *args):
        super(CustomToolbar, self).pan(self, *args)
        self.update_boxes()

    def circle(self, *args):
        'activate circle selection mode'
        if self_active == 'CIRCLE':
            self._active = None
        else:
            self._active = 'CIRCLE'

        if self._idPress is not None:
            self._idPress = self.canvas.mpl_disconnect(self._idPress)
            self.mode = ''

        if self._idRelease is not None:
            self._idRelease = self.mpl_disconnect(self._idRelease)
            self.mode = ''

        if self._active:
            self._idPress = self.canvas.mpl_connect('button_press_event', self.press_circle)
            self._idRelease = self.canvas.mpl_connect('button_release_event', self.release_circle)
            self.mode = 'circle'
            self.canvas.widgetlock(self)
        else:
            self.canvas.widgetlock.release(self)

        for i in self.canvas.figure.get_axes():
            a.set_navigate_mode(self._active)

        self.set_message(self.mode)

    def box(self, *args):
        pass

    def lasso(self, *args):
        pass

    def press_circle(self, event):
        'the press mouse button in circle mode callback'
        if event.button == 1:
            self._button_pressed=1
        elif  event.button == 3:
            self._button_pressed=3
        else:
            self._button_pressed=None
            return

        self._roi = MplCirculeTool(self.canvas.figure.get_axes())
        self._roi.connect()
        self.press(event)

    def update_boxes(self):
        self.buttons['zoom'].setChecked(self._active == 'ZOOM')
        self.buttons['move'].setChecked(self._active == 'PAN')
        self.buttons['circle'].setChecked(self._active == 'CIRCLE')
        self.buttons['box'].setChecked(self._active == 'BOX')
        self.buttons['lasso'].setChecked(self._active == 'LASSO')
