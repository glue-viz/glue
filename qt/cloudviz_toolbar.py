import os

import matplotlib
from matplotlib.backends.backend_qt4 import NavigationToolbar2QT
from PyQt4 import QtCore, QtGui, Qt
from PyQt4.QtGui import QIcon

from cloudviz.roi import MplCircularROI, MplRectangularROI, MplPolygonalROI
import cloudviz as cv

class CloudvizToolbar(NavigationToolbar2QT):
    def __init__(self, parent):
        super(CloudvizToolbar, self).__init__(parent.canvas, parent.frame)
        self.parent = parent

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
        a.setCheckable(True)
        self.buttons['circle'] = a
        
        a = self.addAction(QIcon('icons/square.png'), 'Box', self.box)
        a.setToolTip('Modify current subset using box selection')
        a.setCheckable(True)
        self.buttons['box'] = a

        a = self.addAction(QIcon('icons/lasso.png'), 'Lasso', self.lasso)
        a.setToolTip('Modify current subset using lasso selection')
        a.setCheckable(True)
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
        super(CloudvizToolbar, self).zoom(self, *args)
        self.update_boxes()

    def pan(self, *args):
        super(CloudvizToolbar, self).pan(self, *args)
        self.update_boxes()

    def shape_button(self, mode, roi, **kwargs):
        if not isinstance(self.parent.active_layer, cv.subset.RoiSubset):
            return

        'activate circle selection mode'
        if self._active == mode.upper():
            self._active = None
            if self.roi: 
                self.roi.disconnect()
                self.roi = None
        else:
            self._active = mode.upper()

        if self._idPress is not None:
            self._idPress = self.canvas.mpl_disconnect(self._idPress)
            self.mode = ''

        if self._idRelease is not None:
            self._idRelease = self.canvas.mpl_disconnect(self._idRelease)
            self.mode = ''

        if self._active:
            subset = self.parent.active_layer
            ax, = self.canvas.figure.get_axes()
            self.roi = roi(ax, subset=subset, **kwargs)
            self._idPress = self.roi._press
            self._idMotion = self.roi._motion
            self._idRelease = self.roi._release

            self.mode = mode.lower()
            self.canvas.widgetlock(self)
        else:
            self.canvas.widgetlock.release(self)

        self.set_message(self.mode)
        self.update_boxes()

    def circle(self, *args):
        self.shape_button('circle', MplCircularROI)

    def box(self, *args):
        self.shape_button('box', MplRectangularROI)

    def lasso(self, *args):
        self.shape_button('lasso', MplPolygonalROI, lasso=True)

    def update_boxes(self):
        self.buttons['zoom'].setChecked(self._active == 'ZOOM')
        self.buttons['move'].setChecked(self._active == 'PAN')
        self.buttons['circle'].setChecked(self._active == 'CIRCLE')
        self.buttons['box'].setChecked(self._active == 'BOX')
        self.buttons['lasso'].setChecked(self._active == 'LASSO')
        
    def deselect_roi(self):
        self.buttons['circle'].setChecked(False)        
        self.buttons['box'].setChecked(False)
        self.buttons['lasso'].setChecked(False)
        if self._active in ['CIRCLE', 'BOX' 'LASSO']:
            if self._idPress is not None:
                self._idPress = self.canvas.mpl_disconnect(self._idPress)
            if self._idMotion is not None:
                self._idMotion = self.canvas.mpl_disconnect(self._idMotion)
            if self._idRelease is not None:
                self._idRelease = self.canvas.mpl_disconnect(self._idRelease)
            self._active = None
        
    def set_roi_enabled(self, state):
        if not state:
            self.deselect_roi()
        self.buttons['circle'].setDisabled(not state)
        self.buttons['box'].setDisabled(not state)
        self.buttons['lasso'].setDisabled(not state)
        

