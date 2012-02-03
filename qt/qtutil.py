from matplotlib.colors import ColorConverter
from PyQt4.QtGui import QColor

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
    

