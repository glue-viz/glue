from __future__ import absolute_import, division, print_function

from glue.config import settings
from glue.external.echo import CallbackProperty
from glue.external import six

# Define acceptable line styles
VALID_LINESTYLES = ['solid', 'dashed', 'dash-dot', 'dotted', 'none']

__all__ = ['VisualAttributes']


class VisualAttributes(object):

    '''
    This class is used to define visual attributes for any kind of objects

    The essential attributes of a VisualAttributes instance are:

    :param color: A matplotlib color string
    :param alpha: Opacity (0-1)
    :param linewidth: The linewidth (float or int)
    :param linestyle: The linestyle (``'solid' | 'dashed' | 'dash-dot' | 'dotted' | 'none'``)
    :param marker: The matplotlib marker shape (``'o' | 's' | '^' | etc``)
    :param markersize: The size of the marker (int)

    '''

    # Color can be specified using Matplotlib notation. Specifically, it
    # can be:
    #  * A string with a common color (e.g. 'black', 'red', 'orange')
    #  * A string containing a float in the rng [0:1] for a shade of
    #    gray ('0.0' = black,'1.0' = white)
    #  * A tuple of three floats in the rng [0:1] for (R, G, B)
    # * An HTML hexadecimal string (e.g. '#eeefff')
    color = CallbackProperty(settings.DATA_COLOR,
                             docstring="Color using Matplotlib notation.")
    alpha = CallbackProperty(0.5, docstring=("Transparency, given as a floating "
                                             "point value between 0 and 1."))

    # Line width in points (float or int)
    linewidth = CallbackProperty(1., docstring="Line width in points.")

    # Line style, which can be one of 'solid', 'dashed', 'dash-dot',
    # 'dotted', or 'none'
    linestyle = CallbackProperty('solid', docstring="Line style.")

    marker = CallbackProperty('o', docstring="Marker symbol.")
    markersize = CallbackProperty(3, docstring="Marker size.")

    def __init__(self, parent=None, washout=False, color=settings.DATA_COLOR):
        self.parent = parent
        self._atts = ['color', 'alpha', 'linewidth', 'linestyle', 'marker',
                      'markersize']
        self.color = color

    # TODO: fix equality comparison

    def __eq__(self, other):
        if not isinstance(other, VisualAttributes):
            return False
        elif self is other:
            return True
        else:
            return all(getattr(self, a) == getattr(other, a) for a in self._atts)

    # In Python 3, if __eq__ is defined, then __hash__ has to be re-defined
    if six.PY3:
        __hash__ = object.__hash__

    def set(self, other):
        """
        Update this instance's properties based on another VisualAttributes instance.
        """
        for att in self._atts:
            setattr(self, att, getattr(other, att))

    def copy(self, new_parent=None):
        """
        Create a new instance with the same visual properties
        """
        result = VisualAttributes()
        result.set(self)
        if new_parent is not None:
            result.parent = new_parent
        return result

    def __setattr__(self, attribute, value):

        # Check that line style is valid
        if attribute == 'linestyle' and value not in VALID_LINESTYLES:
            raise Exception("Line style should be one of %s" %
                            '/'.join(VALID_LINESTYLES))

        # Check that line width is valid
        if attribute == 'linewidth':
            if type(value) not in [float, int]:
                raise Exception("Line width should be a float or an int")
            if value < 0:
                raise Exception("Line width should be positive")

        # Check that the attribute exists (don't allow new attributes)
        allowed = set(['color', 'linewidth', 'linestyle',
                       'alpha', 'parent', 'marker', 'markersize', '_atts'])
        if attribute not in allowed:
            raise Exception("Attribute %s does not exist" % attribute)

        changed = getattr(self, attribute, None) != value
        object.__setattr__(self, attribute, value)

        # if parent has a broadcast method, broadcast the change
        if (changed and hasattr(self, 'parent') and
            hasattr(self.parent, 'broadcast') and
                attribute != 'parent'):
            self.parent.broadcast('style')
