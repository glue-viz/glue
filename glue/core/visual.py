from __future__ import absolute_import, division, print_function

from matplotlib.colors import ColorConverter

from glue.config import settings
from glue.external.echo import callback_property
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

    def __init__(self, parent=None, washout=False, color=None, alpha=None):

        # We have to set the defaults here, otherwise the settings are fixed
        # once the class is defined.
        color = color or settings.DATA_COLOR
        alpha = alpha or settings.DATA_ALPHA

        self.parent = parent
        self._atts = ['color', 'alpha', 'linewidth', 'linestyle', 'marker',
                      'markersize']
        self.color = color
        self.alpha = alpha
        self.linewidth = 1
        self.linestyle = 'solid'
        self.marker = 'o'
        self.markersize = 3

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

    @callback_property
    def color(self):
        """
        Color specified using Matplotlib notation

        Specifically, it can be:

         * A string with a common color (e.g. 'black', 'red', 'orange')
         * A string containing a float in the rng [0:1] for a shade of
           gray ('0.0' = black,'1.0' = white)
         * A tuple of three floats in the rng [0:1] for (R, G, B)
         * An HTML hexadecimal string (e.g. '#eeefff')
        """
        return self._color

    @color.setter
    def color(self, value):
        if isinstance(value, six.string_types):
            self._color = value.lower()
        else:
            self._color = value

    @callback_property
    def alpha(self):
        """
        Transparency, given as a floating point value between 0 and 1.
        """
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value

    @property
    def rgba(self):
        r, g, b = ColorConverter().to_rgb(self.color)
        return (r, g, b, self.alpha)

    @callback_property
    def linestyle(self):
        """
        The line style, which can be one of 'solid', 'dashed', 'dash-dot',
        'dotted', or 'none'.
        """
        return self._linestyle

    @linestyle.setter
    def linestyle(self, value):
        if value not in VALID_LINESTYLES:
            raise Exception("Line style should be one of %s" %
                            '/'.join(VALID_LINESTYLES))
        self._linestyle = value

    @callback_property
    def linewidth(self):
        """
        The line width, in points.
        """
        return self._linewidth

    @linewidth.setter
    def linewidth(self, value):
        if type(value) not in [float, int]:
            raise Exception("Line width should be a float or an int")
        if value < 0:
            raise Exception("Line width should be positive")
        self._linewidth = value

    @callback_property
    def marker(self):
        """
        The marker symbol.
        """
        return self._marker

    @marker.setter
    def marker(self, value):
        self._marker = value

    @callback_property
    def markersize(self):
        return self._markersize

    @markersize.setter
    def markersize(self, value):
        self._markersize = int(value)

    def __setattr__(self, attribute, value):

        # Check that the attribute exists (don't allow new attributes)
        allowed = set(['color', 'linewidth', 'linestyle',
                       'alpha', 'parent', 'marker', 'markersize'])
        if attribute not in allowed and not attribute.startswith('_'):
            raise Exception("Attribute %s does not exist" % attribute)

        changed = getattr(self, attribute, None) != value
        super(VisualAttributes, self).__setattr__(attribute, value)

        # if parent has a broadcast method, broadcast the change
        if (changed and hasattr(self, 'parent') and
            hasattr(self.parent, 'broadcast') and
                attribute != 'parent' and not attribute.startswith('_')):
            self.parent.broadcast('style')
