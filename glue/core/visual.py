from __future__ import absolute_import, division, print_function

from ..config import settings

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

    def __init__(self, parent=None, washout=False, color=settings.DATA_COLOR):

        # Color can be specified using Matplotlib notation. Specifically, it
        # can be:
        #  * A string with a common color (e.g. 'black', 'red', 'orange')
        #  * A string containing a float in the rng [0:1] for a shade of
        #    gray ('0.0' = black,'1.0' = white)
        #  * A tuple of three floats in the rng [0:1] for (R, G, B)
        # * An HTML hexadecimal string (e.g. '#eeefff')
        self.color = color
        self.alpha = .5

        # Line width in points (float or int)
        self.linewidth = 1

        # Line style, which can be one of 'solid', 'dashed', 'dash-dot',
        # 'dotted', or 'none'
        self.linestyle = 'solid'

        self.marker = 'o'
        self.markersize = 3

        self.parent = parent

        self._atts = ['color', 'alpha', 'linewidth', 'linestyle', 'marker',
                      'markersize']

    def __eq__(self, other):
        if not isinstance(other, VisualAttributes):
            return False
        return all(getattr(self, a) == getattr(other, a) for a in self._atts)

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

    def __eq__(self, other):
        return all(getattr(self, att) == getattr(other, att)
                   for att in self._atts)

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
