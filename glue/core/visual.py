from matplotlib.colors import ColorConverter, Colormap

from glue.config import settings, colormaps

from echo import callback_property, HasCallbackProperties

from glue.utils.matplotlib import MATPLOTLIB_GE_36

if MATPLOTLIB_GE_36:
    from matplotlib import colormaps as cmap
else:
    from matplotlib.cm import get_cmap

# Define acceptable line styles
VALID_LINESTYLES = ['solid', 'dashed', 'dash-dot', 'dotted', 'none']

__all__ = ['VisualAttributes']


class VisualAttributes(HasCallbackProperties):

    """
    This class is used to define visual attributes for any kind of objects.

    Parameters
    ----------
    parent : `QObject`, optional
        The object that this visual attributes object is attached to. Default is `None`.
    color : `str`, optional
        A matplotlib color string. Default is set from :class:`~glue.config.SettingRegistry`.
    alpha : `float`, optional
        Opacity, between 0-1. Default is set from :class:`~glue.config.SettingRegistry`.
    preferred_cmap : `str` or :class:`~matplotlib.colors.Colormap`, optional
        A colormap to be used as the preferred colormap, by name or instance. Default is `None`.
    linewidth : `float`, optional
        The linewidth. Default is 1.
    linestyle : `str`, optional
        The linestyle. Default is `'solid'`.
    marker : `str`, optional
        The matplotlib marker shape. Default is `'o'`.
    markersize : `float`, optional
        The size of the marker. Default is 3.
    """

    DEFAULT_ATTS = ['color', 'alpha', 'linewidth', 'linestyle', 'marker',
                    'markersize', 'preferred_cmap']

    def __init__(self, parent=None, color=None, alpha=None, preferred_cmap=None, linewidth=1, linestyle='solid', marker='o', markersize=3):

        super(VisualAttributes, self).__init__()

        # We have to set the defaults here, otherwise the settings are fixed
        # once the class is defined.
        color = color or settings.DATA_COLOR
        alpha = alpha or settings.DATA_ALPHA

        self.parent = parent
        self._atts = self.DEFAULT_ATTS.copy()
        self.color = color
        self.alpha = alpha
        self.preferred_cmap = preferred_cmap
        self.linewidth = linewidth
        self.linestyle = linestyle
        self.marker = marker
        self.markersize = markersize

    def __eq__(self, other):
        if not isinstance(other, VisualAttributes):
            return False
        elif self is other:
            return True
        else:
            return all(getattr(self, a) == getattr(other, a) for a in self._atts)

    # If __eq__ is defined, then __hash__ has to be re-defined
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
        if isinstance(value, str):
            self._color = value.lower()
        else:
            self._color = value

    @callback_property
    def preferred_cmap(self):
        """
        A preferred colormap specified using Matplotlib notation
        """
        return self._preferred_cmap

    @preferred_cmap.setter
    def preferred_cmap(self, value):
        if isinstance(value, str):
            try:
                self._preferred_cmap = cmap[value] if MATPLOTLIB_GE_36 else get_cmap(value)
            except (ValueError, KeyError):
                # This checks for the formal name of the colormap.
                # e.g., 'viridis' is 'Viridis'
                for element in colormaps.members:
                    if element[0] == value:
                        self._preferred_cmap = element[1]
                        break
                else:
                    # If the string name fails to be validated
                    raise ValueError(f"{value} is not a valid colormap name.")
        elif isinstance(value, Colormap) or value is None:
            self._preferred_cmap = value
        else:
            raise TypeError("`preferred_cmap` must be a string or an instance of a matplotlib.colors.Colormap")

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
                       'alpha', 'parent', 'marker', 'markersize',
                       'preferred_cmap'])
        if attribute not in allowed and not attribute.startswith('_'):
            raise Exception("Attribute %s does not exist" % attribute)

        changed = getattr(self, attribute, None) != value
        super(VisualAttributes, self).__setattr__(attribute, value)

        # if parent has a broadcast method, broadcast the change
        if (changed and hasattr(self, 'parent') and
            hasattr(self.parent, 'broadcast') and
                attribute != 'parent' and not attribute.startswith('_')):
            self.parent.broadcast('style')
