# Define acceptable line styles
valid_linestyles = ['solid', 'dashed', 'dash-dot', 'dotted', 'none']

#assign colors so as to avoid repeats
default_colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00',
                  '#A65628', '#F781BF']
washout_colors = ['#F6B4B4', '#BDD5E8', '#C4E5C3', '#DDC5E1', '#FFD5AB',
                  '#E2C7B8', '#FCD5EA']
color_pos = 0


class VisualAttributes(object):
    '''
    This class is used to define visual attributes for any kind of objects
    '''

    def __init__(self, parent=None, washout=False):

        # Color can be specified using Matplotlib notation. Specifically, it
        # can be:
        #  * A string with a common color (e.g. 'black', 'red', 'orange')
        #  * A string containing a float in the rng [0:1] for a shade of
        #    gray ('0.0' = black,'1.0' = white)
        #  * A tuple of three floats in the rng [0:1] for (R, G, B)
        #  * An HTML hexadecimal string (e.g. '#eeefff')
        global color_pos

        col = washout_colors if washout else default_colors
        self.color = col[color_pos % len(default_colors)]
        color_pos += 1
        self.alpha = 1.

        # Line width in points (float or int)
        self.linewidth = 1

        # Line style, which can be one of 'solid', 'dashed', 'dash-dot',
        # 'dotted', or 'none'
        self.linestyle = 'solid'

        self.marker = 'o'
        self.markersize = 40
        self.label = None

        self.parent = parent

    def set(self, other):
        self.color = other.color
        self.alpha = other.alpha
        self.linewidth = other.linewidth
        self.linestyle = other.linestyle
        self.marker = other.marker
        self.markersize = other.markersize

    def __setattr__(self, attribute, value):

        # Check that line style is valid
        if attribute == 'linestyle' and value not in valid_linestyles:
            raise Exception("Line style should be one of %s" %
                            '/'.join(valid_linestyles))

        # Check that line width is valid
        if attribute == 'linewidth':
            if type(value) not in [float, int]:
                raise Exception("Line width should be a float or an int")
            if value < 0:
                raise Exception("Line width should be positive")

        # Check that the attribute exists (don't allow new attributes)
        allowed = set(['color', 'linewidth', 'linestyle',
                       'alpha', 'parent', 'marker', 'markersize', 'label'])
        if attribute not in allowed:
            raise Exception("Attribute %s does not exist" % attribute)

        object.__setattr__(self, attribute, value)

        # if parent has a broadcast method, broadcast the change
        if hasattr(self, 'parent') and hasattr(self.parent, 'broadcast'):
            self.parent.broadcast(self)
