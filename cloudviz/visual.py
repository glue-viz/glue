import string

# Define acceptable line styles
valid_linestyles = ['solid', 'dashed', 'dash-dot', 'dotted', 'none']


class VisualAttributes(object):
    '''
    This class is used to define visual attributes for any kind of objects
    '''

    def __init__(self):

        # Color can be specified using Matplotlib notation. Specifically, it
        # can be:
        #  * A string with a common color (e.g. 'black', 'red', 'orange')
        #  * A string containing a float in the range [0:1] for a shade of
        #    gray ('0.0' = black,'1.0' = white)
        #  * A tuple of three floats in the range [0:1] for (R, G, B)
        #  * An HTML hexadecimal string (e.g. '#eeefff')
        self.color = 'black'

        # Line width in points (float or int)
        self.linewidth = 1

        # Line style, which can be one of 'solid', 'dashed', 'dash-dot',
        # 'dotted', or 'none'
        self.linestyle = 'solid'

    def __setattr__(self, attribute, value):

        # Check that line style is valid
        if attribute == 'linestyle' and value not in valid_linestyles:
            raise Exception("Line style should be one of %s" %
                            string.join(valid_linestyles, '/'))

        # Check that line width is valid
        if attribute == 'linewidth':
            if type(value) not in [float, int]:
                raise Exception("Line width should be a float or an int")
            if value < 0:
                raise Exception("Line width should be positive")

        # Check that the attribute exists (don't allow new attributes)
        if attribute not in ['color', 'linewidth', 'linestyle']:
            raise Exception("Attribute %s does not exist" % value)
