"""
This module provides some routines for performing layout
calculations to organize rectangular windows in a larger canvas
"""

from collections import Counter


class Rectangle(object):

    def __init__(self, x, y, w, h):
        """ A rectangle (obviously).

        :param x: Left edge
        :param y: Bottom edge
        :param w: Width
        :param h: Height
        """
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def __eq__(self, other):
        return (self.x == other.x and
                self.y == other.y and
                self.w == other.w and
                self.h == other.h)

    # If __eq__ is defined, then __hash__ has to be re-defined
    __hash__ = object.__hash__

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "Rectangle(%f, %f, %f, %f)" % (self.x, self.y, self.w, self.h)

    def snap(self, xstep, ystep=None, padding=0.0):
        """
        Snap the rectangle onto a grid, with optional padding.

        :param xstep: The number of intervals to split the x=[0, 1] range into.
        :param ystep: The number of intervals to split the y=[0, 1] range into.
        :param padding: Uniform padding to add around the result. This shrinks
                        the result so that the edges + padding line up with the
                        grid.

        :returns: A new Rectangle, obtained by snapping self onto the grid,
                  and applying padding
        """

        if ystep is None:
            ystep = xstep

        return Rectangle(round(self.x * xstep) / xstep + padding,
                         round(self.y * ystep) / ystep + padding,
                         round(self.w * xstep) / xstep - 2 * padding,
                         round(self.h * ystep) / ystep - 2 * padding)


def _snap_size(rectangles):
    x = Counter([round(1 / r.w) for r in rectangles])
    y = Counter([round(1 / r.h) for r in rectangles])
    return x.most_common()[0][0], y.most_common()[0][0]


def snap_to_grid(rectangles, padding=0.0):
    """
    Snap a collection of rectangles onto a grid, in a sensible fashion

    :param rectangles: List of Rectangle instances
    :returns: A dictionary mapping each input rectangle to a snapped position
    """
    result = {}
    xs, ys = _snap_size(rectangles)
    for r in rectangles:
        result[r] = r.snap(xs, ys, padding=padding)
    return result
