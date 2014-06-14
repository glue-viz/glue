"""
This module provides some routines for performing layout
calculations to organize rectangular windows in a larger canvas
"""
from collections import Counter
from operator import mul
from itertools import permutations, chain

# pre-specified layouts for 1-5 boxes
layouts = {
    1: (((0, 0, 1, 1),),),
    2: (
        ((0, 0, 1, .5), (0, .5, 1, .5)),
        ((0, 0, .5, 1), (.5, 0, .5, 1)),
    ),
    3: (
        ((0, .5, 1, .5), (0, 0, .5, .5), (.5, 0, .5, .5)),
        ((0, 0, 1, .5), (0, .5, .5, .5), (.5, .5, .5, .5)),
        ((0, 0, 1, .5), (0, .5, .5, .5), (.5, .5, .5, .5)),
        ((0, 0, .5, .5), (0, .5, .5, .5), (.5, 0, .5, 1)),
        ((0, 0, 1 / 3., 1), (1 / 3., 0, 1 / 3., 1), (2 / 3., 0, 2 / 3., 1)),
        ((0, 0, 1, 1 / 3.), (0, 1 / 3., 1, 1 / 3.), (0, 2 / 3., 1, 1 / 3.)),
    ),
    4: (
        ((0, 0, .5, .5), (.5, 0, .5, .5), (0, .5, .5, .5), (.5, .5, .5, .5)),
        ((0, .5, 1, .5), (0, 0, 1 / 3., .5), (1 / 3., 0, 1 / 3., .5), (2 / 3., 0, 1 / 3., .5)),
        ((0, 0, 1, .5), (0, .5, 1 / 3., .5), (1 / 3., .5, 1 / 3., .5), (2 / 3., .5, 1 / 3., .5)),
        ((0, 0, .5, 1), (.5, 0, .5, 1 / 3.), (.5, 1 / 3., .5, 1 / 3.), (.5, 2 / 3., .5, 2 / 3.)),
        ((.5, 0, .5, 1), (0, 0, .5, 1 / 3.), (0, 1 / 3., .5, 1 / 3.), (0, 2 / 3., .5, 2 / 3.)),
    ),
    5: (
        ((0, .5, .5, .5), (.5, .5, .5, .5), (0, 0, 1 / 3., .5), (1 / 3., 0, 1 / 3., .5), (2 / 3., 0, 1 / 3., .5)),
        ((0, 0, .5, .5), (.5, 0, .5, .5), (0, .5, 1 / 3., .5), (1 / 3., .5, 1 / 3., .5), (2 / 3., .5, 1 / 3., .5)),
        ((0, 0, .5, .5), (0, .5, .5, .5), (.5, 0, .5, 1 / 3.), (.5, 1 / 3., .5, 1 / 3.), (.5, 2 / 3., .5, 1 / 3.)),
        ((.5, 0, .5, .5), (.5, .5, .5, .5), (0, 0, .5, 1 / 3.), (0, 1 / 3., .5, 1 / 3.), (0, 2 / 3., .5, 1 / 3.)),
    )
}


def _overlap(layout, rectangles):
    layout = [Rectangle(*l) for l in layout]
    return reduce(mul, (a.overlap_area(b)
                  for a, b in zip(layout, rectangles)))


def _choose_best_preset(mapping):
    listing = list(mapping.items())
    rectangles = [l[1] for l in listing]

    nr = len(rectangles)

    if nr not in layouts:
        raise ValueError("No standard layouts for %i rectangles" % nr)
    options = layouts[nr]
    result = max(chain(*(permutations(o) for o in options)),
                 key=lambda o: _overlap(o, rectangles))
    result = [Rectangle(*r) for r in result]
    result = dict((input, output) for (input, _), output in
                  zip(listing, result))
    return result


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

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "Rectangle(%f, %f, %f, %f)" % (self.x, self.y, self.w, self.h)

    def snap(self, xstep, ystep=None):
        """
        Snap the rectangle onto a grid, with optional padding.

        :param xstep: The number of intervals to split the x=[0, 1] range into.
        :param ystep: The number of intervals to split the y=[0, 1] range into.

        :returns: A new Rectangle, obtained by snapping self onto the grid,
                  and applying padding
        """

        if ystep is None:
            ystep = xstep

        return Rectangle(round(self.x * xstep) / xstep,
                         round(self.y * ystep) / ystep,
                         round(self.w * xstep) / xstep,
                         round(self.h * ystep) / ystep)

    def overlaps(self, other, tol=1e-2):
        return ((self.r - other.l) > tol and
                (self.l - other.r) < -tol and
                (self.t - other.b) > tol and
                (self.b - other.t) < -tol)

    def overlap_area(self, other):
        dw = max(min(self.r, other.r) - max(self.l, other.l), 0)
        dh = max(min(self.t, other.t) - max(self.b, other.b), 0)
        return dw * dh

    def pad(self, padding):
        self.x += padding
        self.w -= 2 * padding
        self.y += padding
        self.h -= 2 * padding

    @property
    def l(self):
        return self.x

    @property
    def r(self):
        return self.x + self.w

    @property
    def b(self):
        return self.y

    @property
    def t(self):
        return self.y + self.h


def _snap_size(rectangles):
    x = Counter([round(1 / r.w) for r in rectangles])
    y = Counter([round(1 / r.h) for r in rectangles])
    return x.most_common()[0][0], y.most_common()[0][0]


def _unoverlap(rectangles, padding=0.0):
    """
    Rearrange a rectangle mapping to avoid overlaps
    """
    s = sorted(rectangles, key=lambda x: x.l)
    for i, r1 in enumerate(s):
        r1 = rectangles[r1]
        for r2 in s[:i]:
            r2 = rectangles[r2]
            if r1.overlaps(r2):
                r1.x = r2.r + 2 * padding

    return rectangles


def _pad(rectangles, padding):
    for r in rectangles.values():
        r.pad(padding)
    return rectangles


def snap_to_grid(rectangles, padding=0.0):
    """
    Snap a collection of rectangles onto a grid, in a sensible fashion

    :param rectangles: List of Rectangle instances
    :param padding: Uniform padding to add around the result. This shrinks
                    the result so that the edges + padding line up with the
                    grid.

    :returns: A dictionary mapping each input rectangle to a snapped position
    """
    result = {}
    xs, ys = _snap_size(rectangles)
    for r in rectangles:
        result[r] = r.snap(xs, ys)
    try:
        result = _choose_best_preset(result)
    except ValueError:
        pass

    result = _unoverlap(result)
    result = _pad(result, padding)
    return result
