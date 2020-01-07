import numpy as np
from numpy.testing import assert_equal

from glue.tests.helpers import requires_scipy

from ..geometry import polygon_line_intersections, floodfill


def test_square_nonclosed():

    x = [0, 2, 2, 0]
    y = [-1, -1, 3, 3]

    assert polygon_line_intersections(x, y, xval=-0.1) == []
    assert polygon_line_intersections(x, y, xval=0) == [(-1, 3)]
    assert polygon_line_intersections(x, y, xval=1) == [(-1, 3)]
    assert polygon_line_intersections(x, y, xval=2) == [(-1, 3)]
    assert polygon_line_intersections(x, y, xval=2.1) == []

    assert polygon_line_intersections(x, y, yval=-1.1) == []
    assert polygon_line_intersections(x, y, yval=-1) == []
    assert polygon_line_intersections(x, y, yval=0) == [(0, 2)]
    assert polygon_line_intersections(x, y, yval=1) == [(0, 2)]
    assert polygon_line_intersections(x, y, yval=2) == [(0, 2)]
    assert polygon_line_intersections(x, y, yval=3) == []
    assert polygon_line_intersections(x, y, yval=3.1) == []


def test_polygon():

    x = [0, 0, 2, 2, 1, 1, 3, 3, 2, 0]
    y = [2, 4, 4, 3, 3, 2, 2, 1, 0, 2]

    assert polygon_line_intersections(x, y, xval=-0.1) == []
    assert polygon_line_intersections(x, y, xval=+0.0) == []
    assert polygon_line_intersections(x, y, xval=+0.5) == [(1.5, 4)]
    assert polygon_line_intersections(x, y, xval=+1.0) == [(1, 2), (3, 4)]
    assert polygon_line_intersections(x, y, xval=+1.5) == [(0.5, 2), (3, 4)]
    assert polygon_line_intersections(x, y, xval=+2.0) == [(0, 2)]
    assert polygon_line_intersections(x, y, xval=+2.5) == [(0.5, 2)]
    assert polygon_line_intersections(x, y, xval=+3.0) == []
    assert polygon_line_intersections(x, y, xval=+3.1) == []

    assert polygon_line_intersections(x, y, yval=-0.1) == []
    assert polygon_line_intersections(x, y, yval=+0.0) == []
    assert polygon_line_intersections(x, y, yval=+0.5) == [(1.5, 2.5)]
    assert polygon_line_intersections(x, y, yval=+1.0) == [(1, 3)]
    assert polygon_line_intersections(x, y, yval=+1.5) == [(0.5, 3)]
    assert polygon_line_intersections(x, y, yval=+2.0) == [(0, 1), (1, 3)]
    assert polygon_line_intersections(x, y, yval=+2.5) == [(0, 1)]
    assert polygon_line_intersections(x, y, yval=+3.0) == [(0, 1), (1, 2)]
    assert polygon_line_intersections(x, y, yval=+3.1) == [(0, 2)]
    assert polygon_line_intersections(x, y, yval=+3.5) == [(0, 2)]
    assert polygon_line_intersections(x, y, yval=+4.0) == [(0, 2)]
    assert polygon_line_intersections(x, y, yval=+4.1) == []


DATA = np.array([[9, 6, 2, 3],
                 [4, 5, 2, 5],
                 [2, 4, 1, 0],
                 [5, 6, 0, -1]])


@requires_scipy
def test_floodfill():

    result = floodfill(DATA, (1, 0), 1.01)

    assert_equal(result, [[0, 0, 0, 0],
                          [1, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]])

    result = floodfill(DATA, (1, 0), 1.3)

    assert_equal(result, [[0, 0, 0, 0],
                          [1, 1, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 0, 0]])

    result = floodfill(DATA, (1, 0), 1.6)

    assert_equal(result, [[0, 1, 1, 1],
                          [1, 1, 1, 1],
                          [1, 1, 0, 0],
                          [1, 1, 0, 0]])

    result = floodfill(DATA, (1, 0), 2.1)

    assert_equal(result, [[0, 1, 1, 1],
                          [1, 1, 1, 1],
                          [1, 1, 1, 1],
                          [1, 1, 1, 0]])

    result = floodfill(DATA, (1, 0), 3)

    assert_equal(result, [[1, 1, 1, 1],
                          [1, 1, 1, 1],
                          [1, 1, 1, 1],
                          [1, 1, 1, 1]])
