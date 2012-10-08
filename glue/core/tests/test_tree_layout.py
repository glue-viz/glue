import numpy as np

import pytest

from .. import tree_layout as tl
from ..tree import NewickTree


def test_invalid_input():
    with pytest.raises(TypeError) as exc:
        layout = tl.TreeLayout(None)
    assert exc.value.args[0] == 'Input not a tree object: %s' % type(None)


def test_layout_indexable_by_tree_or_id():
    tree = NewickTree('((0,1)4,(2,3)5)6;')
    layout = tl.TreeLayout(tree)
    assert layout[tree] is layout[tree.id]


def test_default_layout():
    tree = NewickTree('((0,1)4,(2,3)5)6;')
    layout = tl.TreeLayout(tree)

    t6 = tree
    t4, t5 = t6.children
    t0, t1 = t4.children
    t2, t3 = t5.children

    ts = [t0, t1, t2, t3, t4, t5, t6]
    xs = [-1.5, -.5, .5, 1.5, -1, 1, 0]
    ys = [2, 2, 2, 2, 1, 1, 0]

    for t, x, y in zip(ts, xs, ys):
        assert layout[t].x == x
        assert layout[t].y == y
        assert layout[t].width == 1
        assert layout[t].height == 0


def test_layout_single_leaf():
    tree = NewickTree('0;')
    layout = tl.TreeLayout(tree)
    assert layout[tree].x == 0
    assert layout[tree].y == 0


def test_pick():

    tree = NewickTree('((0,1)4,(2,3)5)6;')
    layout = tl.TreeLayout(tree)

    #exact match
    assert layout.pick(0, 0) is tree

    #closest match, below
    assert layout.pick(0, -1) is tree

    #only pick if y position is <= node
    assert layout.pick(-.01, .01) is tree.children[0]
    assert layout.pick(0, 2.1) is None


def test_tree_to_xy():
    tree = NewickTree('(0,1)2;')
    layout = tl.TreeLayout(tree)

    x = np.array([0, 0, -.5, -.5, None, 0, .5, .5, None], dtype=float)
    y = np.array([0, 0, 0, 1, None, 0, 0, 1, None], dtype=float)

    xx, yy = layout.tree_to_xy(tree)

    np.testing.assert_array_almost_equal(x, np.array(xx, dtype=float))
    np.testing.assert_array_almost_equal(y, np.array(yy, dtype=float))


def test_tree_to_xy_list():
    tree = NewickTree('(0,1)2;')
    layout = tl.TreeLayout(tree)

    x = np.array([-0.5, None, .5, None], dtype=float)
    y = np.array([1, None, 1, None], dtype=float)

    xx, yy = layout.tree_to_xy(tree.children)

    np.testing.assert_array_almost_equal(x, np.array(xx, dtype=float))
    np.testing.assert_array_almost_equal(y, np.array(yy, dtype=float))


def test_branch_to_xy_branch():
    tree = NewickTree('((0,1)4,(2,3)5)6;')
    layout = tl.TreeLayout(tree)

    x = [-1, -1, 0]
    y = [1, 0, 0]

    xx, yy = layout.branch_to_xy(tree.children[0])

    np.testing.assert_array_almost_equal(x, xx)
    np.testing.assert_array_almost_equal(y, yy)


def test_branch_to_xy_root():
    tree = NewickTree('((0,1)4,(2,3)5)6;')
    layout = tl.TreeLayout(tree)

    x = [0]
    y = [0]
    xx, yy = layout.branch_to_xy(tree)
    np.testing.assert_array_almost_equal(x, xx)
    np.testing.assert_array_almost_equal(y, yy)


def test_branch_to_xy_leaf():
    tree = NewickTree('((0,1)4,(2,3)5)6;')
    layout = tl.TreeLayout(tree)

    x = [-1.5, -1.5, -1]
    y = [2, 1, 1]
    xx, yy = layout.branch_to_xy(tree.children[0].children[0])
    np.testing.assert_array_almost_equal(x, xx)
    np.testing.assert_array_almost_equal(y, yy)


def test_branch_to_xy_list():
    tree = NewickTree('((0,1)4,(2,3)5)6;')
    layout = tl.TreeLayout(tree)
    x = np.array([0, None, 0, None], dtype=float)
    y = np.array([0, None, 0, None], dtype=float)

    xx, yy = layout.branch_to_xy([tree, tree])
    np.testing.assert_array_almost_equal(x, np.array(xx, dtype=float))
    np.testing.assert_array_almost_equal(y, np.array(yy, dtype=float))
