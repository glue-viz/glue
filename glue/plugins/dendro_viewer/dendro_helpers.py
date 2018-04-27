from __future__ import absolute_import, division, print_function

import numpy as np


def dendrogram_layout(parent, height, key):

    children = _dendro_children(parent)
    pos = np.zeros(key.size) - 1
    cur_pos = 0

    for struct in _iter_sorted(children, parent, key):
        if children[struct].size == 0:  # leaf
            pos[struct] = cur_pos
            cur_pos += 1
        else:  # branch
            assert pos[children[struct]].mean() >= 0
            pos[struct] = pos[children[struct]].mean()

    layout = np.zeros((2, 3 * height.size))
    layout[0, ::3] = pos
    layout[0, 1::3] = pos
    layout[0, 2::3] = np.where(parent >= 0, pos[parent], np.nan)

    layout[1, ::3] = height
    layout[1, 1::3] = np.where(parent >= 0, height[parent], height.min())
    layout[1, 2::3] = layout[1, 1::3]

    return layout


def _substructures(parent, idx):
    """
    Return an array of all substructure indices of a given index.
    The input is included in the output.

    Parameters
    ----------
    idx : int
        The structure to extract.

    Returns
    -------
    array
    """
    children = _dendro_children(parent)
    result = []
    if np.isscalar(idx):
        todo = [idx]
    else:
        todo = idx.tolist()

    while todo:
        result.append(todo.pop())
        todo.extend(children[result[-1]])
    return np.array(result, dtype=np.int)


def _dendro_children(parent):
    children = [[] for _ in range(parent.size)]
    for i, p in enumerate(parent):
        if p < 0:
            continue
        children[p].append(i)
    return list(map(np.asarray, children))


def _iter_sorted(children, parent, key):
    # must yield both children before parent
    yielded = set()
    trunks = np.array([i for i, p in enumerate(parent) if p < 0], dtype=np.int)
    for idx in np.argsort(key[trunks]):
        idx = trunks[idx]
        for item in _postfix_iter(idx, children, parent, yielded, key):
            yield item


def _postfix_iter(node, children, parent, yielded, key):
    """
    Iterate over a node and its children, in the following fashion:

    parents are yielded after children
    children are yielded in order of ascending key value
    """

    todo = [node]
    expanded = set()

    while todo:
        node = todo[-1]

        if node in yielded:
            todo.pop()
            continue

        if children[node].size == 0 or node in expanded:
            yield todo.pop()
            yielded.add(node)
            continue

        c = children[node]
        ind = np.argsort(key[c])[::-1]
        todo.extend(c[ind])
        expanded.add(node)
