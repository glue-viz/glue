"""
A plot to visualize trees
"""

from __future__ import absolute_import, division, print_function

import numpy as np

from glue.core.edit_subset_mode import EditSubsetMode
from glue.core.state import lookup_class_with_patches
from glue.core.subset import CategorySubsetState
from glue.core.roi import PointROI
from glue.core.callback_property import CallbackProperty, add_callback, delay_callback
from glue.core.data import IncompatibleAttribute, Data
from glue.viewers.common.viz_client import GenericMplClient
from glue.plugins.dendro_viewer.layer_artist import DendroLayerArtist
from glue.utils import nonpartial


class DendroClient(GenericMplClient):

    height_attr = CallbackProperty()
    parent_attr = CallbackProperty()
    order_attr = CallbackProperty()
    ylog = CallbackProperty(False)
    display_data = CallbackProperty(None)
    select_substruct = CallbackProperty(True)

    def __init__(self, *args, **kwargs):
        super(DendroClient, self).__init__(*args, **kwargs)
        self._layout = None
        self.axes.set_xticks([])
        self.axes.spines['top'].set_visible(False)
        self.axes.spines['bottom'].set_visible(False)

    def _connect(self):
        add_callback(self, 'ylog', self._set_ylog)
        add_callback(self, 'height_attr', nonpartial(self._relayout))
        add_callback(self, 'order_attr', nonpartial(self._relayout))
        add_callback(self, 'parent_attr', nonpartial(self._relayout))

    def _default_attributes(self):
        assert self.display_data is not None

        fallback = self.display_data.components[0]
        with delay_callback(self, 'height_attr', 'parent_attr',
                            'order_attr'):

            if self.height_attr is None:
                comp = self.display_data.find_component_id('height') or fallback
                self.height_attr = comp

            if self.parent_attr is None:
                comp = self.display_data.find_component_id('parent') or fallback
                self.parent_attr = comp

            if self.order_attr is None:
                self.order_attr = self.height_attr

    def new_layer_artist(self, layer):
        return DendroLayerArtist(layer, self.axes)

    def _set_ylog(self, log):
        self.axes.set_yscale('log' if log else 'linear')
        self._redraw()

    def _relayout(self):

        if self.display_data is None:
            return
        if self.height_attr is None:
            return
        if self.parent_attr is None:
            return
        if self.order_attr is None:
            return

        try:
            parent = np.asarray(self.display_data[self.parent_attr],
                                dtype=np.int).ravel()
            y = self.display_data[self.height_attr].ravel()
            key = self.display_data[self.order_attr].ravel()
        except IncompatibleAttribute:
            return

        children = self._children
        pos = np.zeros(key.size) - 1
        cur_pos = 0

        for struct in _iter_sorted(children, parent, key):
            if children[struct].size == 0:  # leaf
                pos[struct] = cur_pos
                cur_pos += 1
            else:  # branch
                assert pos[children[struct]].mean() >= 0
                pos[struct] = pos[children[struct]].mean()

        layout = np.zeros((2, 3 * y.size))
        layout[0, ::3] = pos
        layout[0, 1::3] = pos
        layout[0, 2::3] = np.where(parent >= 0, pos[parent], np.nan)

        layout[1, ::3] = y
        layout[1, 1::3] = np.where(parent >= 0, y[parent], y.min())
        layout[1, 2::3] = layout[1, 1::3]

        self._layout = layout
        self._snap_limits()
        self._update_all()

    def _snap_limits(self):
        if self._layout is None:
            return

        x, y = self._layout[:, ::3]
        xlim = np.array([x.min(), x.max()])
        xpad = .05 * xlim.ptp()
        xlim[0] -= xpad
        xlim[1] += xpad

        ylim = np.array([y.min(), y.max()])
        if self.ylog:
            ylim = np.maximum(ylim, 1e-5)
            pad = 1.05 * ylim[1] / ylim[0]
            ylim[0] /= pad
            ylim[1] *= pad
        else:
            pad = .05 * ylim.ptp()
            ylim[0] -= pad
            ylim[1] += pad

        self.axes.set_xlim(*xlim)
        self.axes.set_ylim(*ylim)

    def add_layer(self, layer):

        if layer.data.ndim != 1:
            return

        if layer.data not in self.data:
            raise TypeError("Layer not in data collection")

        if layer in self.artists:
            return self.artists[layer][0]

        self.display_data = self.display_data or layer.data

        result = DendroLayerArtist(layer, self.axes)
        self.artists.append(result)
        self._update_layer(layer)
        self._ensure_subsets_added(layer)

        self._default_attributes()

        return result

    def _ensure_subsets_added(self, layer):
        if not isinstance(layer, Data):
            return
        for subset in layer.subsets:
            self.add_layer(subset)

    def _update_layer(self, layer):

        for artist in self.artists[layer]:
            if not isinstance(artist, DendroLayerArtist):
                continue
            artist.layout = self._layout
            artist.update()

        self._redraw()

    def remove_layer(self, layer):
        super(DendroClient, self).remove_layer(layer)
        if layer is self.display_data:
            self.display_data = None

    @property
    def _parents(self):
        return np.asarray(self.display_data[self.parent_attr],
                          dtype=np.int).ravel()

    @property
    def _children(self):
        children = _dendro_children(self._parents)
        return children

    def _substructures(self, idx):
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
        children = self._children
        result = []
        todo = [idx]

        while todo:
            result.append(todo.pop())
            todo.extend(children[result[-1]])
        return np.array(result, dtype=np.int)

    def apply_roi(self, roi):

        if not isinstance(roi, PointROI):
            raise NotImplementedError("Only PointROI supported")

        if self._layout is None or self.display_data is None:
            return

        x, y = roi.x, roi.y
        if not roi.defined():
            return

        xs, ys = self._layout[:, ::3]
        parent_ys = self._layout[1, 1::3]

        delt = np.abs(x - xs)
        delt[y > ys] = np.nan
        delt[y < parent_ys] = np.nan

        if np.isfinite(delt).any():
            select = np.nanargmin(delt)
            if self.select_substruct:
                select = self._substructures(select)
            select = np.asarray(select, dtype=np.int)
        else:
            select = np.array([], dtype=np.int)

        state = CategorySubsetState(self.display_data.pixel_component_ids[0],
                                    select)

        EditSubsetMode().update(self.collect, state,
                                focus_data=self.display_data)

    def restore_layers(self, layers, context):
        """
        Re-generate a list of plot layers from a glue-serialized list
        """
        for l in layers:
            cls = lookup_class_with_patches(l.pop('_type'))
            if cls != DendroLayerArtist:
                raise ValueError("Dendrogram client cannot restore layer of type "
                                 "%s" % cls)
            props = dict((k, context.object(v)) for k, v in l.items())
            layer = self.add_layer(props['layer'])
            layer.properties = props


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
