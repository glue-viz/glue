"""
A plot to visualize trees
"""
import numpy as np

from ..core.client import Client
from ..core.data import Data, IncompatibleAttribute
from ..core.callback_property import CallbackProperty, add_callback, delay_callback
from .viz_client import init_mpl
from ..core.util import nonpartial
from ..core.roi import PointROI

from .layer_artist import DendroLayerArtist, LayerArtistContainer
from ..core.subset import CategorySubsetState
from ..core.edit_subset_mode import EditSubsetMode


class GenericMplClient(Client):

    """
    This client base class handles the logic of adding, removing,
    and updating layers.

    Subsets are auto-added and removed with datasets.
    New subsets are auto-added iff the data has already been added
    """

    def __init__(self, data=None, figure=None, axes=None,
                 artist_container=None):

        super(GenericMplClient, self).__init__(data=data)
        figure, self.axes = init_mpl(figure, axes)
        self.artists = artist_container
        if self.artists is None:
            self.artists = LayerArtistContainer()

        self._connect()

    def _connect(self):
        pass

    @property
    def collect(self):
        # a better name
        return self.data

    def _redraw(self):
        self.axes.figure.canvas.draw()

    def new_layer_artist(self, layer):
        raise NotImplementedError

    def apply_roi(self, roi):
        raise NotImplementedError

    def _update_layer(self, layer):
        raise NotImplementedError

    def add_layer(self, layer):
        """
        Add a new Data or Subset layer to the plot.

        Returns the created layer artist

        :param layer: The layer to add
        :type layer: :class:`~glue.core.Data` or :class:`~glue.core.Subset`
        """
        if layer.data not in self.collect:
            return

        if layer in self.artists:
            return self.artists[layer][0]

        result = self.new_layer_artist(layer)
        self.artists.append(result)
        self._update_layer(layer)

        self.add_layer(layer.data)
        for s in layer.data.subsets:
            self.add_layer(s)

        if layer.data is layer:  # Added Data object. Relimit view
            self.axes.autoscale_view(True, True, True)

        return result

    def remove_layer(self, layer):
        if layer not in self.artists:
            return

        self.artists.pop(layer)
        if isinstance(layer, Data):
            map(self.remove_layer, layer.subsets)

        self._redraw()

    def set_visible(self, layer, state):
        """
        Toggle a layer's visibility

        :param layer: which layer to modify
        :param state: True or False
        """

    def _update_all(self):
        for layer in self.artists.layers:
            self._update_layer(layer)

    def __contains__(self, layer):
        return layer in self.artists

    # Hub message handling
    def _add_subset(self, message):
        self.add_layer(message.sender)

    def _remove_subset(self, message):
        self.remove_layer(message.sender)

    def _update_subset(self, message):
        self._update_layer(message.sender)

    def _update_data(self, message):
        self._update_layer(message.sender)

    def _remove_data(self, message):
        self.remove_layer(message.data)


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

        super(DendroClient, self).add_layer(layer)
        self.display_data = layer.data
        self._default_attributes()

    def _update_layer(self, layer):

        for artist in self.artists[layer]:
            if not isinstance(artist, DendroLayerArtist):
                continue
            artist.layout = self._layout
            artist.update()

        self._redraw()

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

        :param idx: The structure to extract. Int
        :returns: array
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


def _dendro_children(parent):
    children = [[] for _ in range(parent.size)]
    for i, p in enumerate(parent):
        if p < 0:
            continue
        children[p].append(i)
    return map(np.asarray, children)


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
