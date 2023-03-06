"""
A lightly modified version of geopandas.plotting.py for efficiently 
plotting multiple polygons in matplotlib.
"""
# Copyright (c) 2013-2022, GeoPandas developers.
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of GeoPandas nor the names of its contributors may
#   be used to endorse or promote products derived from this software without
#   specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection


import shapely
from shapely.geometry.base import BaseMultipartGeometry

from packaging.version import Version

class UpdatablePatchCollection(PatchCollection):
    """
    Allow properties of PatchCollection to be modified after creation.
    """

    def __init__(self, patches, *args, **kwargs):
        self.patches = patches
        PatchCollection.__init__(self, patches, *args, **kwargs)

    def get_paths(self):
        self.set_paths(self.patches)
        return self._paths

class UpdateableRegionCollection(UpdatablePatchCollection):

    def set_centers(self, x, y):
        if len(x) == 0:
            self.patches = []
            return
        offsets = np.vstack((x, y)).transpose()
        self.patches.set_offsets(offsets)


def _sanitize_geoms(geoms, prefix="Multi"):
    """
    Returns Series like geoms and index, except that any Multi geometries
    are split into their components and indices are repeated for all component
    in the same Multi geometry. At the same time, empty or missing geometries are
    filtered out.  Maintains 1:1 matching of geometry to value.
    Prefix specifies type of geometry to be flatten. 'Multi' for MultiPoint and similar,
    "Geom" for GeometryCollection.
    Returns
    -------
    components : list of geometry
    component_index : index array
        indices are repeated for all components in the same Multi geometry
    """
    # TODO(shapely) look into simplifying this with
    # shapely.get_parts(geoms, return_index=True) from shapely 2.0
    components, component_index = [], []

    geom_types = get_geometry_type(geoms).astype('str')

    if (
        not np.char.startswith(geom_types, prefix).any()
        #and not geoms.is_empty.any()
        #and not geoms.isna().any()
    ):
        return geoms, np.arange(len(geoms))

    for ix, (geom, geom_type) in enumerate(zip(geoms, geom_types)):
        if geom is not None and geom_type.startswith(prefix):
            for poly in geom.geoms:
                components.append(poly)
                component_index.append(ix)
        elif geom is None:
            continue
        else:
            components.append(geom)
            component_index.append(ix)

    return components, np.array(component_index)



def get_geometry_type(data):
    _names = {
    "MISSING": None,
    "NAG": None,
    "POINT": "Point",
    "LINESTRING": "LineString",
    "LINEARRING": "LinearRing",
    "POLYGON": "Polygon",
    "MULTIPOINT": "MultiPoint",
    "MULTILINESTRING": "MultiLineString",
    "MULTIPOLYGON": "MultiPolygon",
    "GEOMETRYCOLLECTION": "GeometryCollection",
}

    type_mapping = {p.value: _names[p.name] for p in shapely.GeometryType}
    geometry_type_ids = list(type_mapping.keys())
    geometry_type_values = np.array(list(type_mapping.values()), dtype=object)
    res = shapely.get_type_id(data)
    return geometry_type_values[np.searchsorted(geometry_type_ids, res)]


def plot_series(
    s, ax, cmap=None, color=None, empty=False, **style_kwds
):
    """
    Plot a GeoSeries.
    Generate a plot of a GeoSeries geometry with matplotlib.
    Parameters
    ----------
    s : Series
        The GeoSeries to be plotted. Currently Polygon,
        MultiPolygon, LineString, MultiLineString, Point and MultiPoint
        geometries can be plotted.
    ax : matplotlib.pyplot.Artist
        axes on which to draw the plot
    cmap : str (default None)
        The name of a colormap recognized by matplotlib. Any
        colormap will work, but categorical colormaps are
        generally recommended. Examples of useful discrete
        colormaps include:
            tab10, tab20, Accent, Dark2, Paired, Pastel1, Set1, Set2
    color : str, np.array, pd.Series, List (default None)
        If specified, all objects will be colored uniformly.
    **style_kwds : dict
        Color options to be passed on to the actual plot function, such
        as ``linewidth``, ``alpha``.
    Returns
    -------
    ax : matplotlib axes instance
    """

    # have colors been given for all geometries?
    color_given = pd.api.types.is_list_like(color) and len(color) == len(s)

    # if cmap is specified, create range of colors based on cmap
    values = None
    if cmap is not None:
        values = np.arange(len(s))
        if hasattr(cmap, "N"):
            values = values % cmap.N
        style_kwds["vmin"] = style_kwds.get("vmin", values.min())
        style_kwds["vmax"] = style_kwds.get("vmax", values.max())

    # decompose GeometryCollections
    s_geometry = get_geometry_type(s)
    geoms, multiindex = _sanitize_geoms(s, prefix="Geom")
    values = np.take(values, multiindex, axis=0) if cmap else None
    # ensure indexes are consistent
    if color_given and isinstance(color, pd.Series):
        color = color.reindex(s.index)
    expl_color = np.take(color, multiindex, axis=0) if color_given else color
    expl_series = geoms

    geom_types = get_geometry_type(expl_series)
    poly_idx = np.asarray((geom_types == "Polygon") | (geom_types == "MultiPolygon"))
    line_idx = np.asarray(
        (geom_types == "LineString")
        | (geom_types == "MultiLineString")
        | (geom_types == "LinearRing")
    )
    point_idx = np.asarray((geom_types == "Point") | (geom_types == "MultiPoint"))

    # plot all Polygons and all MultiPolygon components in the same collection
    polys = expl_series[poly_idx]
    color_ = expl_color[poly_idx] if color_given else color

    values_ = values[poly_idx] if cmap else None
    _plot_polygon_collection(
        ax, polys, values_, color=color_, cmap=cmap, empty=empty, **style_kwds
    )
    return ax


def _plot_polygon_collection(
    ax, geoms, values=None, color=None, cmap=None, vmin=None, vmax=None, empty=False, **kwargs
):
    """
    Plots a collection of Polygon and MultiPolygon geometries to `ax`
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        where shapes will be plotted
    geoms : a sequence of `N` Polygons and/or MultiPolygons (can be mixed)
    values : a sequence of `N` values, optional
        Values will be mapped to colors using vmin/vmax/cmap. They should
        have 1:1 correspondence with the geometries (not their components).
        Otherwise plot using `color`.
    color : single color or sequence of `N` colors
        Sets both `edgecolor` and `facecolor`
    cmap : str (default None)
        The name of a colormap recognized by matplotlib.
    vmin : float
    vmax : float
    empty : bool
        Whether to plot just the outline of the polygons 
    **kwargs
        Additional keyword arguments passed to the collection
    Returns
    -------
    collection : matplotlib.collections.Collection that was plotted
    """
    from matplotlib.collections import PatchCollection

    geoms, multiindex = _sanitize_geoms(geoms)
    if values is not None:
        values = np.take(values, multiindex, axis=0)

    # PatchCollection does not accept some kwargs.
    kwargs = {
        att: value
        for att, value in kwargs.items()
        if att not in ["markersize", "marker"]
    }

    # Add to kwargs for easier checking below.
    if color is not None:
        kwargs["color"] = color

    _expand_kwargs(kwargs, multiindex)

    collection = PatchCollection([_PolygonPatch(poly) for poly in geoms], **kwargs)

    if values is not None:
        collection.set_array(np.asarray(values))
        collection.set_cmap(cmap)
        if "norm" not in kwargs:
            collection.set_clim(vmin, vmax)

    if empty:
        collection.set_facecolor('none')
    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection

def _PolygonPatch(polygon, **kwargs):
    """Constructs a matplotlib patch from a Polygon geometry
    The `kwargs` are those supported by the matplotlib.patches.PathPatch class
    constructor. Returns an instance of matplotlib.patches.PathPatch.
    Example (using Shapely Point and a matplotlib axes)::
        b = shapely.geometry.Point(0, 0).buffer(1.0)
        patch = _PolygonPatch(b, fc='blue', ec='blue', alpha=0.5)
        ax.add_patch(patch)
    GeoPandas originally relied on the descartes package by Sean Gillies
    (BSD license, https://pypi.org/project/descartes) for PolygonPatch, but
    this dependency was removed in favor of the below matplotlib code.
    """
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    path = Path.make_compound_path(
        Path(np.asarray(polygon.exterior.coords)[:, :2]),
        *[Path(np.asarray(ring.coords)[:, :2]) for ring in polygon.interiors],
    )
    return PathPatch(path, **kwargs)

def _expand_kwargs(kwargs, multiindex):
    """
    Most arguments to the plot functions must be a (single) value, or a sequence
    of values. This function checks each key-value pair in 'kwargs' and expands
    it (in place) to the correct length/formats with help of 'multiindex', unless
    the value appears to already be a valid (single) value for the key.
    """
    import matplotlib
    from matplotlib.colors import is_color_like
    from typing import Iterable

    mpl = Version(matplotlib.__version__)
    if mpl >= Version("3.4"):
        # alpha is supported as array argument with matplotlib 3.4+
        scalar_kwargs = ["marker", "path_effects"]
    else:
        scalar_kwargs = ["marker", "alpha", "path_effects"]

    for att, value in kwargs.items():
        if "color" in att:  # color(s), edgecolor(s), facecolor(s)
            if is_color_like(value):
                continue
        elif "linestyle" in att:  # linestyle(s)
            # A single linestyle can be 2-tuple of a number and an iterable.
            if (
                isinstance(value, tuple)
                and len(value) == 2
                and isinstance(value[1], Iterable)
            ):
                continue
        elif att in scalar_kwargs:
            # For these attributes, only a single value is allowed, so never expand.
            continue

        if pd.api.types.is_list_like(value):
            kwargs[att] = np.take(value, multiindex, axis=0)

