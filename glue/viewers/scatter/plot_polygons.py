"""
Some lightly edited code from geopandas.plotting.py for efficiently
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
import shapely
from matplotlib.collections import PatchCollection
from shapely.errors import GeometryTypeError


class UpdateableRegionCollection(PatchCollection):
    """
    Allow paths in PatchCollection to be modified after creation.
    """

    def __init__(self, patches, *args, **kwargs):
        self.patches = patches
        PatchCollection.__init__(self, patches, *args, **kwargs)

    def get_paths(self):
        self.set_paths(self.patches)
        return self._paths


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

    geom_types = get_geometry_type(geoms).astype("str")

    if (
        not np.char.startswith(geom_types, prefix).any()
        # and not geoms.is_empty.any()
        # and not geoms.isna().any()
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


def transform_shapely(func, geom):
    """
    A simplified/modified version of shapely.ops.transform where the func
    call signature is tuned for the coordinate transform functions
    coming from glue.
    """
    if geom.is_empty:
        return geom
    if geom.geom_type in ("Point", "LineString", "LinearRing", "Polygon"):
        if geom.geom_type in ("Point", "LineString", "LinearRing"):
            return type(geom)(func(geom.coords))
        elif geom.geom_type == "Polygon":
            shell = type(geom.exterior)(func(geom.exterior.coords))
            holes = list(
                type(ring)(func(ring.coords))
                for ring in geom.interiors
            )
            return type(geom)(shell, holes)

    elif geom.geom_type.startswith("Multi") or geom.geom_type == "GeometryCollection":
        return type(geom)([transform_shapely(func, part) for part in geom.geoms])
    else:
        raise GeometryTypeError(f"Type {geom.geom_type!r} not recognized")


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
