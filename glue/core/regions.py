"""
Functions to support data that defines regions
"""
import numpy as np

from glue.core.roi import PolygonalROI
from glue.core.data_region import RegionData

from glue.config import layer_action
from glue.core.subset import RoiSubsetState, MultiOrState


def reg_to_roi(reg):
    if reg.geom_type == "Polygon":
        ext_coords = np.array(reg.exterior.coords.xy)
        roi = PolygonalROI(vx=ext_coords[0], vy=ext_coords[1])  # Need to account for interior rings
    return roi


@layer_action(label='Subset of regions -> Subset over region extent', single=True, subset=True)
def layer_to_subset(layer, data_collection):
    """
    This should be limited to the case where subset.Data is RegionData
    and/or return a warning when applied to some other kind of data.
    """
    if isinstance(layer.data, RegionData):

        extended_comp = layer.data._extended_component_ids[0]
        regions = layer[extended_comp]
        list_of_rois = [reg_to_roi(region) for region in regions]

        roisubstates = [RoiSubsetState(layer.data.ext_x,
                                       layer.data.ext_y,
                                       roi=roi
                                       )
                        for roi in list_of_rois]
        if len(list_of_rois) > 1:
            composite_substate = MultiOrState(roisubstates)
        else:
            composite_substate = roisubstates[0]
        _ = data_collection.new_subset_group(subset_state=composite_substate)
