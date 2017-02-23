from __future__ import absolute_import, division, print_function

from qtpy import compat
from glue import config
from glue.core import Subset
from glue.core.subset import MaskSubsetState


def import_subset_mask(data_or_subset, data_collection):

    subset_mask_importers = {}
    for e in config.subset_mask_importer:
        if e.extension == '':
            fltr = "{0} (*)".format(e.label)
        else:
            fltr = "{0} ({1})".format(e.label, ' '.join('*.' + ext for ext in e.extension))
        subset_mask_importers[fltr] = e.function

    filters = ';;'.join(sorted(subset_mask_importers))

    print(filters)

    filename, fltr = compat.getopenfilename(caption="Choose a subset mask file",
                                            filters=filters)

    filename = str(filename)
    if not filename:
        return

    masks = subset_mask_importers[fltr](filename)

    # Make sure shape is unique
    shapes = set(mask.shape for mask in masks.values())
    if len(shapes) == 0:
        raise ValueError("No subset masks were returned")
    elif len(shapes) > 1:
        raise ValueError("Not all subsets have the same shape")

    if list(shapes)[0] != data_or_subset.shape:
        raise ValueError("Mask shape(s) {0} does not match data shape {1}".format(list(shapes)[0], data_or_subset.shape))

    if isinstance(data_or_subset, Subset):

        subset = data_or_subset

        if len(masks) != 1:
            raise ValueError("Can only read in a single subset when importing into a subset")

        mask = list(masks.values())[0]

        subset_state = MaskSubsetState(mask, subset.pixel_component_ids)
        subset.subset_state = subset_state

    else:

        data = data_or_subset

        for label, mask in masks.items():

            subset_state = MaskSubsetState(mask, data.pixel_component_ids)
            data_collection.new_subset_group(label=label, subset_state=subset_state)
