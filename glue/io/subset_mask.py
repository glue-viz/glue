from collections import OrderedDict

from glue.core import Subset
from glue.core.subset import MaskSubsetState

__all__ = ['SubsetMaskImporter', 'SubsetMaskExporter']


class SubsetMaskImporter(object):

    def get_filename_and_reader(self):
        raise NotImplementedError

    def run(self, data_or_subset, data_collection):

        filename, reader = self.get_filename_and_reader()

        if filename is None:
            return

        # Read in the masks
        masks = reader(filename)

        # Make sure shape is unique
        shapes = set(mask.shape for mask in masks.values())

        if len(shapes) == 0:
            raise ValueError("No subset masks were returned")

        elif len(shapes) > 1:
            raise ValueError("Not all subsets have the same shape")

        if list(shapes)[0] != data_or_subset.shape:
            raise ValueError("Mask shape {0} does not match data shape {1}".format(list(shapes)[0], data_or_subset.shape))

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


class SubsetMaskExporter(object):

    def get_filename_and_writer(self):
        raise NotImplementedError

    def run(self, data_or_subset):

        filename, writer = self.get_filename_and_writer()

        if filename is None:
            return

        # Prepare dictionary of masks
        masks = OrderedDict()

        if isinstance(data_or_subset, Subset):

            subset = data_or_subset
            masks[subset.label] = subset.to_mask()

        else:

            data = data_or_subset

            if len(data.subsets) == 0:
                raise ValueError("Data has no subsets")

            for subset in data.subsets:
                masks[subset.label] = subset.to_mask()

        writer(filename, masks)
