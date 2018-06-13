from __future__ import absolute_import, division, print_function

from glue.core.subset import SliceSubsetState

__all__ = ['PixelSubsetState']


class PixelSubsetState(SliceSubsetState):
    def copy(self):
        return PixelSubsetState(self.reference_data, self.slices)
