from __future__ import absolute_import, division, print_function

from collections import defaultdict

import numpy as np

from glue.core import Data
from glue.config import colormaps
from glue.viewers.matplotlib.state import (MatplotlibDataViewerState,
                                           MatplotlibLayerState,
                                           DeferredDrawCallbackProperty as DDCProperty,
                                           DeferredDrawSelectionCallbackProperty as DDSCProperty)
from glue.core.state_objects import StateAttributeLimitsHelper
from glue.utils import defer_draw, view_shape, unbroadcast
from glue.external.echo import delay_callback
from glue.core.data_combo_helper import ManualDataComboHelper, ComponentIDComboHelper
from glue.core.exceptions import IncompatibleDataException, IncompatibleAttribute

__all__ = ['ImageViewerState', 'ImageLayerState', 'ImageSubsetLayerState', 'AggregateSlice']


class AggregateSlice(object):

    def __init__(self, slice=None, center=None, function=None):
        self.slice = slice
        self.center = center
        self.function = function

    def __gluestate__(self, context):
        state = dict(slice=context.do(self.slice),
                     center=self.center,
                     function=context.do(self.function))
        return state

    @classmethod
    def __setgluestate__(cls, rec, context):
        return cls(slice=context.object(rec['slice']),
                   center=rec['center'],
                   function=context.object(rec['function']))


class ImageViewerState(MatplotlibDataViewerState):
    """
    A state class that includes all the attributes for an image viewer.
    """

    x_att = DDCProperty(docstring='The component ID giving the pixel component '
                                  'shown on the x axis')
    y_att = DDCProperty(docstring='The component ID giving the pixel component '
                                  'shown on the y axis')
    x_att_world = DDSCProperty(docstring='The component ID giving the world component '
                                         'shown on the x axis', default_index=-1)
    y_att_world = DDSCProperty(docstring='The component ID giving the world component '
                                         'shown on the y axis', default_index=-2)
    aspect = DDSCProperty(0, docstring='Whether to enforce square pixels (``equal``) '
                                            'or fill the axes (``auto``)')
    reference_data = DDSCProperty(docstring='The dataset that is used to define the '
                                            'available pixel/world components, and '
                                            'which defines the coordinate frame in '
                                            'which the images are shown')
    slices = DDCProperty(docstring='The current slice along all dimensions')
    color_mode = DDSCProperty(0, docstring='Whether each layer can have '
                                                    'its own colormap (``Colormaps``) or '
                                                    'whether each layer is assigned '
                                                    'a single color (``One color per layer``)')

    dpi = DDCProperty(72, docstring='The resolution (in dots per inch) of density maps, if present')

    def __init__(self, **kwargs):

        super(ImageViewerState, self).__init__()

        self.limits_cache = {}

        # NOTE: we don't need to use StateAttributeLimitsHelper here because
        # we can simply call reset_limits below when x/y attributes change.
        # Using StateAttributeLimitsHelper makes things a lot slower.

        self.ref_data_helper = ManualDataComboHelper(self, 'reference_data')

        self.xw_att_helper = ComponentIDComboHelper(self, 'x_att_world',
                                                    numeric=False, categorical=False,
                                                    world_coord=True)

        self.yw_att_helper = ComponentIDComboHelper(self, 'y_att_world',
                                                    numeric=False, categorical=False,
                                                    world_coord=True)

        self.add_callback('reference_data', self._reference_data_changed, priority=1000)
        self.add_callback('layers', self._layers_changed, priority=1000)

        self.add_callback('x_att', self._on_xatt_change, priority=500)
        self.add_callback('y_att', self._on_yatt_change, priority=500)

        self.add_callback('x_att_world', self._update_att, priority=500)
        self.add_callback('y_att_world', self._update_att, priority=500)

        self.add_callback('x_att_world', self._on_xatt_world_change, priority=1000)
        self.add_callback('y_att_world', self._on_yatt_world_change, priority=1000)

        aspect_display = {'equal': 'Square Pixels', 'auto': 'Automatic'}
        ImageViewerState.aspect.set_choices(self, ['equal', 'auto'])
        ImageViewerState.aspect.set_display_func(self, aspect_display.get)

        ImageViewerState.color_mode.set_choices(self, ['Colormaps', 'One color per layer'])

        self.update_from_dict(kwargs)

    def reset_limits(self):

        if self.reference_data is None or self.x_att is None or self.y_att is None:
            return

        nx = self.reference_data.shape[self.x_att.axis]
        ny = self.reference_data.shape[self.y_att.axis]

        with delay_callback(self, 'x_min', 'x_max', 'y_min', 'y_max'):
            self.x_min = -0.5
            self.x_max = nx - 0.5
            self.y_min = -0.5
            self.y_max = ny - 0.5

    def _reference_data_changed(self, *args):
        # This signal can get emitted if just the choices but not the actual
        # reference data change, so we check here that the reference data has
        # actually changed
        if self.reference_data is not getattr(self, '_last_reference_data', None):
            self._last_reference_data = self.reference_data
            with delay_callback(self, 'x_att_world', 'y_att_world', 'slices'):
                self._update_combo_att()
                self._set_default_slices()

    def _layers_changed(self, *args):

        # The layers callback gets executed if anything in the layers changes,
        # but we only care about whether the actual set of 'layer' attributes
        # for all layers change.

        layers_data = self.layers_data
        layers_data_cache = getattr(self, '_layers_data_cache', [])

        if layers_data == layers_data_cache:
            return

        self._update_combo_ref_data()
        self._set_reference_data()
        self._update_syncing()

        self._layers_data_cache = layers_data

    def _update_syncing(self):

        # If there are multiple layers for a given dataset, we disable the
        # syncing by default.

        layer_state_by_data = defaultdict(list)

        for layer_state in self.layers:
            if isinstance(layer_state.layer, Data):
                layer_state_by_data[layer_state.layer].append(layer_state)

        for data, layer_states in layer_state_by_data.items():
            if len(layer_states) > 1:
                for layer_state in layer_states:
                    # Scatter layers don't have global_sync so we need to be
                    # careful here and make sure we return a default value
                    if getattr(layer_state, 'global_sync', False):
                        layer_state.global_sync = False

    def _update_combo_ref_data(self):
        self.ref_data_helper.set_multiple_data(self.layers_data)

    def _update_combo_att(self):
        with delay_callback(self, 'x_att_world', 'y_att_world'):
            if self.reference_data is None:
                self.xw_att_helper.set_multiple_data([])
                self.yw_att_helper.set_multiple_data([])
            else:
                self.xw_att_helper.set_multiple_data([self.reference_data])
                self.yw_att_helper.set_multiple_data([self.reference_data])

    def _update_priority(self, name):
        if name == 'layers':
            return 3
        elif name == 'reference_data':
            return 2
        elif name.endswith(('_min', '_max')):
            return 0
        else:
            return 1

    @defer_draw
    def _update_att(self, *args):
        # Need to delay the callbacks here to make sure that we get a chance to
        # update both x_att and y_att otherwise could end up triggering image
        # slicing with two pixel components that are the same.
        with delay_callback(self, 'x_att', 'y_att'):
            if self.x_att_world is not None:
                index = self.reference_data.world_component_ids.index(self.x_att_world)
                self.x_att = self.reference_data.pixel_component_ids[index]
            if self.y_att_world is not None:
                index = self.reference_data.world_component_ids.index(self.y_att_world)
                self.y_att = self.reference_data.pixel_component_ids[index]

    @defer_draw
    def _on_xatt_change(self, *args):
        if self.x_att is not None:
            self.x_att_world = self.reference_data.world_component_ids[self.x_att.axis]
        self.reset_limits()

    @defer_draw
    def _on_yatt_change(self, *args):
        if self.y_att is not None:
            self.y_att_world = self.reference_data.world_component_ids[self.y_att.axis]
        self.reset_limits()

    @defer_draw
    def _on_xatt_world_change(self, *args):
        if self.x_att_world is not None and self.x_att_world == self.y_att_world:
            world_ids = self.reference_data.world_component_ids
            if self.x_att_world == world_ids[-1]:
                self.y_att_world = world_ids[-2]
            else:
                self.y_att_world = world_ids[-1]

    @defer_draw
    def _on_yatt_world_change(self, *args):
        if self.y_att_world is not None and self.y_att_world == self.x_att_world:
            world_ids = self.reference_data.world_component_ids
            if self.y_att_world == world_ids[-1]:
                self.x_att_world = world_ids[-2]
            else:
                self.x_att_world = world_ids[-1]

    def _set_reference_data(self):
        if self.reference_data is None:
            for layer in self.layers:
                if isinstance(layer.layer, Data):
                    self.reference_data = layer.layer
                    return

    def _set_default_slices(self):
        # Need to make sure this gets called immediately when reference_data is changed
        if self.reference_data is None:
            self.slices = ()
        else:
            self.slices = (0,) * self.reference_data.ndim

    @property
    def numpy_slice_aggregation_transpose(self):
        """
        Returns slicing information usable by Numpy.

        This returns two objects: the first is an object that can be used to
        slice Numpy arrays and return a 2D array, and the second object is a
        boolean indicating whether to transpose the result.
        """
        if self.reference_data is None:
            return None
        slices = []
        agg_func = []
        for i in range(self.reference_data.ndim):
            if i == self.x_att.axis or i == self.y_att.axis:
                slices.append(slice(None))
                agg_func.append(None)
            else:
                if isinstance(self.slices[i], AggregateSlice):
                    slices.append(self.slices[i].slice)
                    agg_func.append(self.slices[i].function)
                else:
                    slices.append(self.slices[i])
        transpose = self.y_att.axis > self.x_att.axis
        return slices, agg_func, transpose

    @property
    def wcsaxes_slice(self):
        """
        Returns slicing information usable by WCSAxes.

        This returns an iterable of slices, and including ``'x'`` and ``'y'``
        for the dimensions along which we are not slicing.
        """
        if self.reference_data is None:
            return None
        slices = []
        for i in range(self.reference_data.ndim):
            if i == self.x_att.axis:
                slices.append('x')
            elif i == self.y_att.axis:
                slices.append('y')
            else:
                if isinstance(self.slices[i], AggregateSlice):
                    slices.append(self.slices[i].center)
                else:
                    slices.append(self.slices[i])
        return slices[::-1]

    def flip_x(self):
        """
        Flip the x_min/x_max limits.
        """
        with delay_callback(self, 'x_min', 'x_max'):
            self.x_min, self.x_max = self.x_max, self.x_min

    def flip_y(self):
        """
        Flip the y_min/y_max limits.
        """
        with delay_callback(self, 'y_min', 'y_max'):
            self.y_min, self.y_max = self.y_max, self.y_min


class BaseImageLayerState(MatplotlibLayerState):

    _viewer_callbacks_set = False
    _image_cache = None
    _pixel_cache = None

    def get_sliced_data_shape(self, view=None):

        if (self.viewer_state.reference_data is None or
            self.viewer_state.x_att is None or
                self.viewer_state.y_att is None):
            return None

        x_axis = self.viewer_state.x_att.axis
        y_axis = self.viewer_state.y_att.axis

        shape = self.viewer_state.reference_data.shape
        shape_slice = shape[y_axis], shape[x_axis]

        if view is None:
            return shape_slice
        else:
            return view_shape(shape_slice, view)

    def get_sliced_data(self, view=None):

        # Getting the sliced data can be computationally expensive in some cases
        # in particular when reprojecting data/subsets. To avoid recomputing
        # these in cases where it isn't necessary, for example if the reference
        # data is a spectral cube and the layer is a 2D mosaic, we set up a
        # cache at the end of this method, and we then set up callbacks to
        # reset the cache if any of the following properties change. We need
        # to set a very high priority so that this is the first thing to happen.
        # Note that we need to set up the callbacks here as the viewer_state is
        # not always set in the __init__, for example when loading up sessions.
        # We also need to make sure that the cache gets reset when the links
        # change or when the subset changes. This is taken care of by calling
        # reset_cache in the layer artist update() method, which gets called
        # for these cases.

        if not self._viewer_callbacks_set:
            self.viewer_state.add_callback('slices', self.reset_cache_from_slices,
                                           echo_old=True, priority=100000)
            self.viewer_state.add_callback('x_att', self.reset_cache, priority=100000)
            self.viewer_state.add_callback('y_att', self.reset_cache, priority=100000)
            if self.is_callback_property('attribute'):  # this isn't the case for subsets
                self.add_callback('attribute', self.reset_cache, priority=100000)
            self._viewer_callbacks_set = True

        if self._image_cache is not None:
            if view == self._image_cache['view']:
                return self._image_cache['image']

        # In the cache, we need to keep track of which slice indices should
        # cause the cache to be reset. By default, we assume that any changes
        # in slices should cause the cache to get reset, and in the reprojection
        # code below we then set up more specific conditions.
        reset_slices = True

        full_view, agg_func, transpose = self.viewer_state.numpy_slice_aggregation_transpose

        # The view should be that which should just be applied to the data
        # slice, not to all the dimensions of the data - thus it should have at
        # most two dimension

        if view is not None:

            if len(view) > 2:
                raise ValueError('view should have at most two elements')
            if len(view) == 1:
                view = view + [slice(None)]

            x_axis = self.viewer_state.x_att.axis
            y_axis = self.viewer_state.y_att.axis

            full_view[x_axis] = view[1]
            full_view[y_axis] = view[0]

        # First, check whether the data is simply the reference data - if so
        # we can just use _get_image (which assumed alignment with reference_data)
        # to get the image to use.

        if self.layer.data is self.viewer_state.reference_data:
            image = self._get_image(view=full_view)
        else:

            # Second, we check whether the current data is linked pixel-wise with
            # the reference data.

            order = self.layer.data.pixel_aligned_data.get(self.viewer_state.reference_data)

            if order is not None:

                # order gives the order of the pixel components of the reference
                # data in the current data. With this we adjust the view and then
                # check that the result is a 2D array - if not, it means for example
                # that the layer is a 2D image and the reference data is a 3D cube
                # and that we are not slicing one of the dimensions in the 3D cube
                # that is also in the 2D image, resulting in a 1D array (which it
                # doesn't make sense to show.

                full_view = [full_view[idx] for idx in order]
                image = self._get_image(view=full_view)

                if image.ndim != 2:
                    raise IncompatibleDataException()
                else:
                    # Now check whether we need to transpose the image - we need
                    # to update this since the previously defined ``tranpose``
                    # value assumed data in the order of the reference data
                    x_axis = self.viewer_state.x_att.axis
                    y_axis = self.viewer_state.y_att.axis
                    transpose = order.index(x_axis) < order.index(y_axis)

            else:

                # Now the real fun begins! The pixel grids are not lined up. Fun
                # times!

                # Let's make sure there are no AggregateSlice variables in
                # the view as we can't deal with this currently
                if any(isinstance(v, AggregateSlice) for v in full_view):
                    raise IncompatibleDataException()
                else:
                    agg_func = None

                # Start off by finding all the pixel coordinates of the current
                # view in the reference frame of the current layer data. In
                # principle we could do something as simple as:
                #
                #   pixel_coords = [self.viewer_state.reference_data[pix, full_view]
                #                   for pix in self.layer.pixel_component_ids]
                #   coords = [np.round(p.ravel()).astype(int) for p in pixel_coords]
                #
                # However this is sub-optimal because in reality some of these
                # pixel coordinate conversions won't change when the view is
                # changed (e.g. when a slice index changes). We therefore
                # cache each transformed pixel coordinate.

                if self._pixel_cache is None:
                    # The cache hasn't been set yet or has been reset so we
                    # initialize it here.
                    self._pixel_cache = {'reset_slices': [None] * self.layer.ndim,
                                         'coord': [None] * self.layer.ndim,
                                         'shape': [None] * self.layer.ndim,
                                         'view': None}

                coords = []

                sub_data_view = [slice(0, 2)] * self.viewer_state.reference_data.ndim

                for ipix, pix in enumerate(self.layer.pixel_component_ids):

                    if self._pixel_cache['view'] != view or self._pixel_cache['coord'][ipix] is None:

                        # Start off by finding all the pixel coordinates of the current
                        # view in the reference frame of the current layer data.
                        pixel_coord = self.viewer_state.reference_data[pix, full_view]
                        coord = np.round(pixel_coord.ravel()).astype(int)

                        # Now update cache - basically check which dimensions in
                        # the output of the transformation rely on broadcasting.
                        # The 'reset_slices' item is a list that indicates
                        # whether the cache should be reset when the index along
                        # a given dimension changes.
                        sub_data = self.viewer_state.reference_data[pix, sub_data_view]
                        sub_data = unbroadcast(sub_data)
                        self._pixel_cache['reset_slices'][ipix] = [x > 1 for x in sub_data.shape]
                        self._pixel_cache['coord'][ipix] = coord
                        self._pixel_cache['shape'][ipix] = pixel_coord.shape
                        original_shape = pixel_coord.shape

                    else:

                        coord = self._pixel_cache['coord'][ipix]
                        original_shape = self._pixel_cache['shape'][ipix]

                    coords.append(coord)

                self._pixel_cache['view'] = view

                # TODO: add test when image is smaller than cube

                # We now do a nearest-neighbor interpolation. We don't use
                # map_coordinates because it is picky about array endian-ness
                # and if we just use normal Numpy slicing we can preserve the
                # data type (and avoid memory copies)
                keep = np.ones(len(coords[0]), dtype=bool)
                image = np.zeros(len(coords[0])) * np.nan
                for icoord, coord in enumerate(coords):
                    keep[(coord < 0) | (coord >= self.layer.shape[icoord])] = False
                coords = [coord[keep] for coord in coords]
                image[keep] = self._get_image(view=coords)

                # Finally convert array back to a 2D array
                image = image.reshape(original_shape)

                # Determine which slice indices should cause the cache to get
                # reset and the image to be re-projected.

                reset_slices = []
                single_pixel = (0,) * self.layer.ndim
                for pix in self.viewer_state.reference_data.pixel_component_ids:
                    try:
                        self.layer[pix, single_pixel]
                        reset_slices.append(True)
                    except IncompatibleAttribute:
                        reset_slices.append(False)

        # Apply aggregation functions if needed

        if agg_func is None:

            if image.ndim != 2:
                raise IncompatibleDataException()

        else:

            if image.ndim != len(agg_func):
                raise ValueError("Sliced image dimensions ({0}) does not match "
                                 "aggregation function list ({1})"
                                 .format(image.ndim, len(agg_func)))

            for axis in range(image.ndim - 1, -1, -1):
                func = agg_func[axis]
                if func is not None:
                    image = func(image, axis=axis)

            if image.ndim != 2:
                raise ValueError("Image after aggregation should have two dimensions")

        if transpose:
            image = image.transpose()

        self._image_cache = {'view': view, 'image': image, 'reset_slices': reset_slices}

        return image

    def reset_cache_from_slices(self, slice_before, slice_after):

        # When the slice changes, we don't necessarily need to reset the cache
        # as the slice being changed may not have a counterpart in the image
        # shown. For instance, when showing a spectral cube and a 2D image, the
        # 2D image doesn't need to be reprojected every time the spectral slice
        # changes. The reset_slices key in the cache dictionary is either `True`
        # if any change in slice should cause the cache to get reset, or it is
        # a list of boolean values for each slice dimension.

        # We do this first for the image cache, which is the cache of the
        # reprojected slice.

        if self._image_cache is not None:
            if self._image_cache['reset_slices'] is True:
                self._image_cache = None
            else:
                reset_slices = self._image_cache['reset_slices']
                for islice in range(len(slice_before)):
                    if slice_before[islice] != slice_after[islice] and reset_slices[islice]:
                        self._image_cache = None
                        break

        # And we then deal with the pixel transformation cache.

        if self._pixel_cache is not None:
            for ipix in range(self.layer.ndim):
                reset_slices = self._pixel_cache['reset_slices'][ipix]
                if reset_slices is not None:
                    for islice in range(len(slice_before)):
                        if slice_before[islice] != slice_after[islice] and reset_slices[islice]:
                            self._pixel_cache['coord'][ipix] = None
                            self._pixel_cache['reset_slices'][ipix] = None
                            break

    def reset_cache(self, *event):
        self._image_cache = None
        self._pixel_cache = None

    def _get_image(self, view=None):
        raise NotImplementedError()


class ImageLayerState(BaseImageLayerState):
    """
    A state class that includes all the attributes for data layers in an image plot.
    """

    attribute = DDSCProperty(docstring='The attribute shown in the layer')
    v_min = DDCProperty(docstring='The lower level shown')
    v_max = DDCProperty(docstring='The upper leven shown')
    percentile = DDSCProperty(docstring='The percentile value used to '
                                        'automatically calculate levels')
    contrast = DDCProperty(1, docstring='The contrast of the layer')
    bias = DDCProperty(0.5, docstring='A constant value that is added to the '
                                      'layer before rendering')
    cmap = DDCProperty(docstring='The colormap used to render the layer')
    stretch = DDSCProperty(docstring='The stretch used to render the layer, '
                                     'which should be one of ``linear``, '
                                     '``sqrt``, ``log``, or ``arcsinh``')
    global_sync = DDCProperty(True, docstring='Whether the color and transparency '
                                              'should be synced with the global '
                                              'color and transparency for the data')

    def __init__(self, layer=None, viewer_state=None, **kwargs):

        super(ImageLayerState, self).__init__(layer=layer, viewer_state=viewer_state)

        self.attribute_lim_helper = StateAttributeLimitsHelper(self, attribute='attribute',
                                                               percentile='percentile',
                                                               lower='v_min', upper='v_max')

        self.attribute_att_helper = ComponentIDComboHelper(self, 'attribute',
                                                           numeric=True, categorical=False)

        percentile_display = {100: 'Min/Max',
                              99.5: '99.5%',
                              99: '99%',
                              95: '95%',
                              90: '90%',
                              'Custom': 'Custom'}

        ImageLayerState.percentile.set_choices(self, [100, 99.5, 99, 95, 90, 'Custom'])
        ImageLayerState.percentile.set_display_func(self, percentile_display.get)

        stretch_display = {'linear': 'Linear',
                           'sqrt': 'Square Root',
                           'arcsinh': 'Arcsinh',
                           'log': 'Logarithmic'}

        ImageLayerState.stretch.set_choices(self, ['linear', 'sqrt', 'arcsinh', 'log'])
        ImageLayerState.stretch.set_display_func(self, stretch_display.get)

        self.add_callback('global_sync', self._update_syncing)
        self.add_callback('layer', self._update_attribute)

        self._update_syncing()

        if layer is not None:
            self._update_attribute()

        self.update_from_dict(kwargs)

        if self.cmap is None:
            self.cmap = colormaps.members[0][1]

    def _update_attribute(self, *args):
        if self.layer is not None:
            self.attribute_att_helper.set_multiple_data([self.layer])
            self.attribute = self.layer.visible_components[0]

    def _update_priority(self, name):
        if name == 'layer':
            return 3
        elif name == 'attribute':
            return 2
        elif name == 'global_sync':
            return 1.5
        elif name.endswith(('_min', '_max')):
            return 0
        else:
            return 1

    def _update_syncing(self, *args):
        if self.global_sync:
            self._sync_color.enable_syncing()
            self._sync_alpha.enable_syncing()
        else:
            self._sync_color.disable_syncing()
            self._sync_alpha.disable_syncing()

    def _get_image(self, view=None):
        return self.layer[self.attribute, view]

    def flip_limits(self):
        """
        Flip the image levels.
        """
        self.attribute_lim_helper.flip_limits()

    def reset_contrast_bias(self):
        with delay_callback(self, 'contrast', 'bias'):
            self.contrast = 1
            self.bias = 0.5


class ImageSubsetLayerState(BaseImageLayerState):
    """
    A state class that includes all the attributes for subset layers in an image plot.
    """

    # TODO: we can save memory by not showing subset multiple times for
    # different image datasets since the footprint should be the same.

    def _get_image(self, view=None):
        return self.layer.to_mask(view=view)
