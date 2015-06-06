from __future__ import absolute_import, division, print_function

import numpy as np

from ..data import Data

from .helpers import has_extension, set_default_factory, __factories__
from ..coordinates import coordinates_from_wcs

img_fmt = ['jpg', 'jpeg', 'bmp', 'png', 'tiff', 'tif']

__all__ = []


def img_loader(file_name):
    """Load an image to a numpy array, using either PIL or skimage

    :param file_name: Path of file to load
    :rtype: Numpy array
    """
    try:
        from skimage import img_as_ubyte
        from skimage.io import imread
        return np.asarray(img_as_ubyte(imread(file_name)))
    except ImportError:
        pass

    try:
        from PIL import Image
        return np.asarray(Image.open(file_name))
    except ImportError:
        raise ImportError("Reading %s requires PIL or scikit-image" %
                          file_name)


def img_data(file_name):
    """Load common image files into a Glue data object"""
    result = Data()

    data = img_loader(file_name)
    data = np.flipud(data)
    shp = data.shape

    comps = []
    labels = []

    # split 3 color images into each color plane
    if len(shp) == 3 and shp[2] in [3, 4]:
        comps.extend([data[:, :, 0], data[:, :, 1], data[:, :, 2]])
        labels.extend(['red', 'green', 'blue'])
        if shp[2] == 4:
            comps.append(data[:, :, 3])
            labels.append('alpha')
    else:
        comps = [data]
        labels = ['PRIMARY']

    # look for AVM coordinate metadata
    try:
        from pyavm import AVM
        avm = AVM(str(file_name))  # avoid unicode
        wcs = avm.to_wcs()
    except:
        pass
    else:
        result.coords = coordinates_from_wcs(wcs)

    for c, l in zip(comps, labels):
        result.add_component(c, l)

    return result

img_data.label = "Image"
img_data.identifier = has_extension(' '.join(img_fmt))
for i in img_fmt:
    set_default_factory(i, img_data)

__factories__.append(img_data)