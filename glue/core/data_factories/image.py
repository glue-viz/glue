import numpy as np

from glue.core.coordinates import coordinates_from_wcs
from glue.core.data_factories.helpers import has_extension
from glue.core.data import Data
from glue.config import data_factory


IMG_FMT = ['jpg', 'jpeg', 'bmp', 'png', 'tiff', 'tif']

__all__ = ['img_data']


def img_loader(file_name):
    """Load an image to a numpy array, using either PIL or skimage

    Parameters
    ----------
    file_name : str
        Path of the file to load.

    Returns
    -------
    :class:`~numpy.ndarray`
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


@data_factory(label='Image', identifier=has_extension(' '.join(IMG_FMT)))
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
        avm = AVM.from_image(str(file_name))  # avoid unicode
        wcs = avm.to_wcs()
    except Exception:
        pass
    else:
        result.coords = coordinates_from_wcs(wcs)

    for c, l in zip(comps, labels):
        result.add_component(c, l)

    return result
