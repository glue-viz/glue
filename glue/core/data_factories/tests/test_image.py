import os

from numpy.testing import assert_array_equal

from glue.core.coordinates import WCSCoordinates
from glue.core import data_factories as df
from glue.tests.helpers import requires_pyavm, requires_pil_or_skimage, make_file

DATA = os.path.join(os.path.dirname(__file__), 'data')


@requires_pil_or_skimage
def test_grey_png_loader():
    # Greyscale PNG
    data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x02\x00\x00\x00\x02\x08\x00\x00\x00\x00W\xddR\xf8\x00\x00\x00\x0eIDATx\x9ccdddab\x04\x00\x00&\x00\x0b\x8e`\xe7A\x00\x00\x00\x00IEND\xaeB`\x82'
    with make_file(data, '.png') as fname:
        d = df.load_data(fname)
        assert df.find_factory(fname) is df.img_data
    assert_array_equal(d['PRIMARY'], [[3, 4], [1, 2]])


@requires_pil_or_skimage
def test_color_png_loader():
    # Colorscale PNG
    data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x02\x00\x00\x00\x02\x08\x02\x00\x00\x00\xfd\xd4\x9as\x00\x00\x00\x15IDAT\x08\xd7\x05\xc1\x01\x01\x00\x00\x00\x80\x10\xffO\x17B\x14\x1a!\xec\x04\xfc\xf2!Q\\\x00\x00\x00\x00IEND\xaeB`\x82'
    with make_file(data, '.png') as fname:
        d = df.load_data(fname)
        assert df.find_factory(fname) is df.img_data
    assert_array_equal(d['red'], [[255, 0], [255, 0]])
    assert_array_equal(d['green'], [[255, 0], [0, 255]])
    assert_array_equal(d['blue'], [[0, 255], [0, 0]])


@requires_pyavm
@requires_pil_or_skimage
def test_avm():
    data = df.load_data(os.path.join(DATA, 'ssc2006-16a1_Ti.jpg'))
    assert isinstance(data.coords, WCSCoordinates)
