# Make sure that session files can be read in a backward-compatible manner

from __future__ import absolute_import, division, print_function

import os

import numpy as np

from glue.tests.helpers import requires_astropy, requires_h5py, requires_qt
from glue.core.state import GlueUnSerializer

DATA = os.path.join(os.path.dirname(__file__), 'data')


@requires_qt
@requires_astropy
def test_load_simple_tables_04():

    # This loads a session file made with Glue v0.4. In this session, we have
    # loaded four tables. The first two are from the same file, but one loaded
    # via the auto loader and the other via the Astropy FITS table loader. The
    # second two were loaded similarly to the first two, but the file contains
    # two HDUs this time. However, in Glue v0.4, only the first HDU was read so
    # we shouldn't have access to columns c and d in ``double_tables.fits``.

    with open(os.path.join(DATA, 'simple_tables.glu'), 'r') as f:
        template = f.read()

    content = template.replace('{DATA_PATH}', (DATA + os.sep).replace('\\', '\\\\'))
    state = GlueUnSerializer.loads(content)

    ga = state.object('__main__')

    dc = ga.session.data_collection

    # All tables should actually be the same because the FITS reader back at
    # 0.4 only read in the first HDU so the new reader is back-compatible
    # since it preserves HDU order.

    assert len(dc) == 4

    assert dc[0].label == 'single_table_auto'
    assert dc[1].label == 'single_table'
    assert dc[2].label == 'double_tables_auto'
    assert dc[3].label == 'double_tables'

    np.testing.assert_equal(dc[0]['a'], [1, 2, 3])
    np.testing.assert_equal(dc[0]['b'], [4, 5, 6])
    np.testing.assert_equal(dc[0]['a'], dc[1]['a'])
    np.testing.assert_equal(dc[0]['b'], dc[1]['b'])
    np.testing.assert_equal(dc[0]['a'], dc[2]['a'])
    np.testing.assert_equal(dc[0]['b'], dc[2]['b'])
    np.testing.assert_equal(dc[0]['a'], dc[3]['a'])
    np.testing.assert_equal(dc[0]['b'], dc[3]['b'])

    ga.close()


@requires_qt
@requires_h5py
def test_load_hdf5_grids_04():

    # This loads a session file made with Glue v0.4. In this session, we have
    # loaded two gridded datasets from an HDF5 datafile: the first one loaded
    # via the auto loader and the other via the FITS/HDF5 loader.

    with open(os.path.join(DATA, 'simple_hdf5_grid.glu'), 'r') as f:
        template = f.read()

    content = template.replace('{DATA_PATH}', (DATA + os.sep).replace('\\', '\\\\'))
    state = GlueUnSerializer.loads(content)

    ga = state.object('__main__')

    dc = ga.session.data_collection

    assert len(dc) == 2

    assert dc[0].label == 'single_grid_auto'
    assert dc[1].label == 'single_grid'

    np.testing.assert_equal(dc[0]['/array1'], 1)
    np.testing.assert_equal(dc[0]['/array1'].shape, (2, 3, 4))

    ga.close()


@requires_qt
@requires_astropy
def test_load_link_helpers_04():

    # This loads a session file made with Glue v0.4. In this session, we have
    # two tables, and we use all the celestial link functions that were present
    # in Glue v0.4. We now check that the paths are patched when loading the
    # session (since the functions have been moved to a deprecated location)

    with open(os.path.join(DATA, 'session_links.glu'), 'r') as f:
        content = f.read()

    state = GlueUnSerializer.loads(content)

    ga = state.object('__main__')


@requires_qt
@requires_astropy
def test_load_viewers_04():

    # This loads a session file made with Glue v0.4. In this session, we have
    # three viewers: one scatter viewer, one image viewer, and one histogram
    # viewer.

    with open(os.path.join(DATA, 'simple_viewers.glu'), 'r') as f:
        content = f.read()

    state = GlueUnSerializer.loads(content)

    ga = state.object('__main__')

    assert len(ga.viewers[0]) == 3
    labels = sorted([x.LABEL for x in ga.viewers[0]])

    assert labels == ['Histogram', 'Image Viewer', 'Scatter Plot']

    viewers = {}
    for x in ga.viewers[0]:
        viewers[x.LABEL] = x

    h = viewers['Histogram']
    assert h.viewer_size == (1235, 531)
    assert h.position == (0, 535)
    assert h.component.label == 'b'

    i = viewers['Image Viewer']
    assert i.viewer_size == (562, 513)
    assert i.position == (672, 0)
    assert i.attribute.label == "image"

    s = viewers['Scatter Plot']
    assert s.viewer_size == (670, 512)
    assert s.position == (0, 0)
    assert s.xatt.label == 'b'
    assert s.yatt.label == 'a'
    assert s.xlog
    assert not s.ylog
    assert not s.xflip
    assert s.yflip
