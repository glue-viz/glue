# Make sure that session files can be read in a backward-compatible manner

from __future__ import absolute_import, division, print_function

import os
import pytest
import numpy as np

from glue.tests.helpers import requires_astropy, requires_h5py, requires_qt, PYSIDE2_INSTALLED  # noqa
from glue.core.component import CoordinateComponent, Component
from glue.core.state import GlueUnSerializer
from glue.core.component_id import PixelComponentID

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
    ga.close()


@requires_qt
@requires_astropy
@pytest.mark.skipif('PYSIDE2_INSTALLED')
def test_load_viewers_04():

    # FIXME - for some reason this test with PySide2 causes a leftover reference
    # to GlueApplication and appears to be due to x_log being True in the
    # scatter plot. I suspect maybe there is some kind of circular reference

    # This loads a session file made with Glue v0.4. In this session, we have
    # three viewers: one scatter viewer, one image viewer, and one histogram
    # viewer.

    with open(os.path.join(DATA, 'simple_viewers.glu'), 'r') as f:
        content = f.read()

    state = GlueUnSerializer.loads(content)

    ga = state.object('__main__')

    assert len(ga.viewers[0]) == 3
    labels = sorted([x.LABEL for x in ga.viewers[0]])

    assert labels == ['1D Histogram', '2D Image', '2D Scatter']

    viewers = {}
    for x in ga.viewers[0]:
        viewers[x.LABEL] = x

    h = viewers['1D Histogram']
    assert h.viewer_size == (1235, 531)
    assert h.position == (0, 535)
    assert h.state.x_att.label == 'b'

    i = viewers['2D Image']
    assert i.viewer_size == (562, 513)
    assert i.position == (672, 0)
    assert i.state.layers[0].attribute.label == "image"

    s = viewers['2D Scatter']
    assert s.viewer_size == (670, 512)
    assert s.position == (0, 0)
    assert s.state.x_att.label == 'b'
    assert s.state.y_att.label == 'a'
    assert s.state.x_log
    assert not s.state.y_log

    ga.close()


@requires_qt
def test_load_pixel_components_07():

    # This loads a session file made with Glue v0.7. In 0.7 and before,
    # PixelComponentID did not exist, so we need to make sure that when loading
    # in such files, we transform the appropriate ComponentIDs to
    # PixelComponentIDs.

    with open(os.path.join(DATA, 'glue_v0.7_pixel_roi_selection.glu'), 'r') as f:
        content = f.read()

    state = GlueUnSerializer.loads(content)

    ga = state.object('__main__')

    assert isinstance(ga.data_collection[0].pixel_component_ids[0], PixelComponentID)
    assert isinstance(ga.data_collection[0].pixel_component_ids[1], PixelComponentID)

    ga.close()


@requires_qt
def test_table_widget_010():

    from glue.viewers.table.qt.tests.test_data_viewer import check_values_and_color

    # This loads a session file made with Glue v0.10 that includes a table
    # viewer. This is to make sure that loading table viewers from old files
    # will always be backward-compatible.

    with open(os.path.join(DATA, 'glue_v0.10_table.glu'), 'r') as f:
        state = GlueUnSerializer.load(f)

    ga = state.object('__main__')

    viewer = ga.viewers[0][0]

    data = {'x': [1, 2, 3],
            'y': [4, 5, 6]}

    colors = ['#e31a1c', '#6d7326', None]

    check_values_and_color(viewer.model, data, colors)

    ga.close()


@requires_qt
@pytest.mark.parametrize('protocol', (0, 1))
def test_load_log(protocol):

    # Prior to Glue v0.13, components were added to the data as: first
    # non-coordinate component, then coordinate components, then remaining non-
    # coordinate components. In Glue v0.13, this changed to be coordinate
    # components then non-coordinate components. The LoadLog functionality
    # relies on an absolute component index, so we need to be careful - if the
    # session file was created prior to Glue v0.13, we need to load the
    # components in the log using the old order. The load_log_1.glu file was
    # made with Glue v0.12.2, while the load_log_2.glu file was made with
    # Glue v0.13.

    with open(os.path.join(DATA, 'load_log_{0}.glu'.format(protocol)), 'r') as f:
        template = f.read()

    content = template.replace('{DATA_PATH}', (DATA + os.sep).replace('\\', '\\\\'))
    state = GlueUnSerializer.loads(content)

    ga = state.object('__main__')

    dc = ga.session.data_collection

    assert len(dc) == 1

    data = dc[0]

    assert data.label == 'simple'

    np.testing.assert_equal(data['Pixel Axis 0 [x]'], [0, 1, 2])
    np.testing.assert_equal(data['World 0'], [0, 1, 2])
    np.testing.assert_equal(data['a'], [1, 3, 5])
    np.testing.assert_equal(data['b'], [2, 2, 3])

    if protocol == 0:
        assert data.components == [data.id['a'], data.id['Pixel Axis 0 [x]'], data.id['World 0'], data.id['b']]
    else:
        assert data.components == [data.id['Pixel Axis 0 [x]'], data.id['World 0'], data.id['a'], data.id['b']]

    assert type(data.get_component('Pixel Axis 0 [x]')) == CoordinateComponent
    assert type(data.get_component('World 0')) == CoordinateComponent
    assert type(data.get_component('a')) == Component
    assert type(data.get_component('b')) == Component

    ga.close()
