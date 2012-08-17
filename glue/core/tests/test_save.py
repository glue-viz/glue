from ..glue_pickle import dumps, loads
from ... import core


def test_subset_pickleable():
    s = core.Subset(None)
    dumps(s)


def test_visual_attriubute_pickleable():
    v = core.visual.VisualAttributes()
    dumps(v)


def test_numpy_pickleable():
    import numpy as np
    a = np.array([1, 2, 3])
    dumps(a)


def test_coordinates_pickleable():
    from test_coordinates import header_from_string, HDR_2D_VALID
    from ..coordinates import WCSCoordinates

    hdr = header_from_string(HDR_2D_VALID)
    coords = WCSCoordinates(hdr)
    assert coords._wcs.wcs is not None
    out = dumps(coords)
    new_coords = loads(out)
    assert new_coords._wcs.wcs is not None


def test_componentid_pickleable():
    cid = core.ComponentID('test label')
    dumps(cid)


def test_component_pickleable():
    import numpy as np
    c = core.Component(np.array([1, 2, 3]))
    dumps(c)


def test_component_link_pickleable():
    from_ = core.ComponentID('test')
    to = core.ComponentID('test2')
    using = lambda x: x + 3
    cl = core.ComponentLink([from_], to, using)

    dumps(cl)


def test_data_pickleable():
    d = core.Data()
    dumps(d)


def test_hub_pickleable():
    h = core.Hub()
    dumps(h)


def test_data_collection():
    d = core.Data()
    d2 = core.Data()
    dc = core.DataCollection()
    dc.append(d)
    dc.append(d2)
    dumps(dc)


def test_client_pickleable():
    c = core.Client(core.DataCollection())
    dumps(c)


def test_link_manager_pickleable():
    l = core.LinkManager()
    dumps(l)


def test_hub_pickles_only_core_subscriptions():
    from ...clients import ScatterClient
    from pickle import loads

    h = core.Hub()
    dc = core.DataCollection()
    sc = ScatterClient(dc)

    dc.register_to_hub(h)
    ct0 = len(h._subscriptions)

    sc.register_to_hub(h)
    assert len(h._subscriptions) > ct0

    result = dumps(h)

    new_hub = loads(result)

    assert len(new_hub._subscriptions) == ct0
