"""
Module to convert Glue objects to and from JSON descriptions

Example Usage:

s = GlueSerializer(object)
s.dumpo() -> a JSON-serializeable dict
s.dumps() -> a JSON string
s.dump(file) -> dump to a file object

varname = s.id(x) -> string identifier that uniquely labels an object in
                     the Serialized state

u = GlueUnSerializer.load(file)
u = GlueUnSerializer.loads(str)
u.object(varname) -> A reconstituted version of `x`
u.object('__main__') -> The object passed to the GlueSerializer constructor

Developer Notes:

Custom methods to serialize a class of objects can be registered either by:
 - wrapping a serialization function in the @saver decorator::

    @saver(TypeToSave)
    def save(object, context):
        ...

 - Defining a __gluestate__(self, context) method

These methods should return a JSON-serializable dict representing the
object.  context is a GlueSerializer instance. The `context.id` and
`context.do` methods are helpful for referencing or serializing or
dependencies

Unserializer methods can be registered either via:
 - wrapping the method in the @loader decorator::

     @loader(TypeToLoad)
     def load(rec, context)

`rec` is the JSON dict created from the saver, and `context` is a
GlueUnserializer object. context.object() is useful for unserializing
dependencies.

Versions:

Both the @saver and @loader take an optional version keyword. Whenever
you modify the serialization format for an object, you should register a
new saver and loader version. This ensures Glue can still load old
serialization protocols. Versions must be sequential integers,
starting from 1.
"""

import os
import json
import uuid
import types
import logging
from io import BytesIO
from itertools import count
from collections import defaultdict, OrderedDict
from base64 import b64encode, b64decode
from inspect import isgeneratorfunction

import numpy as np
from matplotlib.colors import Colormap
from matplotlib import cm
from astropy.wcs import WCS

from glue import core
from glue.core.data import Data
from glue.core.component_id import ComponentID, PixelComponentID
from glue.core.component import (Component, CategoricalComponent,
                                 DerivedComponent, CoordinateComponent)
from glue.core.subset import (OPSYM, SYMOP, CompositeSubsetState,
                              SubsetState, Subset, RoiSubsetState,
                              InequalitySubsetState, RangeSubsetState)
from glue.core import (VisualAttributes, ComponentLink, DataCollection)
from glue.core.component_link import CoordinateComponentLink
from glue.core.roi import Roi
from glue.core import glue_pickle as gp
from glue.core.subset_group import coerce_subset_groups
from glue.config import session_patch
from glue.utils import lookup_class
from glue.utils.matplotlib import MATPLOTLIB_GE_36

if MATPLOTLIB_GE_36:
    from matplotlib import colormaps

literals = tuple([type(None), float, int, bytes, bool])
literals += tuple(s for s in np.ScalarType if s not in (np.datetime64, np.timedelta64))

builtin_iterables = (tuple, list, set)

JSON_ENCODER = json.JSONEncoder()

# We need to make sure that we don't break backward-compatibility when we move
# classes/functions around in Glue, so we have a file that maps the old paths to
# the new location, and we read this in to PATH_PATCHES.
PATCH_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          'state_path_patches.txt'))

# For Mac app, need to get file from source directory
if not os.path.exists(PATCH_FILE) and 'site-packages.zip' in PATCH_FILE:
    PATCH_FILE = PATCH_FILE.replace('site-packages.zip', 'glue')

PATH_PATCHES = {}
with open(PATCH_FILE) as fp:
    for line in fp:
        before, after = line.strip().split(' -> ')
        PATH_PATCHES[before.strip()] = after.strip()


def save(filename, obj):
    s = GlueSerializer(obj)
    with open(filename, 'w') as f:
        s.dump(f)


def load(filename):
    with open(filename, 'r') as f:
        s = GlueUnSerializer.load(f)
    return s.object('__main__')


def lookup_class_with_patches(name):
    """
    A wrapper to lookup_class that also patches paths to ensure
    backward-compatibility when functions/classes are moved around.
    """
    while name in PATH_PATCHES:
        name = PATH_PATCHES[name]
    return lookup_class(name)


class GlueSerializeError(RuntimeError):
    pass


class VersionedDict(object):

    """
    A dict-like object which associates (key, version_int) pairs
    with an object. Bracket syntax (d[key]) returns the highest-version
    value stored with a key.

    Versions must be sequential integers starting with 1, and must be
    added in order

    Examples
    --------
    v = VersionedDict()
    v['key', 1] = 'v1'
    v['key', 2] = 'v2'

    v['key'] -> 'v2', 2
    v.get_version('key', 2) -> 'v2'
    v.get_version('key', 1) -> 'v1'
    'key' in v -> True

    Not allowed:
    v['key', 4] = 'cannot skip versions'
    v['key', 2] = 'cannot overwrite versions'
    v['key', 'bad'] = 'versions must be integers'
    """

    def __init__(self):
        self._data = defaultdict(dict)

    def __contains__(self, key):
        return key in self._data

    def get_version(self, key, version=None):
        """
        Get a specific version of a value stored with a key

        :param key: The key to fetch
        :param value: the version of the value to fetch. Defaults to latest
        """
        if version is None:
            if key not in self._data:
                raise KeyError("No value associated with any version of %s"
                               % key)
            vs = self._data[key]
            return vs[max(vs)]

        try:
            return self._data[key][version]
        except KeyError:
            raise KeyError("No value associated with version %s of %s" %
                           (version, key))

    def __getitem__(self, key):
        """Retrieve the highest-version value stored with a key

        Returns a tuple of the value, and the version it is associated with
        """
        if key not in self._data:
            raise KeyError(key)
        versions = self._data[key]
        return versions[max(versions)], max(versions)

    def __delitem__(self, key):
        raise ValueError("Cannot remove items from VersionedDict")

    def __len__(self):
        return len(self._data)

    def __setitem__(self, key, value):
        """ Assign a new value with a particular key and version

        :param key: a tuple of (key, version)
        version must be an integer, equal to the previous version + 1 (or 1)
        Overwriting versions is not permitted, and will raise a KeyError

        :param value: The value to associate with the (key, version) pair
        """
        if len(key) != 2:
            raise ValueError("Key must be a (item, version) pair")
        item, version = key
        try:
            version = int(version)
        except ValueError:
            raise ValueError("Version must be an integer: %s" % version)
        if version > 1 and (version - 1) not in self._data[item]:
            raise KeyError("Cannot assign version %i of item before adding "
                           "version %i" % (version, version - 1))
        if version in self._data[item]:
            raise KeyError("Cannot overwrite version %i of %s" %
                           (version, item))

        self._data[item][version] = value


def as_nested_lists(obj):
    items = []
    for item in obj:
        if type(item) in builtin_iterables:
            item = as_nested_lists(item)
        items.append(item)
    return items


def flattened(obj):
    items = []
    for item in obj:
        if type(item) in builtin_iterables:
            items += as_nested_lists(item)
        else:
            items.append(item)
    return items


class GlueSerializer(object):

    """
    Serialize an object graph
    """
    dispatch = VersionedDict()

    def __init__(self, obj, include_data=False, absolute_paths=True):
        self._names = {}  # map id(object) -> name
        self._objs = {}   # map name -> object
        self._working = set()
        self._main = obj
        self.id(obj)
        self.include_data = include_data
        self.absolute_paths = absolute_paths

    @classmethod
    def serializes(cls, obj, version=1):
        def decorator(func):
            cls.dispatch[(obj, version)] = func
            return func
        return decorator

    def _label(self, obj):
        if obj is self._main:
            return '__main__'
        elif hasattr(obj, 'label'):
            return self._disambiguate(obj.label)
        else:
            return self._disambiguate(type(obj).__name__)

    def id(self, obj):
        """
        Return a unique name for an object, and add it to the ID registry
        if necessary.
        """
        if isinstance(obj, str):
            return 'st__%s' % obj

        if type(obj) in literals:
            return obj

        # Now check for list, set, and tuple, and skip if they don't contain
        # any non-literals.
        if type(obj) in builtin_iterables:
            if all(isinstance(x, literals) for x in flattened(obj)):
                return as_nested_lists(obj)

        oid = id(obj)

        if oid in self._names:
            return self._names[oid]

        name = self._label(obj)
        assert name not in self._objs

        logging.debug("Registering %r as %s", obj, name)
        self._objs[name] = obj
        self._names[oid] = name

        return name

    def object(self, name):
        return self._objs[name]

    def do_all(self):
        sz = -1
        while sz != len(self._objs):
            sz = len(self._objs)
            # we need to construct this in two steps otherwise we get a
            # 'dictionary changed size during iteration' error.
            result = [(oid, self.do(obj)) for oid, obj in list(self._objs.items())]
            result = dict(result)
        return result

    def do(self, obj):
        """
        Serialize an object, but do not add it to
        the ID registry
        """
        if isinstance(obj, str):
            return 'st__' + obj

        if type(obj) in literals:
            return obj

        # Now check for list, set, and tuple, and skip if they don't contain
        # any non-literals
        if type(obj) in builtin_iterables:
            if all(isinstance(x, literals) for x in flattened(obj)):
                return as_nested_lists(obj)

        oid = id(obj)
        if oid in self._working:
            raise GlueSerializeError("Circular reference detected")
        self._working.add(oid)

        fun, version = self._dispatch(obj)
        logging.debug("Serializing %s with %s", obj, fun)
        result = fun(obj, self)

        if isinstance(obj, types.FunctionType):
            result['_type'] = 'types.FunctionType'
        elif isinstance(obj, types.MethodType):
            result['_type'] = 'types.MethodType'
        else:
            result['_type'] = "%s.%s" % (type(obj).__module__,
                                         type(obj).__name__)
        if version > 1:
            result['_protocol'] = version

        self._working.remove(oid)
        return result

    def _dispatch(self, obj):

        if hasattr(obj, '__gluestate__'):
            return type(obj).__gluestate__, 1

        try:
            for typ in type(obj).mro():
                if typ in self.dispatch:
                    return self.dispatch[typ]
        except TypeError:  # no mro
            pass

        raise GlueSerializeError("Don't know how to serialize"
                                 " %r of type %s" % (obj, type(obj)))

    def _disambiguate(self, name):
        if name not in self._objs:
            return name

        for i in count(0):
            newname = "%s_%i" % (name, i)
            if newname not in self._objs:
                return newname

    def dumpo(self):
        """
        Dump an object (with needed dependencies) into a
        JSON Serializable data structure.

        Note: If eventually dumping to a string or file, dumps or dump
              are more robust
        """
        return self.do_all()

    @staticmethod
    def json_default(o):
        """Default JSON enconding, to handle some special cases

        In particular, coerces numpy scalars to the equivalent
        python types

        Can be used as default kwarg in json.dumps/json.dump
        """
        if np.isscalar(o) and isinstance(o, np.generic):
            return o.item()  # coerce numpy number to pure-python type
        elif isinstance(o, (tuple, set)):
            return list(o)
        else:
            # Dispatch to JSONEncoder so that we see errors straight away
            # if an object can't be serialized.
            return JSON_ENCODER.default(o)

    def dumps(self, indent=None):
        result = self.dumpo()
        return json.dumps(result, default=self.json_default,
                          indent=indent, sort_keys=True)

    def dump(self, outfile, indent=None):
        result = self.dumpo()
        return json.dump(result, outfile, default=self.json_default,
                         indent=indent, sort_keys=True)


class GlueUnSerializer(object):
    dispatch = VersionedDict()

    def __init__(self, string=None, fobj=None):
        if string is None and fobj is None:
            raise ValueError("Most provide either a string or a file")
        self._names = {}  # map id(object) -> name
        self._objs = {}   # map name -> object
        self._working = set()
        self._rec = json.loads(string) if string else json.load(fobj)
        self._callbacks = []

        # apply Glue defined patches
        apply_inplace_patches(self._rec)
        # apply user-defined patches
        for patcher in session_patch:
            patcher.function(self._rec)

    @classmethod
    def loads(cls, string):
        return cls(string=string)

    @classmethod
    def load(cls, fobj):
        return cls(fobj=fobj)

    @classmethod
    def unserializes(cls, obj, version=1):
        def decorator(func):
            cls.dispatch[(obj, version)] = func
            return func
        return decorator

    def _dispatch(self, rec):

        typ = lookup_class_with_patches(rec['_type'])

        if typ is None:
            raise GlueSerializeError("Unkonwn type %s" % rec['_type'])

        version = rec.get('_protocol', 1)

        if hasattr(typ, '__setgluestate__'):
            return typ.__setgluestate__

        for t in typ.mro():
            try:
                return self.dispatch.get_version(t, version)
            except KeyError:
                continue

        raise GlueSerializeError("Don't know how to load"
                                 " objects of type %s" % typ)

    def register_object(self, obj_id, obj):
        self._objs[obj_id] = obj

    @core.registry.disable
    def object(self, obj_id):

        if isinstance(obj_id, str):

            if obj_id.startswith('st__'):  # a string literal
                return obj_id[4:]

            if obj_id in self._objs:
                return self._objs[obj_id]

            if obj_id not in self._rec:
                raise GlueSerializeError("Unrecognized object %s" % obj_id)

            if obj_id in self._working:
                raise GlueSerializeError("Circular Reference detected: %s" % obj_id)

            self._working.add(obj_id)
            rec = self._rec[obj_id]

        elif isinstance(obj_id, literals) or isinstance(obj_id, (tuple, list)):
            return obj_id
        else:
            rec = obj_id

        func = self._dispatch(rec)

        try:

            obj = func(rec, self)

            if hasattr(obj, '__setgluestate_callback__'):
                self._callbacks.append(obj.__setgluestate_callback__)

            # loader functions might yield the constructed value,
            # and then futher populate it. This deals with circular
            # dependencies.
            if isgeneratorfunction(func):
                gen, obj = obj, next(obj)  # get the partially-constructed value...

            if isinstance(obj_id, str):  # ... add it to the registry ...
                self._objs[obj_id] = obj
                self._working.remove(obj_id)

            if isgeneratorfunction(func):
                for _ in gen:  # ... and finish constructing it
                    pass

        finally:

            # If anything in the try: block above fails, we need to remove the
            # obj_id from te list of IDs we are currently working on, as we
            # may want to try again (this happens when using the callbacks below)

            if isinstance(obj_id, str) and obj_id in self._working:
                self._working.remove(obj_id)

        self._try_callbacks()

        return obj

    def _try_callbacks(self):
        for callback in self._callbacks[:]:
            try:
                callback(self)
            except Exception:
                pass
            else:
                # In some cases (unclear how to trigger this) callback is no
                # longer in the list by the time we try and remove it, hence
                # why we need this try...except.
                try:
                    self._callbacks.remove(callback)
                except ValueError:
                    pass


saver = GlueSerializer.serializes
loader = GlueUnSerializer.unserializes


@saver(dict)
def _save_dict(state, context):
    return dict(contents=dict((context.id(key), context.id(value))
                              for key, value in state.items()))


@loader(dict)
def _load_dict(rec, context):
    return dict((context.object(key), context.object(value))
                for key, value in rec['contents'].items())


@saver(tuple)
def _save_tuple(state, context):
    return dict(contents=[context.do(item) for item in state])


@loader(tuple)
def _load_tuple(rec, context):
    return tuple(_load_list(rec, context))


@saver(list)
def _save_list(state, context):
    return dict(contents=[context.id(item) for item in state])


@loader(list)
def _load_list(rec, context):
    return [context.object(item) for item in rec['contents']]


@saver(set)
def _save_set(state, context):
    return dict(contents=[context.do(item) for item in state])


@loader(set)
def _load_set(rec, context):
    return set(_load_list(rec, context))


@saver(slice)
def _save_slice(slc, context):
    return dict(start=slc.start, stop=slc.stop, step=slc.step)


@loader(slice)
def _load_slice(rec, context):
    return slice(rec['start'], rec['stop'], rec['step'])


@saver(WCS)
def _save_wcs(wcs, context):
    return dict(header=wcs.to_header_string())


@loader(WCS)
def _load_wcs(rec, context):
    from astropy.io import fits
    return WCS(fits.Header.fromstring(rec['header']))


@saver(CompositeSubsetState)
def _save_composite_subset_state(state, context):
    return dict(state1=context.id(state.state1),
                state2=context.id(state.state2))


@loader(CompositeSubsetState)
def _load_composite_subset_state(rec, context):
    cls = lookup_class_with_patches(rec['_type'])
    result = cls(context.object(rec['state1']),
                 context.object(rec['state2']))
    return result


@saver(SubsetState)
def _save_subset_state(state, context):
    return {}


@loader(SubsetState)
def _load_subset_state(rec, context):
    return SubsetState()


@saver(RangeSubsetState)
def _save_range_subset_state(state, context):
    return dict(lo=context.id(state.lo),
                hi=context.id(state.hi),
                att=context.id(state.att))


@loader(RangeSubsetState)
def _load_range_subset_state(rec, context):
    return RangeSubsetState(context.object(rec['lo']),
                            context.object(rec['hi']),
                            context.object(rec['att']))


@saver(RoiSubsetState)
def _save_roi_subset_state(state, context):
    return dict(xatt=context.id(state.xatt),
                yatt=context.id(state.yatt),
                roi=context.id(state.roi),
                pretransform=context.id(state.pretransform))


@loader(RoiSubsetState)
def _load_roi_subset_state(rec, context):
    return RoiSubsetState(context.object(rec['xatt']),
                          context.object(rec['yatt']),
                          context.object(rec['roi']),
                          context.object(rec['pretransform'] if 'pretransform' in rec else None))


@saver(InequalitySubsetState)
def _save_inequality_subset_state(state, context):
    return dict(left=context.id(state.left),
                right=context.id(state.right),
                op=OPSYM.get(state.operator))


@loader(InequalitySubsetState)
def _load_inequality_subset_state(rec, context):
    return InequalitySubsetState(context.object(rec['left']),
                                 context.object(rec['right']),
                                 SYMOP[rec['op']])


@saver(Roi)
def _save_roi(roi, context):
    raise NotImplementedError


@loader(Roi)
def _load_roi(roi, context):
    raise NotImplementedError


@saver(VisualAttributes)
def _save_style(style, context):
    return dict((a, getattr(style, a)) for a in style._atts)


@loader(VisualAttributes)
def _load_style(rec, context):
    result = VisualAttributes()
    if 'preferred_cmap' in result._atts:
        result._atts.remove('preferred_cmap')
    for attr in result._atts:
        setattr(result, attr, rec[attr])
    return result


@saver(Subset)
def _save_subset(subset, context):
    return dict(style=context.do(subset.style),
                state=context.id(subset.subset_state),
                label=subset.label)


@loader(Subset)
def _load_subset(rec, context):
    result = Subset(None)
    result.style = context.object(rec['style'])
    result.subset_state = context.object(rec['state'])
    result.label = rec['label']
    return result


@saver(DataCollection)
def _save_data_collection(dc, context):
    cids = [c for data in dc for c in data.component_ids()]
    components = [data.get_component(c)
                  for data in dc for c in data.component_ids()]
    return dict(data=list(map(context.id, dc)),
                links=list(map(context.id, dc.links)),
                cids=list(map(context.id, cids)),
                components=list(map(context.id, components)))


@saver(DataCollection, version=2)
def _save_data_collection_2(dc, context):
    result = _save_data_collection(dc, context)
    result['groups'] = list(map(context.id, dc.subset_groups))
    return result


@saver(DataCollection, version=3)
def _save_data_collection_3(dc, context):
    result = _save_data_collection_2(dc, context)
    result['subset_group_count'] = dc._sg_count
    return result


@saver(DataCollection, version=4)
def _save_data_collection_4(dc, context):
    cids = [c for data in dc for c in data.component_ids()]
    components = [data.get_component(c)
                  for data in dc for c in data.component_ids()]
    return dict(data=list(map(context.id, dc)),
                links=list(map(context.id, dc.external_links)),
                cids=list(map(context.id, cids)),
                components=list(map(context.id, components)),
                groups=list(map(context.id, dc.subset_groups)),
                subset_group_count=dc._sg_count)


@loader(DataCollection)
def _load_data_collection(rec, context):

    datasets = list(map(context.object, rec['data']))

    links = [context.object(link) for link in rec['links']]

    # Filter out CoordinateComponentLinks that may have been saved in the past
    # as these are now re-generated on-the-fly.
    links = [link for link in links if not isinstance(link, CoordinateComponentLink)]

    # Go through and split links into links internal to datasets and ones
    # between datasets as this dictates whether they should be set on the
    # data collection or on the data objects.
    external, internal = [], []
    for link in links:
        parent_to = link.get_to_id().parent
        for cid in link.get_from_ids():
            if cid.parent is not parent_to:
                external.append(link)
                break
        else:
            internal.append(link)

    # Remove components in datasets that have external links
    for data in datasets:
        remove = []
        for cid in data.derived_components:
            comp = data.get_component(cid)

            # Neihter in external nor in links overall
            if rec.get('_protocol', 0) <= 3:
                if comp.link not in internal:
                    remove.append(cid)

            if isinstance(comp.link, CoordinateComponentLink):
                remove.append(cid)

            if len(comp.link.get_from_ids()) == 1 and comp.link.get_from_ids()[0].parent is comp.link.get_to_id().parent and comp.link.get_from_ids()[0].label == comp.link.get_to_id().label:
                remove.append(cid)

        for cid in remove:
            data.remove_component(cid)

    dc = DataCollection(datasets)

    dc.set_links(external)
    coerce_subset_groups(dc)
    return dc


@loader(DataCollection, version=2)
def _load_data_collection_2(rec, context):
    result = _load_data_collection(rec, context)
    result._subset_groups = list(map(context.object, rec['groups']))
    for grp in result.subset_groups:
        grp.register_to_hub(result.hub)
    return result


@loader(DataCollection, version=3)
def _load_data_collection_3(rec, context):
    result = _load_data_collection_2(rec, context)
    result._sg_count = rec['subset_group_count']
    return result


@loader(DataCollection, version=4)
def _load_data_collection_4(rec, context):

    dc = DataCollection(list(map(context.object, rec['data'])))
    links = [context.object(link) for link in rec['links']]
    dc.set_links(links)
    coerce_subset_groups(dc)

    dc._subset_groups = list(map(context.object, rec['groups']))
    for grp in dc.subset_groups:
        grp.register_to_hub(dc.hub)

    dc._sg_count = rec['subset_group_count']

    return dc


@saver(Data)
def _save_data(data, context):

    state = dict(components=[(context.id(c),
                             context.id(data.get_component(c)))
                             for c in data._components],
                 subsets=[context.id(s) for s in data.subsets],
                 label=data.label)

    if data.coords is not None:
        state['coords'] = context.id(data.coords)

    return state


@saver(Data, version=2)
def _save_data_2(data, context):
    result = _save_data(data, context)
    result['style'] = context.do(data.style)
    return result


@loader(Data)
def _load_data(rec, context):

    label = rec['label']
    result = Data(label=label)
    if 'coords' in rec:
        result.coords = context.object(rec['coords'])

    # we manually rebuild pixel/world components, so
    # we override this function. This is pretty ugly
    result._create_pixel_and_world_components = lambda ndim: None

    comps = [list(map(context.object, [cid, comp]))
             for cid, comp in rec['components']]

    for icomp, (cid, comp) in enumerate(comps):
        if isinstance(comp, CoordinateComponent):
            comp._data = result

            # For backward compatibility, we need to check for cases where
            # the component ID for the pixel components was not a PixelComponentID
            # and upgrade it to one. This can be removed once we no longer
            # support pre-v0.8 session files.
            if not comp.world and not isinstance(cid, PixelComponentID):
                cid = PixelComponentID(comp.axis, cid.label, parent=cid.parent)
                comps[icomp] = (cid, comp)

        result.add_component(comp, cid)

    assert result._world_component_ids == []

    coord = [c for c in comps if isinstance(c[1], CoordinateComponent)]
    coord = [x[0] for x in sorted(coord, key=lambda x: x[1])]

    if getattr(result, 'coords') is not None:
        assert len(coord) == result.ndim * 2
        result._world_component_ids = coord[:len(coord) // 2]
        result._pixel_component_ids = coord[len(coord) // 2:]
    else:
        assert len(coord) == result.ndim
        result._pixel_component_ids = coord

    # We can now re-generate the coordinate links
    result._set_up_coordinate_component_links(result.ndim)

    for s in rec['subsets']:
        result.add_subset(context.object(s))

    return result


@loader(Data, version=2)
def _load_data_2(rec, context):
    # adds style saving
    result = _load_data(rec, context)
    result.style = context.object(rec['style'])
    return result


@saver(Data, version=3)
def _save_data_3(data, context):
    result = _save_data_2(data, context)
    result['_key_joins'] = [[context.id(k), context.id(v0), context.id(v1)]
                            for k, (v0, v1) in data._key_joins.items()]
    return result


@loader(Data, version=3)
def _load_data_3(rec, context):
    result = _load_data_2(rec, context)
    yield result
    result._key_joins = dict((context.object(k), (context.object(v0), context.object(v1)))
                             for k, v0, v1 in rec['_key_joins'])


@saver(Data, version=4)
def _save_data_4(data, context):
    result = _save_data_2(data, context)

    def save_cid_tuple(cids):
        return tuple(context.id(cid) for cid in cids)

    result['_key_joins'] = [[context.id(k), save_cid_tuple(v0), save_cid_tuple(v1)]
                            for k, (v0, v1) in data._key_joins.items()]
    result['uuid'] = data.uuid
    return result


@loader(Data, version=4)
def _load_data_4(rec, context):
    result = _load_data_2(rec, context)
    yield result

    def load_cid_tuple(cids):
        return tuple(context.object(cid) for cid in cids)

    result._key_joins = dict((context.object(k), (load_cid_tuple(v0), load_cid_tuple(v1)))
                             for k, v0, v1 in rec['_key_joins'])
    if 'uuid' in rec and rec['uuid'] is not None:
        result.uuid = rec['uuid']
    else:
        result.uuid = str(uuid.uuid4())


@saver(Data, version=5)
def _save_data_5(data, context):
    result = _save_data_4(data, context)
    result['primary_owner'] = [context.id(cid) for cid in data.components if cid.parent is data]
    # Filter out keys/values that can't be serialized
    meta_filtered = OrderedDict()
    for key, value in data.meta.items():
        try:
            context.do(key)
            context.do(value)
        except GlueSerializeError:
            continue
        else:
            meta_filtered[key] = value
    result['meta'] = context.do(meta_filtered)
    return result


@loader(Data, version=5)
def _load_data_5(rec, context):
    result = _load_data_2(rec, context)
    if 'primary_owner' in rec:
        for cid in rec['primary_owner']:
            cid = context.object(cid)
            cid.parent = result
    yield result

    def load_cid_tuple(cids):
        return tuple(context.object(cid) for cid in cids)

    result._key_joins = dict((context.object(k), (load_cid_tuple(v0), load_cid_tuple(v1)))
                             for k, v0, v1 in rec['_key_joins'])
    if 'uuid' in rec and rec['uuid'] is not None:
        result.uuid = rec['uuid']
    else:
        result.uuid = str(uuid.uuid4())
    if 'meta' in rec:
        result.meta.update(context.object(rec['meta']))


@saver(ComponentID)
def _save_component_id(cid, context):
    return dict(label=cid.label)


@loader(ComponentID)
def _load_component_id(rec, context):
    return ComponentID(rec['label'])


@saver(PixelComponentID)
def _save_pixel_component_id(cid, context):
    return dict(axis=cid.axis, label=cid.label)


@loader(PixelComponentID)
def _load_pixel_component_id(rec, context):
    if 'axis' in rec:
        axis = rec['axis']
    else:  # backward-compatibility
        axis = int(rec['label'].split()[2])
    return PixelComponentID(axis, rec['label'])


@saver(Component)
def _save_component(component, context):

    if not context.include_data and hasattr(component, '_load_log'):
        log = component._load_log
        return dict(log=context.id(log),
                    log_item=log.id(component))

    return dict(data=context.do(component.data),
                units=component.units)


@loader(Component)
def _load_component(rec, context):

    if 'log' in rec:
        return context.object(rec['log']).component(rec['log_item'])

    cls = lookup_class_with_patches(rec['_type'])

    return cls(data=context.object(rec['data']),
               units=rec['units'])


@saver(CategoricalComponent)
def _save_categorical_component(component, context):

    if not context.include_data and hasattr(component, '_load_log'):
        log = component._load_log
        return dict(log=context.id(log),
                    log_item=log.id(component))

    return dict(categorical_data=context.do(component.labels),
                categories=context.do(component.categories),
                jitter_method=context.do(component.jitter_method),
                units=component.units)


@loader(CategoricalComponent)
def _load_categorical_component(rec, context):
    if 'log' in rec:
        return context.object(rec['log']).component(rec['log_item'])

    return CategoricalComponent(categorical_data=context.object(rec['categorical_data']),
                                categories=context.object(rec['categories']),
                                jitter=context.object(rec['jitter_method']),
                                units=rec['units'])


@saver(DerivedComponent)
def _save_derived_component(component, context):
    return dict(link=context.id(component.link))


@loader(DerivedComponent)
def _load_derived_component(rec, context):
    return DerivedComponent(None, link=context.object(rec['link']))


@saver(ComponentLink)
def _save_component_link(link, context):
    frm = list(map(context.id, link.get_from_ids()))
    to = list(map(context.id, [link.get_to_id()]))
    using = context.do(link.get_using())
    inverse = context.do(link.get_inverse())
    return dict(frm=frm, to=to, using=using, inverse=inverse)


@loader(ComponentLink)
def _load_component_link(rec, context):
    frm = list(map(context.object, rec['frm']))
    to = list(map(context.object, rec['to']))[0]
    using = context.object(rec['using'])
    inverse = context.object(rec['inverse'])
    result = ComponentLink(frm, to, using, inverse)
    return result


@saver(CoordinateComponentLink)
def _save_coordinate_component_link(link, context):
    frm = list(map(context.id, link._from_all))
    to = list(map(context.id, [link.get_to_id()]))
    coords = context.id(link.coords)
    index = link.index
    pix2world = link.pixel2world
    return dict(frm=frm, to=to, coords=coords, index=index,
                pix2world=pix2world)


@loader(CoordinateComponentLink)
def _load_coordinate_component_link(rec, context):
    to = list(map(context.object, rec['to']))[0]  # XXX why is this a list?
    coords = context.object(rec['coords'])
    index = rec['index']
    pix2world = rec['pix2world']
    frm = list(map(context.object, rec['frm']))
    return CoordinateComponentLink(frm, to, coords, index, pix2world)


@saver(types.BuiltinFunctionType)
def _save_builtin_function(function, context):
    ref = "%s.%s" % (function.__module__, function.__name__)
    return {'function': ref}


@loader(types.BuiltinFunctionType)
def _load_builtin_function(rec, context):
    return lookup_class_with_patches(rec['function'])


@saver(types.FunctionType)
def _save_function(function, context):
    ref = "%s.%s" % (function.__module__, function.__name__)
    if lookup_class_with_patches(ref) is function:
        l = lookup_class_with_patches(ref)
        return {'function': ref}
    return {'pickle': gp.dumps(function).encode('base64')}


@loader(types.FunctionType)
def _load_function(rec, context):
    if 'pickle' in rec:
        return gp.loads(rec['pickle'].decode('base64'))
    return lookup_class_with_patches(rec['function'])


@saver(types.MethodType)
def _save_method(method, context):
    # Note: this only works for methods for which the class can be serialized
    return {'instance': context.id(method.__self__), 'method': method.__name__}


@loader(types.MethodType)
def _load_method(rec, context):
    instance = context.object(rec['instance'])
    return getattr(instance, rec['method'])


@saver(core.Session)
def _save_session(session, context):
    # we will rely on GlueApplication to re-populate
    return {}


@loader(np.ndarray)
def _load_numpy(rec, context):
    s = BytesIO(b64decode(rec['data']))
    return np.load(s)


@saver(np.ndarray)
def _save_numpy(obj, context):
    f = BytesIO()
    np.save(f, obj)
    data = b64encode(f.getvalue()).decode('ascii')
    return dict(data=data)


@saver(Colormap)
def _save_cmap(cmap, context):
    return {'cmap': cmap.name}


@loader(Colormap)
def _load_cmap(rec, context):
    return colormaps[rec['cmap']] if MATPLOTLIB_GE_36 else cm.get_cmap(rec['cmap'])


@saver(np.datetime64)
def _save_datetime64(dt, context):
    return {'datetime64': str(dt)}


@loader(np.datetime64)
def _load_datetime64(rec, context):
    return np.datetime64(rec['datetime64'])


def apply_inplace_patches(rec):
    """
    Apply in-place patches to a loaded session file. Ideally this should be
    empty, except for user patches, but we use this to fix session files that
    need fixing to be interpretable by the current version of glue.
    """

    # The following is a patch for session files made with glue 0.15.* or
    # earlier that were read in with a developer version of glue for part of
    # the 0.16 development cycle, and re-saved. Essentially, if coords is set
    # to the default identity Coordinates class, we need to make sure we
    # always preserve the world coordinate components, and we do that by
    # setting force_coords to True.
    for key, value in rec.items():
        if value['_type'] == 'glue.core.data.Data':
            if 'coords' in value and value['coords'] is not None:
                coords = rec[value['coords']]
                if coords['_type'] == 'glue.core.coordinates.Coordinates':
                    for cid, comp in value['components']:
                        if 'log' in rec[comp]:
                            load_log = rec[rec[comp]['log']]
                            if 'force_coords' not in load_log:
                                load_log['force_coords'] = True

        # The following accounts for the addition of the degree mode to the
        # full-sphere projection. Originally, this was only used for polar mode
        # and so the `coords` parameter was not needed. If this is not present,
        # the plot is polar and we can set coords to be ['x']
        if value['_type'] == 'glue.core.roi_pretransforms.RadianTransform':
            if 'state' in value and value['state'] is not None:
                state = value['state']
                if 'contents' in state and state['contents'] is not None:
                    contents = state['contents']
                    if 'st__coords' not in contents:
                        contents['st__coords'] = ['x']
