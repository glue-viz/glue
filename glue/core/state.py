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

from __future__ import absolute_import, division, print_function

from itertools import count
from collections import defaultdict
import json
import types
import logging
from inspect import isgeneratorfunction

import numpy as np

from .subset import (OPSYM, SYMOP, CompositeSubsetState,
                     SubsetState, Subset, RoiSubsetState,
                     InequalitySubsetState, RangeSubsetState)
from .data import (Data, Component, ComponentID, DerivedComponent,
                   CoordinateComponent)
from . import (VisualAttributes, ComponentLink, DataCollection)
from .component_link import CoordinateComponentLink
from .util import lookup_class
from .roi import Roi
from . import glue_pickle as gp
from .. import core
from .subset_group import coerce_subset_groups
from ..external import six
from io import BytesIO
from base64 import b64encode, b64decode


literals = tuple([type(None), float, int, bytes, bool, list, tuple])

if six.PY2:
    literals += (long,)
literals += (np.ScalarType,)

_lookup = lookup_class


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


class GlueSerializer(object):

    """
    Serialize an object graph
    """
    dispatch = VersionedDict()

    def __init__(self, obj):
        self._names = {}  # map id(object) -> name
        self._objs = {}   # map name -> object
        self._working = set()
        self._main = obj
        self.id(obj)

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
        if isinstance(obj, six.string_types):
            return 'st__%s' % obj

        if isinstance(obj, literals):
            return obj

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
        if isinstance(obj, six.string_types):
            return 'st__' + obj

        if isinstance(obj, literals):
            return obj

        oid = id(obj)
        if oid in self._working:
            raise GlueSerializeError("Circular reference detected")
        self._working.add(oid)

        fun, version = self._dispatch(obj)
        logging.debug("Serializing %s with %s", obj, fun)
        result = fun(obj, self)

        if isinstance(obj, types.FunctionType):
            result['_type'] = 'types.FunctionType'
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
            return np.asscalar(o)  # coerce numpy number to pure-python type
        if isinstance(o, tuple):
            return list(o)
        return o

    def dumps(self, indent=None):
        result = self.dumpo()
        return json.dumps(result, indent=indent, default=self.json_default)

    def dump(self, outfile, indent=None):
        result = self.dumpo()
        return json.dump(result, outfile, default=self.json_default,
                         indent=indent)


class GlueUnSerializer(object):
    dispatch = VersionedDict()

    def __init__(self, string=None, fobj=None):
        if string is None and fobj is None:
            raise ValueError("Most provide either a string or a file")
        self._names = {}  # map id(object) -> name
        self._objs = {}   # map name -> object
        self._working = set()
        self._rec = json.loads(string) if string else json.load(fobj)

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
        typ = _lookup(rec['_type'])
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
        if isinstance(obj_id, six.string_types):
            if obj_id.startswith('st__'):  # a string literal
                return obj_id[4:]

            if obj_id in self._objs:
                return self._objs[obj_id]

            if obj_id not in self._rec:
                raise GlueSerializeError("Unrecognized object %s" % obj_id)

            if obj_id in self._working:
                raise GlueSerializeError(
                    "Circular Reference detected: %s" % obj_id)

            self._working.add(obj_id)
            rec = self._rec[obj_id]

        elif isinstance(obj_id, literals):
            return obj_id
        else:
            rec = obj_id

        func = self._dispatch(rec)
        obj = func(rec, self)

        # loader functions might yield the constructed value,
        # and then futher populate it. This deals with circular
        # dependencies.
        if isgeneratorfunction(func):
            gen, obj = obj, next(obj)  # get the partially-constructed value...

        if isinstance(obj_id, six.string_types):  # ... add it to the registry ...
            self._objs[obj_id] = obj
            self._working.remove(obj_id)

        if isgeneratorfunction(func):
            for _ in gen:  # ... and finish constructing it
                pass

        return obj


saver = GlueSerializer.serializes
loader = GlueUnSerializer.unserializes


@saver(dict)
def _save_dict(state, context):
    return dict(contents=json.dumps(state))


@loader(dict)
def _load_dict(rec, context):
    return json.loads(rec['contents'])


@saver(CompositeSubsetState)
def _save_composite_subset_state(state, context):
    return dict(state1=context.id(state.state1),
                state2=context.id(state.state2))


@loader(CompositeSubsetState)
def _load_composite_subset_state(rec, context):
    cls = _lookup(rec['_type'])
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
    return dict(lo=state.lo, hi=state.hi, att=context.id(state.att))


@loader(RangeSubsetState)
def _load_range_subset_state(rec, context):
    return RangeSubsetState(rec['lo'], rec['hi'], context.object(rec['att']))


@saver(RoiSubsetState)
def _save_roi_subset_state(state, context):
    return dict(xatt=context.id(state.xatt),
                yatt=context.id(state.yatt),
                roi=context.id(state.roi))


@loader(RoiSubsetState)
def _load_roi_subset_state(rec, context):
    return RoiSubsetState(context.object(rec['xatt']),
                          context.object(rec['yatt']),
                          context.object(rec['roi']))


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
def _laod_roi(roi, context):
    raise NotImplementedError


@saver(VisualAttributes)
def _save_style(style, context):
    return dict((a, getattr(style, a)) for a in style._atts)


@loader(VisualAttributes)
def _load_style(rec, context):
    result = VisualAttributes()
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


@loader(DataCollection)
def _load_data_collection(rec, context):
    dc = DataCollection(list(map(context.object, rec['data'])))
    for link in rec['links']:
        dc.add_link(context.object(link))
    coerce_subset_groups(dc)
    return dc


@loader(DataCollection, version=2)
def _load_data_collection_2(rec, context):
    result = _load_data_collection(rec, context)
    result._subset_groups = list(map(context.object, rec['groups']))
    for grp in result.subset_groups:
        grp.register_to_hub(result.hub)
    return result


@saver(Data)
def _save_data(data, context):

    return dict(components=[(context.id(c),
                            context.id(data.get_component(c)))
                            for c in data._components],
                subsets=[context.id(s) for s in data.subsets],
                label=data.label,
                coords=context.id(data.coords))


@saver(Data, version=2)
def _save_data_2(data, context):
    result = _save_data(data, context)
    result['style'] = context.do(data.style)
    return result


@loader(Data)
def _load_data(rec, context):
    label = rec['label']
    result = Data(label=label)
    result.coords = context.object(rec['coords'])

    # we manually rebuild pixel/world components, so
    # we override this function. This is pretty ugly
    result._create_pixel_and_world_components = lambda: None

    comps = [list(map(context.object, [cid, comp]))
             for cid, comp in rec['components']]
    comps = sorted(comps,
                   key=lambda x: isinstance(x[1], (DerivedComponent,
                                                   CoordinateComponent)))
    for cid, comp in comps:
        if isinstance(comp, CoordinateComponent):
            comp._data = result
        result.add_component(comp, cid)

    assert result._world_component_ids == []

    coord = [c for c in comps if isinstance(c[1], CoordinateComponent)]
    coord = [x[0] for x in sorted(coord, key=lambda x: x[1])]

    assert len(coord) == result.ndim * 2

    result._world_component_ids = coord[:len(coord) // 2]
    result._pixel_component_ids = coord[len(coord) // 2:]

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


@saver(ComponentID)
def _save_component_id(cid, context):
    return dict(label=cid.label, hidden=cid.hidden)


@loader(ComponentID)
def _load_component_id(rec, context):
    return ComponentID(rec['label'], rec['hidden'])


@saver(Component)
def _save_component(component, context):
    if hasattr(component, '_load_log'):
        log = component._load_log
        return dict(log=context.id(log),
                    log_item=log.id(component))

    return dict(data=context.do(component.data),
                units=component.units)


@loader(Component)
def _load_component(rec, context):
    if 'log' in rec:
        return context.object(rec['log']).component(rec['log_item'])

    return Component(data=context.object(rec['data']),
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
    hidden = link.hidden
    return dict(frm=frm, to=to, using=using, inverse=inverse, hidden=hidden)


@loader(ComponentLink)
def _load_component_link(rec, context):
    frm = list(map(context.object, rec['frm']))
    to = list(map(context.object, rec['to']))[0]
    using = context.object(rec['using'])
    inverse = context.object(rec['inverse'])
    result = ComponentLink(frm, to, using, inverse)
    result.hidden = rec['hidden']
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


@saver(types.FunctionType)
def _save_function(function, context):
    ref = "%s.%s" % (function.__module__, function.__name__)
    if _lookup(ref) is function:
        l = _lookup(ref)
        return {'function': ref}
    return {'pickle': gp.dumps(function).encode('base64')}


@loader(types.FunctionType)
def _load_function(rec, context):
    if 'pickle' in rec:
        return gp.loads(rec['pickle'].decode('base64'))
    return _lookup(rec['function'])


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
