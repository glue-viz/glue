"""
Code to convert Glue objects to and from JSON descriptions
"""
from itertools import count
from inspect import isgeneratorfunction
import json
import types
import logging
from traceback import extract_stack

import numpy as np

from .subset import (OPSYM, SYMOP, CompositeSubsetState,
                     SubsetState, Subset, RoiSubsetState,
                     InequalitySubsetState)
from .data import (Data, Component, ComponentID, DerivedComponent,
                   CoordinateComponent)
from . import (VisualAttributes, ComponentLink, DataCollection)
from .component_link import CoordinateComponentLink
from .util import lookup_class
from .roi import Roi
from .glue_pickle import dumps, loads
from .. import core

literals = tuple([types.NoneType, types.FloatType,
                 types.IntType, types.LongType,
                 types.NoneType, types.StringType,
                 types.BooleanType, types.UnicodeType, types.ListType])

_lookup = lookup_class


class GlueSerializeError(RuntimeError):
    pass


class GlueSerializer(object):

    """
    strategy:

    do -> return python JSON
    id -> create an ID
    either of these could rely on a not-yet-serialized object
    external references, only make sense when stitched together
    who is in charge of the stiching? This, or another object?

    At last step, will dump into a big object. At that point,
    complain about undefined references

    Top Level Interface
    ----------------
    dump(application, file)
    dumps(application)
    load(file) -> json
    loads(str) -> json

    Low Level Interface
    -------------------
    do : turn an object into json
    id : return an id
    restore(class, json) : return json into object
    object(id): fetch an object referenced by id
    """
    dispatch = {}

    def __init__(self):
        self._names = {}  # map id(object) -> name
        self._objs = {}   # map name -> object
        self._working = set()

    @classmethod
    def serializes(cls, *objs):
        def decorator(func):
            for obj in objs:
                cls.dispatch[obj] = func
            return func
        return decorator

    def id(self, obj):
        """
        Return a unique name for an object,
        and add it to the ID registry if necessary
        """
        if isinstance(obj, literals):
            return obj

        oid = id(obj)

        if oid in self._names:
            return self._names[oid]

        if hasattr(obj, 'label'):
            name = self._disambiguate(obj.label)
        else:
            name = self._disambiguate(type(obj).__name__)
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
            result = dict((oid, self.do(obj))
                          for oid, obj in self._objs.items())
        return result

    def do(self, obj):
        """
        Serialize an object, but do not add it to
        the ID registry
        """
        if isinstance(obj, literals):
            return obj

        oid = id(obj)
        if oid in self._working:
            raise GlueSerializeError("Circular reference detected")
        self._working.add(oid)

        fun = self._dispatch(obj)
        logging.debug("Serializing %s with %s", obj, fun)
        result = fun(obj, self)

        if isinstance(obj, types.FunctionType):
            result['_type'] = 'types.FunctionType'
        else:
            result['_type'] = "%s.%s" % (obj.__module__, type(obj).__name__)

        self._working.remove(oid)
        return result

    def _dispatch(self, obj):
        if hasattr(obj, '__gluestate__'):
            return type(obj).__gluestate__

        for typ in type(obj).mro():
            if typ in self.dispatch:
                return self.dispatch[typ]

        raise GlueSerializeError("Don't know how to serialize"
                                 " %r of type %s" % (obj, type(obj)))

    def _disambiguate(self, name):
        if name not in self._objs:
            return name

        for i in count(0):
            newname = "%s_%i" % (name, i)
            if newname not in self._objs:
                return newname

    @classmethod
    def restore(cls, obj):
        raise NotImplementedError()

    @classmethod
    def dumpo(cls, obj):
        """
        Dump an object (with needed dependencies) into a
        JSON Serializable data structure.

        Note: If eventually dumping to a string or file, dumps or dump
              are more robust
        """
        self = cls()
        oid = self.id(obj)
        return self.do_all()

    @staticmethod
    def json_default(o):
        """Default JSON enconding, to handle some special cases

        In particular, coerces numpy scalars to the equivalent
        python types

        Can be used as default kwarg in json.dumps/json.dump
        """
        if np.isscalar(o):
            return np.asscalar(o)  # coerce to pure-python type
        return o

    @classmethod
    def dumps(cls, obj, indent=None):
        result = cls.dumpo(obj)
        return json.dumps(result, indent=indent, default=cls.json_default)

    @classmethod
    def dump(cls, obj, outfile):
        result = cls.dumpo(obj)
        return json.dump(result, outfile, default=cls.json_default)


class GlueUnSerializer(object):
    dispatch = {}

    def __init__(self, record_set):
        self._names = {}  # map id(object) -> name
        self._objs = {}   # map name -> object
        self._working = set()
        self._rec = json.loads(record_set)
        self._stack_depth = len(extract_stack())

    @classmethod
    def unserializes(cls, *objs):
        def decorator(func):
            for obj in objs:
                cls.dispatch[obj] = func
            return func
        return decorator

    def _dispatch(self, rec):
        typ = _lookup(rec['_type'])

        if hasattr(typ, '__setgluestate__'):
            return typ.__setgluestate__

        for t in typ.mro():
            if t in self.dispatch:
                return self.dispatch[t]

        raise GlueSerializeError("Don't know how to load"
                                 " objects of type %s" % typ)

    def register_object(self, obj_id, obj):
        self._objs[obj_id] = obj

    def object(self, obj_id):
        if isinstance(obj_id, basestring):
            if obj_id in self._objs:
                return self._objs[obj_id]

            d = (len(extract_stack()) - self._stack_depth) / 2
            print '|  ' * d + 'unserialize %s' % obj_id

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

        if isinstance(obj_id, basestring):
            self._objs[obj_id] = obj
            self._working.remove(obj_id)

        return obj


saver = GlueSerializer.serializes
loader = GlueUnSerializer.unserializes


@saver(CompositeSubsetState)
def _save_composite_subset_state(state, context):
    return dict(op=OPSYM[state.op], state1=context.id(state.state1),
                state2=context.id(state.state2))


@loader(CompositeSubsetState)
def _load_composite_subset_state(rec, context):
    result = CompositeSubsetState(context.object(rec['state1']),
                                  context.object(rec['state2']))
    result.op = SYMOP[rec['op']]
    return result


@saver(SubsetState)
def _save_subset_state(state, context):
    return {}


@loader(SubsetState)
def _load_subset_state(rec, context):
    return SubsetState()


@saver(RoiSubsetState)
def _save_roi_subset_state(state, context):
    return dict(xatt=context.id(state.yatt),
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
                op=OPSYM[state.operator])


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
    return dict(data=map(context.id, dc),
                links=map(context.id, dc.links),
                cids=map(context.id, cids),
                components=map(context.id, components))

@loader(DataCollection)
def _load_data_collection(rec, context):
    dc = DataCollection(map(context.object, rec['data']))
    for link in rec['links']:
        dc.add_link(context.object(link))
    return dc

@saver(Data)
def _save_data(data, context):

    return dict(components=dict((context.id(c),
                                 context.id(data.get_component(c)))
                                for c in data.components
                                if not isinstance(data.get_component(c),
                                                  CoordinateComponent)),
                subsets=[context.id(s) for s in data.subsets],
                label=data.label)


@loader(Data)
def _load_data(rec, context):
    label = rec['label']
    result = Data(label=label)
    # xxx coordinates
    comps = [map(context.object, [k, v])
             for k, v in rec['components'].items()]
    comps = sorted(comps,
                   key=lambda x: isinstance(x[1], (DerivedComponent,
                                                   CoordinateComponent)))
    for cid, comp in comps:
        result.add_component(comp, cid)

    for s in rec['subsets']:
        result.add_subset(context.object(s))

    return result


@saver(ComponentID)
def _save_component_id(cid, context):
    return dict(label=cid.label, hidden=cid.hidden)


@loader(ComponentID)
def _load_component_id(rec, context):
    return ComponentID(rec['label'], rec['hidden'])


@saver(Component)
def _save_component(component, context):
    fac, args, kwargs, oid = component.creation_info
    fac = context.do(fac)

    return dict(factory=fac,
                factory_args=args,
                factory_kwargs=kwargs,
                output_index=oid)


@loader(Component)
def _load_component(rec, context):
    fac = context.object(rec['factory'])
    args = rec['factory_args']
    kwargs = rec['factory_kwargs']
    oid = rec['output_index']
    result = fac(*args, **kwargs)
    for o in oid:
        result = result[o]
    return result


@saver(DerivedComponent)
def _save_derived_component(component, context):
    return dict(link=context.id(component.link))


@loader(DerivedComponent)
def _load_derived_component(rec, context):
    return DerivedComponent(None, link=context.object(rec['link']))


@saver(ComponentLink)
def _save_component_link(link, context):
    frm = map(context.id, [context.id(f) for f in link.get_from_ids()])
    to = map(context.id, [link.get_to_id()])
    using = context.do(link.get_using())
    inverse = context.do(link.get_inverse())
    return dict(frm=frm, to=to, using=using, inverse=inverse)


@loader(ComponentLink)
def _load_component_link(rec, context):
    frm = map(context.object, rec['frm'])
    to = map(context.object, rec['to'])[0]
    using = context.object(rec['using'])
    inverse = context.object(rec['inverse'])
    return ComponentLink(frm, to, using, inverse)


@saver(CoordinateComponentLink)
def _save_coordinate_component_link(link, context):
    frm = map(context.id, [context.id(f) for f in link.get_from_ids()])
    to = map(context.id, [link.get_to_id()])
    coords = context.id(link.coords)
    index = link.index
    pix2world = link.pixel2world
    return dict(frm=frm, to=to, coords=coords, index=index,
                pix2world=pix2world)


@loader(CoordinateComponentLink)
def _load_coordinate_component_link(rec, context):
    frm = map(context.object, rec['frm'])
    to = map(context.object, rec['to'])[0]  #XXX why is this a list?
    coords = context.object(rec['coords'])
    index = rec['index']
    pix2world = rec['pix2world']
    return CoordinateComponentLink(frm, to, coords, index, pix2world)


@saver(types.FunctionType)
def _save_function(function, context):
    ref = "%s.%s" % (function.__module__, function.__name__)
    if _lookup(ref) is function:
        return {'function': "%s.%s" % (function.__module__, function.__name__)}
    return {'pickle': dumps(function).encode('base64')}


@loader(types.FunctionType)
def _load_function(rec, context):
    if 'pickle' in rec:
        return loads(rec['pickle'].decode('base64'))
    return _lookup(rec['function'])


@saver(core.Session)
def _save_session(session, context):
    # we will rely on GlueApplication to re-populate
    return {}
