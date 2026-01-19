"""
This module provides several classes and LinkCollection classes to
assist in linking data.

The :class:`LinkCollection` class and its sub-classes are factories to create
multiple ComponentLinks easily. They are meant to be passed to
:meth:`~glue.core.data_collection.DataCollection.add_link()`
"""

import types

from glue.config import link_function, link_helper

from glue.core.data import ComponentID
from glue.core.component_link import ComponentLink

from inspect import getfullargspec


__all__ = ['LinkCollection', 'LinkSame', 'LinkTwoWay', 'MultiLink',
           'LinkAligned', 'BaseMultiLink', 'ManualLinkCollection',
           'JoinLink', 'validate_link', 'LinkValidationError']


@link_function("Link conceptually identical components",
               output_labels=['y'])
def identity(x):
    return x


@link_function("Convert between linear measurements and volume",
               output_labels=['volume'])
def lengths_to_volume(width, height, depth):
    return width * height * depth


class PartialResult(object):

    def __init__(self, func, index, name_prefix=""):
        self.func = func
        self.index = index
        self.__name__ = '%s%s_%i' % (name_prefix, func.__name__, index + 1)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)[self.index]

    def __gluestate__(self, context):
        return dict(func=context.do(self.func), index=self.index)

    @classmethod
    def __setgluestate__(cls, rec, context):
        return cls(context.object(rec['func']), rec['index'])


def _toid(arg):
    """Coerce the input to a ComponentID, if possible"""
    if isinstance(arg, ComponentID):
        return arg
    elif isinstance(arg, str):
        return ComponentID(arg)
    else:
        raise TypeError('Cannot be cast to a ComponentID: %s' % arg)


class LinkCollection(object):
    """
    A collection of links between two datasets.

    Parameters
    ----------
    data1 : `~glue.core.data.Data`
        The first dataset being linked
    data2 : `~glue.core.data.Data`
        The second dataset being linked
    cids1 : list of `~glue.core.component_id.ComponentID`
        The set of `~glue.core.component_id.ComponentID` in ``data1`` which
        can be used to parameterize the links. Note that the links can also
        use other IDs, but the ones defined here are the ones that can be
        modified through e.g. the graphical link editor.
    cids2 : list of `~glue.core.component_id.ComponentID`
        The set of `~glue.core.component_id.ComponentID` in ``data2``. This is
        defined as for ``cids1``.
    """

    # The following is a short name to be used for the link, which is used
    # in e.g. drop-down menus in link editors.
    display = 'Collection of links'

    # The following can be a paragraph description explaining how the set
    # of links works
    description = ''

    # The following are lists of human-readable names for the component IDs
    # to be specified in the initializer. For this base class, these are
    # empty, but can be overridden in sub-classes.
    labels1 = []
    labels2 = []

    def __init__(self, data1=None, data2=None, cids1=None, cids2=None):

        self.cids1 = cids1 or []
        self.cids2 = cids2 or []

        if data1 is None:
            if len(self.cids1) == 0:
                self.data1 = None
            else:
                self.data1 = self.cids1[0].parent
        else:
            self.data1 = data1

        if data2 is None:
            if len(self.cids2) == 0:
                self.data2 = None
            else:
                self.data2 = self.cids2[0].parent
        else:
            self.data2 = data2

        self._links = []

    def __iter__(self):
        for link in self._links:
            yield link

    def __len__(self):
        return len(self._links)

    def __getitem__(self, item):
        return self._links[item]

    def __contains__(self, cid):
        for link in self:
            if cid in link:
                return True
        return False

    def __gluestate__(self, context):
        state = {}
        state['data1'] = context.id(self.data1)
        state['data2'] = context.id(self.data2)
        state['cids1'] = context.id(self.cids1)
        state['cids2'] = context.id(self.cids2)
        return state

    @classmethod
    def __setgluestate__(cls, rec, context):
        if 'data1' in rec:
            self = cls(data1=context.object(rec['data1']),
                       data2=context.object(rec['data2']),
                       cids1=context.object(rec['cids1']),
                       cids2=context.object(rec['cids2']))
        else:  # glue-core <0.15
            cids = context.object(rec['cids'])
            cids1 = [context.object(c) for c in cids[:len(cls.labels1)]]
            cids2 = [context.object(c) for c in cids[len(cls.labels1):]]
            self = cls(cids1=cids1, cids2=cids2)
        return self


class ManualLinkCollection(object):
    """
    A collection of links between two datasets.

    This class is intended for manual link collections, i.e. collections where
    the caller manually adds and removes individual links. These links can be
    between any component IDs as long as they link component IDs between the
    two specified datasets.

    Parameters
    ----------
    data1 : `~glue.core.data.Data`
        The first dataset being linked
    data2 : `~glue.core.data.Data`
        The second dataset being linked
    links : list
        The initial links to add to the collection.
    """

    display = 'Custom list of links'
    description = 'This is a list of links that has been manually constructed'

    def __init__(self, data1=None, data2=None, links=None):
        super(ManualLinkCollection, self).__init__(data1=data1, data2=data2)
        self._links[:] = links or []

    def append(self, link):
        self._links.append(link)

    def extend(self, links):
        self._links.extend(links)

    def __gluestate__(self, context):
        state = super(ManualLinkCollection, self).__gluestate__(context)
        state['values'] = context.id(self._links)
        return state

    @classmethod
    def __setgluestate__(cls, rec, context):
        self = super(ManualLinkCollection, cls).__setgluestate__(rec, context)
        self._values[:] = context.object(rec['values'])
        return self


class BaseMultiLink(LinkCollection):
    """
    A link collection that is generated on-the-fly based on forward and
    backward transformation functions and lists of input/output component IDs.

    The input parameters are as for
    :class:`~glue.core.link_helpers.LinkCollection`. Sub-classes should
    override the :meth:`~glue.core.link_helpers.BaseMultiLink.forwards` and
    :meth:`~glue.core.link_helpers.BaseMultiLink.backwards` methods.
    """

    # Some sub-classes take only data and don't need CIDs, so we have a flag
    # for this in case other parts of glue need to know.
    cid_independent = False

    # TODO: could add a metaclass to set labels1 and labels2 automatically

    def __init__(self, cids1=None, cids2=None, data1=None, data2=None):

        super(BaseMultiLink, self).__init__(data1=data1, data2=data2,
                                            cids1=cids1, cids2=cids2)

        links = []

        if self.forwards is not None:
            if self.forwards is identity:
                links.append(ComponentLink(cids1, cids2[0]))
            elif len(cids2) == 1:
                links.append(ComponentLink(cids1, cids2[0], self.forwards))
            else:
                for i, r in enumerate(cids2):
                    func = PartialResult(self.forwards, i, name_prefix=self.__class__.__name__ + ".")
                    links.append(ComponentLink(cids1, r, using=func,
                                               input_names=self.labels1))

        if self.backwards is not None:
            if self.backwards is identity:
                links.append(ComponentLink(cids2, cids1[0]))
            elif len(cids1) == 1:
                links.append(ComponentLink(cids2, cids1[0], self.backwards))
            else:
                for i, l in enumerate(cids1):
                    func = PartialResult(self.backwards, i, name_prefix=self.__class__.__name__ + ".")
                    links.append(ComponentLink(cids2, l, using=func,
                                               input_names=self.labels1))

        self._links[:] = links

    def forwards(self):
        raise NotImplementedError()

    def backwards(self):
        raise NotImplementedError()


class MultiLink(BaseMultiLink):
    """
    A link collection that is generated on-the-fly based on forward and
    backward transformation functions and lists of input/output component IDs.

    This is similar to :meth:`~glue.core.link_helpers.BaseMultiLink` except
    that the ``forwards`` and ``backwards`` functions are specified in the
    initializer rather than being methods of the class.

    Parameters
    ----------
    forwards : function
        Function that maps ``cids1`` to  ``cids2``. This should have
        the signature ``cids2 = forwards(*cids1)``, and is assumed
        to return a tuple. If not specified, no forward links are calculated.
    backwards : function
        The inverse function to ``forwards``. If not specified, no forward links
        are calculated.
    labels1 : list of str
        The human-readable names of the inputs to the ``forwards`` function.
        If not specified, this is set to the argument names of ``forwards``.
    labels2 : list of str
        The human-readable names of the inputs to the ``backwards`` function.
        If not specified, this is set to the argument names of ``backwards``.
    **kwargs :
        Additional arguments are passed
    """

    def __init__(self, cids1=None, cids2=None, forwards=None, backwards=None, labels1=None, labels2=None, **kwargs):

        # NOTE: we explicitly specify ``cids1`` and ``cids2`` as the two first
        # arguments for backwards-compatibility with callers that use positional
        # arguments.

        if forwards is None and backwards is None:
            raise TypeError("Must supply either forwards or backwards")

        self.forwards = forwards
        self.backwards = backwards

        # NOTE: the getattr(forwards, 'func', forwards) in the following code
        # is to make sure that things work properly if the functions are
        # PartialResult objects.

        if labels1 is None:
            if forwards is not None:
                if isinstance(forwards, types.MethodType):
                    labels1 = getfullargspec(getattr(forwards, 'func', forwards))[0][1:]
                else:
                    labels1 = getfullargspec(getattr(forwards, 'func', forwards))[0]
            else:
                raise ValueError("labels1 needs to be specified if forwards isn't")

        if labels2 is None:
            if backwards is not None:
                if isinstance(backwards, types.MethodType):
                    labels2 = getfullargspec(getattr(backwards, 'func', backwards))[0][1:]
                else:
                    labels2 = getfullargspec(getattr(backwards, 'func', backwards))[0]
            else:
                raise ValueError("labels2 needs to be specified if backwards isn't")

        self.labels1 = labels1
        self.labels2 = labels2

        super(MultiLink, self).__init__(cids1=cids1, cids2=cids2, **kwargs)

    def __gluestate__(self, context):
        state = super(MultiLink, self).__gluestate__(context)
        state['forwards'] = context.id(self._forwards)
        state['backwards'] = context.id(self._backwards)
        return state

    @classmethod
    def __setgluestate__(cls, rec, context):
        self = super(MultiLink, cls).__setgluestate__(rec, context)
        self._forwards = context.object(rec['forwards'])
        self._backwards = context.object(rec['backwards'])
        return self


class LinkSame(MultiLink):
    """
    A bi-directional identity link between two components.
    """

    display = "identity link"

    def __init__(self, cid1=None, cid2=None, **kwargs):
        if cid1 is None:
            cid1 = kwargs['cids1'][0]
        else:
            cid1 = _toid(cid1)
            kwargs['cids1'] = [cid1]

        if cid2 is None:
            cid2 = kwargs['cids2'][0]
        else:
            cid2 = _toid(cid2)
            kwargs['cids2'] = [cid2]

        kwargs['forwards'] = identity
        default_kwargs = {'data1': cid1.parent, 'data2': cid2.parent,
                          'labels1': ['x'], 'labels2': ['y']}

        for keyword, value in default_kwargs.items():
            if keyword not in kwargs:
                kwargs[keyword] = value

        self._cid1 = cid1
        self._cid2 = cid2

        super(LinkSame, self).__init__(**kwargs)

    def __gluestate__(self, context):
        state = {}
        state['cid1'] = context.id(self._cid1)
        state['cid2'] = context.id(self._cid2)
        return state

    @classmethod
    def __setgluestate__(cls, rec, context):
        self = cls(context.object(rec['cid1']), context.object(rec['cid2']))
        return self


class LinkTwoWay(MultiLink):
    """
    Return two links that connect input ComponentIDs in both directions

    Parameters
    ----------
    cid1 : `glue.core.component_id.ComponentID`
        The first ComponentID to link
    cid2 : `glue.core.component_id.ComponentID`
        The second ComponentID to link
    forwards : function
        Function which maps cid1 to cid2 (e.g. ``cid2=f(cid1)``)
    backwards : function
        Function which maps cid2 to cid1 (e.g. ``cid1=f(cid2)``)
    """

    def __init__(self, cid1=None, cid2=None, forwards=None, backwards=None, **kwargs):

        if cid1 is None:
            cid1 = kwargs['cids1']
        else:
            kwargs['data1'] = cid1.parent
            kwargs['cids1'] = [cid1]

        if cid2 is None:
            cid2 = kwargs['cids2']
        else:
            kwargs['data2'] = cid2.parent
            kwargs['cids2'] = [cid2]

        self._cid1 = cid1
        self._cid2 = cid2

        super(LinkTwoWay, self).__init__(forwards=forwards, backwards=backwards, **kwargs)

    def __gluestate__(self, context):
        state = {}
        state['cid1'] = context.id(self._cid1)
        state['cid2'] = context.id(self._cid2)
        state['forwards'] = context.id(self.forwards)
        state['backwards'] = context.id(self.backwards)
        return state

    @classmethod
    def __setgluestate__(cls, rec, context):
        self = cls(context.object(rec['cid1']),
                   context.object(rec['cid2']),
                   context.object(rec['forwards']),
                   context.object(rec['backwards']))
        return self


class LinkSameWithUnits(LinkTwoWay):

    def __init__(self, cid1=None, cid2=None, **kwargs):

        self.units1 = cid1.parent.get_component(cid1).units
        self.units2 = cid2.parent.get_component(cid2).units

        from glue.core.units import UnitConverter

        self._converter = UnitConverter()

        super().__init__(cid1=cid1, cid2=cid2, forwards=self.forwards, backwards=self.backwards)

        # Pre-check that unit conversions work properly as if there are any
        # issues it is better to report them when the link is set up rather
        # than when the links are used as issues may be hidden.
        self.forwards(1)
        self.backwards(1)

    def forwards(self, values):
        return self._converter.to_unit(self._cid1.parent, self._cid1, values, self.units2)

    def backwards(self, values):
        return self._converter.to_unit(self._cid2.parent, self._cid2, values, self.units1)


class LinkAligned(LinkCollection):
    """
    Compute all the links to specify that the input data are pixel-aligned.
    """

    def __init__(self, data1=None, data2=None):
        super(LinkAligned, self).__init__(data1=data1, data2=data2)
        if data1.shape != data2.shape:
            raise TypeError("Input data do not have the same shape")
        links = []
        for j in range(data1.ndim):
            links.extend(LinkSame(data1.pixel_component_ids[j],
                                  data2.pixel_component_ids[j]))
        self._links[:] = links


def functional_link_collection(function, labels1=None, labels2=None,
                               display=None, description=None):

    class FunctionalLinkCollection(LinkCollection):

        def __init__(self, data1=None, data2=None,
                     cids1=None, cids2=None):
            super(FunctionalLinkCollection, self).__init__(data1=data1, data2=data2,
                                                           cids1=cids1, cids2=cids2)
            self._links[:] = function(*self.cids1, *self.cids2)

    FunctionalLinkCollection.labels1 = labels1 or []
    FunctionalLinkCollection.labels2 = labels2 or []
    FunctionalLinkCollection.display = display or ''
    FunctionalLinkCollection.description = description or ''

    return FunctionalLinkCollection


@link_helper(category="Join")
# Ignore eq-without-hash: __eq__ is based on mutable objects, so this class
# should not implement __hash__ (set to None).
class JoinLink(LinkCollection):  # noqa: PLW1641
    cid_independent = False

    display = "Join on ID"
    description = "Join two datasets on a common ID. Other links \
in glue connect data columns (two datasets have 'age' columns but \
the rows are different objects), while Join on ID connects the same \
rows/items across two datasets."

    labels1 = ["Identifier in dataset 1"]
    labels2 = ["Identifier in dataset 2"]

    def __init__(self, *args, cids1=None, cids2=None, data1=None, data2=None):
        # only support linking by one value now, even though link_by_value supports multiple
        assert len(cids1) == 1
        assert len(cids2) == 1

        self.data1 = data1
        self.data2 = data2
        self.cids1 = cids1
        self.cids2 = cids2

        self._links = []

    def __str__(self):
        # The >< here is one symbol for a database join
        return '%s >< %s' % (self.cids1, self.cids2)

    def __repr__(self):
        return "<JoinLink: %s>" % self

    # Define __eq__ and __ne__ to facilitate removing
    # these kinds of links from the link_manager
    def __eq__(self, other):
        if not isinstance(other, JoinLink):
            return False
        same = ((self.data1 == other.data1) and
                (self.data2 == other.data2) and
                (self.cids1 == other.cids1) and
                (self.cids2 == other.cids2))
        flip = ((self.data1 == other.data2) and
                (self.data2 == other.data1) and
                (self.cids1 == other.cids2) and
                (self.cids2 == other.cids1))
        return same or flip

    def __ne__(self, other):
        return not self.__eq__(other)


class LinkValidationError(Exception):
    """Exception raised when link validation fails."""
    pass


def validate_link(link, data_collection=None, raise_on_error=True):
    """
    Validate a ComponentLink or LinkCollection for structural and contextual correctness.

    This function checks that a link is well-formed and, optionally, that
    all ComponentIDs it references exist in a given DataCollection.

    Parameters
    ----------
    link : `~glue.core.component_link.ComponentLink` or `~glue.core.link_helpers.LinkCollection`
        The link to validate.
    data_collection : `~glue.core.data_collection.DataCollection`, optional
        If provided, validates that all ComponentIDs referenced by the link
        exist in datasets within the collection.
    raise_on_error : bool, optional
        If True (default), raises `LinkValidationError` on validation failure.
        If False, returns a tuple of (is_valid, error_messages).

    Returns
    -------
    is_valid : bool
        Only returned if ``raise_on_error=False``. True if the link is valid.
    errors : list of str
        Only returned if ``raise_on_error=False``. List of error messages
        describing validation failures.

    Raises
    ------
    LinkValidationError
        If ``raise_on_error=True`` and validation fails.

    Examples
    --------
    Basic structural validation::

        >>> from glue.core import Data, ComponentID
        >>> from glue.core.component_link import ComponentLink
        >>> from glue.core.link_helpers import validate_link
        >>> d = Data(x=[1, 2, 3])
        >>> link = ComponentLink([d.id['x']], ComponentID('y'))
        >>> validate_link(link)  # Returns True if valid

    Contextual validation with DataCollection::

        >>> from glue.core import DataCollection
        >>> dc = DataCollection([d])
        >>> validate_link(link, data_collection=dc)

    Non-raising mode for graceful handling::

        >>> is_valid, errors = validate_link(link, raise_on_error=False)
        >>> if not is_valid:
        ...     print("Validation errors:", errors)
    """
    errors = []

    # Handle LinkCollection (which contains multiple ComponentLinks)
    if isinstance(link, LinkCollection):
        # Validate the LinkCollection structure
        errors.extend(_validate_link_collection_structure(link))

        # Validate each contained ComponentLink
        for contained_link in link:
            contained_errors = _validate_component_link(contained_link, data_collection)
            errors.extend(contained_errors)
    elif isinstance(link, ComponentLink):
        errors.extend(_validate_component_link(link, data_collection))
    else:
        errors.append(f"Expected ComponentLink or LinkCollection, got {type(link).__name__}")

    if errors:
        if raise_on_error:
            raise LinkValidationError("; ".join(errors))
        return False, errors

    if raise_on_error:
        return True
    return True, []


def _validate_link_collection_structure(link_collection):
    """
    Validate the structure of a LinkCollection.

    Parameters
    ----------
    link_collection : `~glue.core.link_helpers.LinkCollection`
        The link collection to validate.

    Returns
    -------
    errors : list of str
        List of error messages for any validation failures.
    """
    errors = []

    # Check data1 and data2 are set if cids are provided
    if link_collection.cids1 and link_collection.data1 is None:
        errors.append("LinkCollection has cids1 but data1 is None")
    if link_collection.cids2 and link_collection.data2 is None:
        errors.append("LinkCollection has cids2 but data2 is None")

    # Validate that cids belong to their respective data objects
    for i, cid in enumerate(link_collection.cids1):
        if not isinstance(cid, ComponentID):
            errors.append(f"cids1[{i}] is not a ComponentID: {type(cid).__name__}")
        elif link_collection.data1 is not None and cid.parent is not link_collection.data1:
            # Only warn if parent is set and doesn't match - parent can be None for new CIDs
            if cid.parent is not None:
                errors.append(f"cids1[{i}] ({cid}) belongs to {cid.parent.label if cid.parent else 'None'}, "
                              f"not data1 ({link_collection.data1.label})")

    for i, cid in enumerate(link_collection.cids2):
        if not isinstance(cid, ComponentID):
            errors.append(f"cids2[{i}] is not a ComponentID: {type(cid).__name__}")
        elif link_collection.data2 is not None and cid.parent is not link_collection.data2:
            if cid.parent is not None:
                errors.append(f"cids2[{i}] ({cid}) belongs to {cid.parent.label if cid.parent else 'None'}, "
                              f"not data2 ({link_collection.data2.label})")

    return errors


def _validate_component_link(link, data_collection=None):
    """
    Validate a single ComponentLink.

    Parameters
    ----------
    link : `~glue.core.component_link.ComponentLink`
        The link to validate.
    data_collection : `~glue.core.data_collection.DataCollection`, optional
        If provided, validates that ComponentIDs exist in the collection.

    Returns
    -------
    errors : list of str
        List of error messages for any validation failures.
    """
    errors = []

    # Structural validation
    from_ids = link.get_from_ids()
    to_id = link.get_to_id()

    # Check from_ids
    if not isinstance(from_ids, list):
        errors.append(f"get_from_ids() should return a list, got {type(from_ids).__name__}")
    elif len(from_ids) == 0:
        errors.append("Link has no input ComponentIDs (comp_from is empty)")
    else:
        for i, cid in enumerate(from_ids):
            if not isinstance(cid, ComponentID):
                errors.append(f"comp_from[{i}] is not a ComponentID: {type(cid).__name__}")

    # Check to_id
    if to_id is None:
        errors.append("Link has no target ComponentID (comp_to is None)")
    elif not isinstance(to_id, ComponentID):
        errors.append(f"comp_to is not a ComponentID: {type(to_id).__name__}")

    # Check using function
    using = link.get_using()
    if using is not None and not callable(using):
        errors.append(f"'using' function is not callable: {type(using).__name__}")

    # Check inverse function (if present)
    inverse = link.get_inverse()
    if inverse is not None and not callable(inverse):
        errors.append(f"'inverse' function is not callable: {type(inverse).__name__}")

    # Contextual validation (if data_collection provided)
    if data_collection is not None and not errors:
        errors.extend(_validate_link_in_collection(link, data_collection))

    return errors


def _validate_link_in_collection(link, data_collection):
    """
    Validate that a link's ComponentIDs exist in a DataCollection.

    Parameters
    ----------
    link : `~glue.core.component_link.ComponentLink`
        The link to validate.
    data_collection : `~glue.core.data_collection.DataCollection`
        The data collection to validate against.

    Returns
    -------
    errors : list of str
        List of error messages for any validation failures.
    """
    errors = []

    # Build a set of all valid ComponentIDs in the collection
    all_cids = set()
    for data in data_collection:
        all_cids.update(data.component_ids())

    # Check from_ids
    for cid in link.get_from_ids():
        if cid not in all_cids:
            parent_info = f" (parent: {cid.parent.label})" if cid.parent else ""
            errors.append(f"ComponentID '{cid}'{parent_info} not found in DataCollection")

    # Check to_id - it's okay if to_id doesn't exist yet (it will be created as a derived component)
    # But we should check that its parent data exists in the collection if parent is set
    to_id = link.get_to_id()
    if to_id.parent is not None:
        if to_id.parent not in data_collection:
            errors.append(f"Target ComponentID's parent data '{to_id.parent.label}' "
                          "not found in DataCollection")

    return errors
