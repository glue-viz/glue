""" This module provides several classes and LinkCollection classes to
assist in linking data.

The functions in this class (and stored in the __LINK_FUNCTIONS__
list) define common coordinate transformations. They are meant to be
used for the `using` parameter in :class:`ComponentLink` instances.

The LinkCollection class and its sublcasses are factories to create
multiple ComponentLinks easily. They are meant to be passed to
:func:`DataCollection.add_link()`
"""
from functools import wraps

from .component_link import ComponentLink
from .data import ComponentID

__LINK_FUNCTIONS__ = []
__LINK_HELPERS__ = []


def identity(x):
    return x
identity.output_args = ['y']

__LINK_FUNCTIONS__.append(identity)


def _partial_result(func, index):
    @wraps(func)
    def getter(*args, **kwargs):
        return func(*args, **kwargs)[index]
    return getter


def _toid(arg):
    """Coerce the input to a ComponentID, if possible"""
    if isinstance(arg, ComponentID):
        return arg
    elif isinstance(arg, basestring):
        return ComponentID(arg)
    else:
        raise TypeError('Cannot be cast to a ComponentID: %s' % arg)


class LinkCollection(list):
    pass


class LinkSame(LinkCollection):
    """
    Return 2 ComponentLinks to represent that two componentIDs
    describe the same piece of information
    """
    def __init__(self, cid1, cid2):
        self.append(ComponentLink([_toid(cid1)], _toid(cid2)))
        self.append(ComponentLink([_toid(cid2)], _toid(cid1)))


class LinkTwoWay(LinkCollection):
    def __init__(self, cid1, cid2, forwards, backwards):
        """ Return 2 links that connect input ComponentIDs in both directions

        :param cid1: First ComponentID to link
        :param cid2: Second ComponentID to link
        :param forwards: Function which maps cid1 to cid2 (e.g. cid2=f(cid1))
        :param backwards: Function which maps cid2 to cid1 (e.g. cid1=f(cid2))

        :rtype: Tuple of :class:`~glue.core.ComponentLink`

        Returns two ComponentLinks, specifying the link in each direction
        """
        self.append(ComponentLink([_toid(cid1)], _toid(cid2), forwards))
        self.append(ComponentLink([_toid(cid2)], _toid(cid1), backwards))


class MultiLink(LinkCollection):
    """
    Compute all the ComponentLinks to link groups of ComponentIDs

    Uses functions assumed to output tuples

    :param cids_left: first collection of ComponentIDs
    :param cids_right: second collection of ComponentIDs
    :param forwards:
       Function that maps cids_left to cids_right. Assumed to have signature
       cids_right = forwards(*cids_left), and assumed to return a tuple.
       If not provided, the relevant ComponentIDs will not be generated
    :param backwards:
       The inverse function to forwards. If not provided, the relevant
       ComponentIDs will not be generated

    Returns a collection of :class:`~glue.core.ComponentLink`
    objects.
    """

    def __init__(self, cids_left, cids_right, forwards=None, backwards=None):
        cids_left = map(_toid, cids_left)
        cids_right = map(_toid, cids_right)

        if forwards is None and backwards is None:
            raise TypeError("Must supply either forwards or backwards")

        if forwards is not None:
            for i, r in enumerate(cids_right):
                func = _partial_result(forwards, i)
                self.append(ComponentLink(cids_left, r, func))

        if backwards is not None:
            for i, l in enumerate(cids_left):
                func = _partial_result(backwards, i)
                self.append(ComponentLink(cids_right, l, func))


class LinkAligned(LinkCollection):
    """Compute all the links to specify that the input data are pixel-aligned

    :param data: An iterable of :class:`~glue.core.Data` instances
    that are aligned at the pixel level. They must be the same shape.
    """
    def __init__(self, data):
        shape = data[0].shape
        ndim = data[0].ndim
        for i, d in enumerate(data[1:]):
            if d.shape != shape:
                raise TypeError("Input data do not have the same shape")
            for j in range(ndim):
                self.extend(LinkSame(data[0].get_pixel_component_id(j),
                                     data[i + 1].get_pixel_component_id(j)))


try:
    from aplpy.wcs_util import fk52gal, gal2fk5

    class Galactic2Equatorial(MultiLink):
        """
        Instantiate a ComponentList with four ComponentLinks that map galactic
        and equatorial coordinates

        :param l: ComponentID for galactic longitude
        :param b: ComponentID for galactic latitude
        :param ra: ComponentID for J2000 Right Ascension
        :param dec: ComponentID for J2000 Declination

        Returns a :class:`~glue.core.LinkCollection` object which links
        these ComponentIDs
        """

        #attributes used by the Gui
        info_text = """Link Galactic and Equatorial coordinates"""
        input_args = ['l', 'b', 'ra', 'dec']

        def __init__(self, l, b, ra, dec):
            MultiLink.__init__(self, [ra, dec], [l, b], fk52gal, gal2fk5)

    def radec2glon(ra, dec):
        """Compute galactic longitude from right ascension and declination"""
        return fk52gal(ra, dec)[0]
    radec2glon.output_args = ['l']

    def radec2glat(ra, dec):
        """Compute galactic latitude from right ascension and declination"""
        return fk52gal(ra, dec)[1]
    radec2glat.output_args = ['b']

    def lb2ra(lat, lon):
        """Compute right ascension from galactic longitude and latitude"""
        return gal2fk5(lat, lon)[0]
    lb2ra.output_args = ['ra']

    def lb2dec(lat, lon):
        """Compute declination from galactic longitude and latitude"""
        return gal2fk5(lat, lon)[1]
    lb2dec.output_args = ['dec']

    __LINK_FUNCTIONS__.extend([radec2glon, radec2glat, lb2ra, lb2dec])
    __LINK_HELPERS__.append(Galactic2Equatorial)
except ImportError:
    pass
