from .component_link import ComponentLink

__LINK_FUNCTIONS__ = []

def identity(x):
    return x

__LINK_FUNCTIONS__.append(identity)

def _partial_result(func, index):
    return lambda *args, **kwargs: func(*args, **kwargs)[index]


def link_same(cid1, cid2):
    """ Return 2 ComponentLinks to represent
    that two componentIDs describe the same piece of information """
    return (ComponentLink([cid1], cid2), ComponentLink([cid2], cid1))


def link_twoway(cid1, cid2, forwards, backwards):
    """ Return 2 links that connect input ComponentIDs in both directions

    :param cid1: First ComponentID to link
    :param cid2: Second ComponentID to link
    :param forwards: Function which maps cid1 to cid2 (e.g. cid2=f(cid1))
    :param backwards: Function which maps cid2 to cid1 (e.g. cid1=f(cid2))

    :rtype: Tuple of :class:`~glue.core.ComponentLink`

    Returns two ComponentLinks, specifying the link in each direction
    """
    return (ComponentLink([cid1], cid2, forwards),
            ComponentLink([cid2], cid1, backwards))


def multilink(cids_left, cids_right, forwards=None, backwards=None):
    """ Compute all the ComponentLinks to link groups of ComponentIDs

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
    if forwards is None and backwards is None:
        raise TypeError("Must supply either forwards or backwards")

    result = []
    if forwards is not None:
        for i, r in enumerate(cids_right):
            func = _partial_result(forwards, i)
            result.append(ComponentLink(cids_left, r, func))

    if backwards is not None:
        for i, l in enumerate(cids_left):
            func = _partial_result(backwards, i)
            result.append(ComponentLink(cids_right, l, func))

    return tuple(result)


try:
    from aplpy.wcs_util import fk52gal, gal2fk5

    def galactic2ecliptic(l, b, ra, dec):
        """ Return the ComponentLinks that map galactic and ecliptic
        coordinates

        :param l: ComponentID for galactic longitude
        :param b: ComponentID for galactic latitude
        :param ra: ComponentID for J2000 Right Ascension
        :param dec: ComponentID for J2000 Declination

        Returns 4 :class:`~glue.core.ComponentLink` objects which link
        these ComponentIDs
        """
        return multilink([ra, dec], [l, b], fk52gal, gal2fk5)

    def radec2glon(ra, dec):
        """Compute galactic longitude from right ascension and declination"""
        return fk52gal(ra, dec)[0]

    def radec2glat(ra, dec):
        """Compute galactic latitude from right ascension and declination"""
        return fk52gal(ra, dec)[1]

    def lb2ra(lat, lon):
        """Compute right ascension from galactic longitude and latitude"""
        return gal2fk5(lat, lon)[0]

    def lb2dec(lat, lon):
        """Compute declination from galactic longitude and latitude"""
        return gal2fk5(lat, lon)[1]

    __LINK_FUNCTIONS__.extend([radec2glon, radec2glat, lb2ra, lb2dec])

except ImportError:
    pass
