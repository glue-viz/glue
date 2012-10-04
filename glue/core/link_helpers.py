from .component_link import ComponentLink

__LINK_FUNCTIONS__ = []


def identity(x):
    return x

__LINK_FUNCTIONS__.append(identity)


def _partial_result(func, index):
    return lambda *args, **kwargs: func(*args, **kwargs)[index]


class LinkCollection(list):
    pass


class LinkSame(LinkCollection):
    """
    Return 2 ComponentLinks to represent that two componentIDs
    describe the same piece of information
    """
    def __init__(self, cid1, cid2):
        self.append(ComponentLink([cid1], cid2))
        self.append(ComponentLink([cid2], cid1))


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
        self.append(ComponentLink([cid1], cid2, forwards))
        self.append(ComponentLink([cid2], cid1, backwards))


class MultiLink(LinkCollection):
    """
    Cmpute all the ComponentLinks to link groups of ComponentIDs

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
        def __init__(self, l, b, ra, dec):
            MultiLink.__init__(self, [ra, dec], [l, b], fk52gal, gal2fk5)

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
