import numpy as np
from .geometry import path
from astropy import coordinates
from astropy import units as u
import re

csystems = {'galactic':coordinates.Galactic,
            'fk5':coordinates.FK5,
            'fk4':coordinates.FK4,
            'icrs':coordinates.ICRS}
cel_systems = ['fk5','fk4','icrs']
# ecliptic, detector, etc. not supported (because I don't know what they mean)
# (or with ecliptic, how to deal with them)
all_systems = cel_systems+['galactic','image','physical']

class SimpleRegion(object):
    def __init__(self, coord_list, coord_format, name):
        self.name = name
        self.coord_format = coord_format
        self.coord_list = coord_list

    def __repr__(self):
        return "Region: {0}, {1}, {2}".format(self.name, self.coord_list,
                                              self.coord_format)


valid_regions = ['line', 'segment', 'vector']
valid_region_re = [re.compile("^"+n) for n in valid_regions]

def simple_region_parser(regionstring, coord_format):
    rs = regionstring.lstrip("# ")

    rtype = None
    for rt, rre in zip(valid_regions, valid_region_re):
        if rre.search(rs):
            rtype = rt
            break

    if rtype is None:
        # not a usable region
        return

    coordre = re.compile("^[a-z]*\((.*)\)")
    coord_list = coordre.findall(rs)
    if len(coord_list) != 1:
        raise ValueError("Invalid region")

    coords = coord_list[0].split(",")

    outcoords = []
    for ii,cs in enumerate(coords):
        if coord_format in csystems:
            if ":" in cs:
                # sexagesimal
                if coord_format in cel_systems and ii % 2 == 0:
                    # odd, celestial = RA = hours
                    crd = coordinates.Angle(cs, unit=u.hour)
                else:
                    crd = coordinates.Angle(cs, unit=u.deg)
            else:
                try:
                    # if it's a float, it's in degrees
                    crd = float(cs) * u.deg
                except ValueError:
                    crd = coordinates.Angle(cs)
        else:
            # assume pixel units
            crd = float(cs)
        outcoords.append(crd)

    reg = SimpleRegion(coord_list=outcoords, coord_format=coord_format,
                       name=rtype)

    return reg

def load_regions_file(rfile):
    with open(rfile,'r') as fh:
        lines = fh.readlines()
    return load_regions_stringlist(lines)

def load_regions_stringlist(lines):

    coord_format = None
    for line in lines:
        if line.strip() in all_systems:
            coord_format = line.strip()
            break
    if coord_format is None:
        raise ValueError("No valid coordinate format found.")

    regions_ = [simple_region_parser(line, coord_format) for line in lines]
    regions = [r for r in regions_ if r is not None]
    
    return regions


def line_to_path(region):
    """
    Convert a line or segment to a path
    """

    l,b = None,None

    endpoints = []

    for x in region.coord_list:
        if l is None:
            if hasattr(x,'unit'):
                l = x.to(u.deg).value
            else:
                l = x
        else:
            if hasattr(x,'unit'):
                b = x.to(u.deg).value
            else:
                b = x
            if l is not None and b is not None:
                if hasattr(b,'unit') or hasattr(l,'unit'):
                    raise TypeError("Can't work with a list of quantities")
                endpoints.append((l,b))
                l,b = None,None
            else:
                raise ValueError("unmatched l,b")

    lbarr = np.array(endpoints)
    C = csystems[region.coord_format](lbarr[:,0]*u.deg, lbarr[:,1]*u.deg)

    # TODO: add widths for projection

    p = path.Path(C)

    return p

def vector_to_path(vector_region):
    """
    Convert a vector region to a path

    # vector(48.944348,-0.36432694,485.647",124.082) vector=1
    """

    x,y = vector_region.coord_list[:2]
    length = vector_region.coord_list[2]
    angle = vector_region.coord_list[3]
    
    C1 = csystems[vector_region.coord_format](x, y)
    dx,dy = length * np.cos(angle), length * np.sin(angle)
    # -dx because we're in the flippy coordsys
    C2 = csystems[vector_region.coord_format](C1.spherical.lon - dx, C1.spherical.lat + dy)

    C = csystems[vector_region.coord_format]([C1.spherical.lon.deg,C2.spherical.lon.deg]*u.deg,
                                             [C1.spherical.lat.deg,C2.spherical.lat.deg]*u.deg)

    p = path.Path(C)

    return p

region_converters = {'line':line_to_path, 'segment':line_to_path,
                     'vector':vector_to_path}

def paths_from_regfile(regfile):
    """
    Given a ds9 region file, extract pv diagrams for each:
        group of points [NOT IMPLEMENTED]
        panda [NOT IMPLEMENTED]
        vector [NOT IMPLEMENTED]
        segment [NOT IMPLEMENTED]
        group of lines [NOT IMPLEMENTED]
    """
    #import pyregion
    #regions = pyregion.open(regfile)
    regions = load_regions_file(regfile)
    return paths_from_regions(regions)

def paths_from_regions(regions):
    paths = [region_converters[r.name](r)
             for r in regions
             if r.name in region_converters]
    return paths
