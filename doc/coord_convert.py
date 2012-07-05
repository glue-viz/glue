from kapteyn import celestial

def radec2glat(ra, dec):
    lonlat = celestial.sky2sky( (celestial.eq, celestial.fk5), celestial.gal,
                               ra.flat, dec.flat).A
    return lonlat[:,0].ravel()

def radec2glon(ra, dec):
    lonlat = celestial.sky2sky( (celestial.eq, celestial.fk5), celestial.gal,
                               ra.flat, dec.flat).A
    return lonlat[:, 1].ravel()

def lonlat2ra(lon, lat):
    radec = celestial.sky2sky( (celestial.eq, celestial.fk5), celestial.gal,
                               lon.flat, lat.flat).A
    return radec[:, 0].ravel()

def lonlat2dec(lon, lat):
    radec = celestial.sky2sky( (celestial.eq, celestial.fk5), celestial.gal,
                               ra.flat, dec.flat).A
    return radec[:, 1].ravel()
