"""
Mark Yamane, 6/17/2021

Helper functions used to analyze data from the WOA 18 dataset.
"""

from netCDF4 import Dataset
import numpy as np

def indexOfCoord(coord, target_coord):
    """
    Find the index of a given latitude

    Input:      coord           latitudes or longitudes as a 2-D NumPy Array
    Reurn:      i_coord         closest index of a given latitude or longitude
                actual_coord    actual latitude or longitude at i_coord
    """
    coord_lo = coord[0,0]
    i_coord = int((target_coord - coord_lo)/1.)
    actual_coord = coord[i_coord]

    return i_coord, actual_coord

def getLatSlice(lons, lats, deps, vs, target_lat):
    """
    Input:      lons    longitudes as a 1-D NumPy Array
                lats    latitudes as a 1-D NumPy Array
                deps    latitudes as a 1-D NumPy Array
                vs      3-D NumPy Array for a variable (lon, lat, dep)

    Return:     lons    2-D longitudes reshaped for mapping
                lats    2-D latitudes reshaped for mapping
    """
    # create a full grid for each dimension
    lons, lats = np.meshgrid(lons, lats)

    tempDep = []
    for lon in lons:
        tempDep.append(deps)
    deps = np.array(tempDep)

    # find index of target latitude
    i_lat, actual_lat = indexOfCoord(lats, target_lat)

    # create full grid for a latitudinal slice w/ depth
    lonVert = []
    for dep in deps[0,:]:
        lonVert.append(lons[i_lat,:])
    
    # format data for plot
    lonVert = np.array(lonVert)
    deps = np.transpose(deps)
    v_slice = vs[:, i_lat, :]

    return deps, lonVert, v_slice, actual_lat

# function for blank map
def getLand(vs):
    # mask land and ocean based on variable values
    land = []
    for i in range(len(vs)):
        land.append(np.ma.getmask(vs[i]))
    land = np.array(land)
    land = np.ma.masked_where(land > 0.3, land)
    return land