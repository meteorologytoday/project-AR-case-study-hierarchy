import numpy as np

def findfirst(a):
    return np.argmax(a)

def findlast(a):
    return (len(a) - 1) - np.argmax(a[::-1])


def findArgRange(arr, lb, ub):
    if lb > ub:
        raise Exception("Lower bound should be no larger than upper bound")

    if np.any( (arr[1:] - arr[:-1]) <= 0 ):
        raise Exception("input array should be monotonically increasing")

    idx = np.logical_and((lb <= arr),  (arr <= ub))

    idx_low = findfirst(idx)
    idx_max = findlast(idx)

    return idx_low, idx_max



def findRegion_latlon(lat_arr, lat_rng, lon_arr, lon_rng):

    lat_beg, lat_end = findArgRange(lat_arr, lat_rng[0], lat_rng[1])
    lon_beg, lon_end = findArgRange(lon_arr, lon_rng[0], lon_rng[1])

    return (lon_beg, lon_end, lat_beg, lat_end)


