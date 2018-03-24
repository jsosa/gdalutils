# inst: university of bristol
# auth: jeison sosa
# mail: sosa.jeison@gmail.com / j.sosa@bristol.ac.uk

# Jupyter lab options:
# load_ext Cython
# %%cython --compile-args=-fopenmp --compile-args=-O3 --link-args=-fopenmp --force -a

# Compile via console:
# python setup.py build_ext --inplace

from sys import exit
cimport cython
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport sin,asin,cos,pow,sqrt,M_PI

@cython.wraparound(False)
@cython.boundscheck(False)
def haversine(np.float64_t lat1, np.float64_t lng1, np.float64_t lat2, np.float64_t lng2):
    
    cdef np.int32_t AVG_EARTH_RADIUS = 6371  # in km
    cdef np.float64_t lat,lng,d,h

    # Convert all latitudes/longitudes from decimal degrees to radians
    lat1 = lat1*M_PI/180
    lng1 = lng1*M_PI/180
    lat2 = lat2*M_PI/180
    lng2 = lng2*M_PI/180

    lat = lat2 - lat1
    lng = lng2 - lng1
    d = pow(sin(lat*0.5),2) + cos(lat1)*cos(lat2)*pow(sin(lng*0.5),2)
    h = 2*AVG_EARTH_RADIUS*asin(sqrt(d))
    
    return h

@cython.wraparound(False)
@cython.boundscheck(False)
def haversine_array(np.float32_t[:] lat1, np.float32_t[:] lng1, np.float32_t lat2, np.float32_t lng2):
    
    if lat1.shape[0] != lng1.shape[0]:
        exit("ERROR Lat1 and Lng1 have different dimmensions")

    cdef int i
    cdef int N = lat1.shape[0]
    cdef int AVG_EARTH_RADIUS = 6371
    cdef np.float32_t[:] dis = np.zeros_like(lat1)
    cdef np.float32_t lat,lng,d,h,lats2,lngs2,lngs1,lats1

    for i in prange(N, num_threads=4, nogil=True):

        # Convert all latitudes/longitudes from decimal degrees to radians
        lats1 = lat1[i]*M_PI/180
        lngs1 = lng1[i]*M_PI/180
        lats2 = lat2*M_PI/180
        lngs2 = lng2*M_PI/180

        lat = lats2 - lats1
        lng = lngs2 - lngs1
        d = pow(sin(lat*0.5),2) + cos(lats1)*cos(lats2)*pow(sin(lng*0.5),2)
        h = 2*AVG_EARTH_RADIUS*asin(sqrt(d))

        dis[i] = h

    return dis
