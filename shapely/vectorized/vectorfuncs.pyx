"""Vectorized versions of all GEOS functions. This module implements a
vectorized version of topology.py, predicates.py, and impl.py.
"""


import cython
cimport cpython.array
import numpy as np
cimport numpy as np
include "../_geos.pxi"
from ctypes import byref, c_double
from shapely.geos import TopologicalError, lgeos
from shapely.impl import *


class ValidatingVec(object):

    def _validate(self, ob, stop_prepared=False):
        if ob is None:
            raise ValueError("Null geometry supports no operations")
        if stop_prepared and not hasattr(ob, 'type'):
            raise ValueError("Prepared geometries cannot be operated on")

        
class DelegatingVec(ValidatingVec):

    def __init__(self, name):
        self.fn = lgeos.methods[name]

    def _check_topology(self, err, *geoms):
        """Raise TopologicalError if geoms are invalid.

        Else, raise original error.
        """
        for geom in geoms:
            if not geom.is_valid:
                raise TopologicalError(
                    "The operation '%s' could not be performed. "
                    "Likely cause is invalidity of the geometry %s" % (
                        self.fn.__name__, repr(geom)))
        raise err

    
class UnaryPredicateVec(DelegatingVec):

    def __call__(self, this):
        cdef int i
        cdef unsigned int n = this.size
        cdef np.ndarray[np.uint8_t, ndim=1, cast=True] result = np.empty(n, dtype=np.uint8)
        # self._validate(this)
        for i in range(n):
            result[i] = self.fn(this[i]._geom)
        return result.view(dtype=np.bool)

    
class BinaryPredicateVec(DelegatingVec):

    def __call__(self, this, others):
        cdef int i
        cdef unsigned int n = others.size
        cdef np.ndarray[np.uint8_t, ndim=1, cast=True] result = np.empty(n, dtype=np.uint8)
        
        self._validate(this)
        # self._validate(others, stop_prepared=True)
        p1 = this._geom
        for i in range(n):
            result[i] = self.fn(p1, others[i]._geom)
        return result.view(dtype=np.bool)

    
class UnaryRealPropertyVec(DelegatingVec):

    def __call__(self, this):
        cdef int i
        cdef unsigned int n = this.size
        result = np.empty(n)
        # self._validate(this)
        d = c_double()
        for i in range(n):
            retval = self.fn(this[i]._geom, byref(d))
            result[i] = d.value
        return result

    
class BinaryRealPropertyVec(DelegatingVec):

    def __call__(self, this, others):
        cdef int i
        cdef unsigned int n = others.size
        result = np.empty(n)
        # cdef np.ndarray[np.uint8_t, ndim=1, cast=True] result = np.empty(n, dtype=np.uint8)
        
        self._validate(this)
        # self._validate(others, stop_prepared=True)
        p1 = this._geom
        d = c_double()
        for i in range(n):
            retval = self.fn(p1, others[i]._geom, byref(d))
            result[i] = d.value
        return result

    
class UnaryTopologicalOpVec(DelegatingVec):

    def __call__(self, this, *args):
        cdef int i
        cdef unsigned int n = this.size
        result = np.empty(n, dtype = int)
        # self._validate(this)
        for i in range(n):
            result[i] = self.fn(this[i]._geom, *args)
        return result

    
class BinaryTopologicalOpVec(DelegatingVec):

    def __call__(self, this, others, *args):
        cdef int i
        cdef unsigned int n = others.size
        result = np.empty(n, dtype = int)
        self._validate(this)
        # self._validate(others, stop_prepared=True)
        p1 = this._geom
        for i in range(n):
            result[i] = self.fn(p1, others[i]._geom, *args)
        # if product is None:
        #     err = TopologicalError(
        #         "This operation could not be performed. Reason: unknown")
        #     self._check_topology(err, this, others)
        return result



    
# this section reproduces impl.py:

vectorized_dict = {UnaryPredicate: UnaryPredicateVec,
                   BinaryPredicate: BinaryPredicateVec,
                   UnaryRealProperty: UnaryRealPropertyVec,
                   BinaryRealProperty: BinaryRealPropertyVec,
                   UnaryTopologicalOp: UnaryTopologicalOpVec,
                   BinaryTopologicalOp: BinaryTopologicalOpVec}

# return tuples of function names and functions, using the
# vectorized_dict to replace normal function generating classes with
# their vectorized counterparts
def impl_vectorized_items(defs):
    return [(k, vectorized_dict[v[0]](v[1])) for k, v in list(defs.items()) if v[0] in vectorized_dict.keys() ]

imp = GEOSImpl(dict(impl_vectorized_items(IMPL300)))
if lgeos.geos_version >= (3, 1, 0):
    imp.update(impl_vectorized_items(IMPL310))
if lgeos.geos_version >= (3, 1, 1):
    imp.update(impl_vectorized_items(IMPL311))
if lgeos.geos_version >= (3, 2, 0):
    imp.update(impl_vectorized_items(IMPL320))
if lgeos.geos_version >= (3, 3, 0):
    imp.update(impl_vectorized_items(IMPL330))

VectorizedImplementation = imp
