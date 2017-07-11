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
from shapely.geometry.base import geom_factory
from shapely.impl import *


get_pointers = np.frompyfunc(lambda x: x._geom, 1, 1)
make_geoms = np.frompyfunc(geom_factory, 1, 1)


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


class UnaryVec(ValidatingVec):
    
    def __init__(self, name):
        vec_func = np.frompyfunc(lgeos.methods[name], 1, 1)
        def un_func(this):
            p1 = get_pointers(this)
            return vec_func(p1)
        self.fn = un_func

    
class BinaryVec(ValidatingVec):

    def __init__(self, name):
        vec_func = np.frompyfunc(lgeos.methods[name], 2, 1)
        def bin_func(this, others):
            p1 = get_pointers(this)
            p2 = get_pointers(others)
            return vec_func(p1, p2)
        self.fn = bin_func

    
class UnaryPredicateVec(UnaryVec):

    def __call__(self, this):
        return self.fn(this).astype(bool)

    
class BinaryPredicateVec(BinaryVec):

    def __call__(self, this, others):
        return self.fn(this, others).astype(bool)

    
class UnaryRealPropertyVec(DelegatingVec):

    def __init__(self, name):
        d = c_double()
        dptr = byref(d)
        def real_func(this):
            lgeos.methods[name](this, dptr)
            return d.value
        vec_func = np.frompyfunc(real_func, 1, 1)
        def un_func(this):
            p1 = get_pointers(this)
            return vec_func(p1)
        self.fn = un_func

    def __call__(self, this):
        return self.fn(this).astype(float)

    
class BinaryRealPropertyVec(DelegatingVec):
    
    def __init__(self, name):
        d = c_double()
        dptr = byref(d)
        def real_func(this, others):
            lgeos.methods[name](this, others, dptr)
            return d.value
        vec_func = np.frompyfunc(real_func, 2, 1)
        def bin_func(this, others):
            p1 = get_pointers(this)
            p2 = get_pointers(others)
            return vec_func(p1, p2)
        self.fn = bin_func

    def __call__(self, this, others):
        return self.fn(this, others).astype(float)

    
class UnaryTopologicalOpVec(UnaryVec):

    def __call__(self, this, *args):
        return self.fn(this)

class BinaryTopologicalOpVec(BinaryVec):

    def __call__(self, this, others, *args):
        return self.fn(this, others)



    
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
