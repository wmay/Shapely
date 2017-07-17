import pytest

from shapely.geometry import Point, Polygon
from shapely.impl import DefaultImplementation

try:
    from shapely.vectorized.vectorfuncs import *
    import numpy as np
    has_vectorized = True
except ImportError:
    has_vectorized = False



points = np.array([Point(i, i) for i in range(500)], dtype=object)
poly = Polygon([(10,10), (10,100), (100,100), (100, 10)])
polys = np.array([ Polygon([(10,10), (10,100), (i,i), (100, 10)]) for i in range(20, 520) ])

def generate_test(fname, f):
    fclass = type(f)
    if issubclass(fclass, UnaryVec) or isinstance(f, UnaryRealPropertyVec):
        def get_results(fname, f):
            results1 = f(polys)
            results2 = np.array([ DefaultImplementation[fname](polys[i]) for i in range(500) ])
            return results1, results2
    else:
        def get_results(fname, f):
            results1 = f(polys, points)
            results2 = np.array([ DefaultImplementation[fname](polys[i], points[i]) for i in range(500) ])
            return results1, results2
        
    def test_f():
        r1, r2 = get_results(fname, f)
        # convert pointers to shapely geometries if needed
        if isinstance(fclass, UnaryTopologicalOpVec) or isinstance(fclass, BinaryTopologicalOpVec):
            r2 = make_geoms(r2)
        assert((r1 == r2).all())
        
    globals()['test_vectorized_' + fname] = test_f

for fname, f in VectorizedImplementation.map.items():
    generate_test(fname, f)

# def test_all_vectorized_functions():
#     for fname, f in VectorizedImplementation.map.items():
#         fclass = type(f)
#         if issubclass(fclass, UnaryVec) or isinstance(fclass, UnaryRealPropertyVec):
#             results1 = f(polys)
#             results2 = np.array([ DefaultImplementation[fname](polys[i]) for i in range(500) ])
#         else:
#             results1 = f(polys, points)
#             results2 = np.array([ DefaultImplementation[fname](polys[i], points[i]) for i in range(500) ])
#         assert((results1 == results2).all())
    


# def test_error():
#     with pytest.raises(ImplementationError):
#         Point(0, 0).impl['bogus']()
#     with pytest.raises(NotImplementedError):
#         Point(0, 0).impl['bogus']()
#     with pytest.raises(KeyError):
#         Point(0, 0).impl['bogus']()


# def test_delegated():
#     class Poynt(Point):
#         @delegated
#         def bogus(self):
#             return self.impl['bogus']()
#     with pytest.raises(ImplementationError):
#         Poynt(0, 0).bogus()
#     with pytest.raises(AttributeError):
#         Poynt(0, 0).bogus()
