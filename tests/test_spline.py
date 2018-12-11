# Integration tests for root_finding suite
# Harvard University, CS 207, Fall 2018
# Authors: Chen Shi, Stephen Slater, Yue Sun

import sys
sys.path.append('../')
import DeriveAlive.DeriveAlive as da
import DeriveAlive.spline as sp
import numpy as np

# Comment out for testing on Travis CI
import matplotlib.pyplot as plt

def test_spline():
    
    def f1(var):
        return 10**var

    xMin1 = -1
    xMax1 = 1
    nIntervals1 = 10
    nSplinePoints1 = 5

    y1, A1, coeffs1, ks1 = sp.quad_spline_coeff(f1, xMin1, xMax1, nIntervals1)
    fig1 = sp.quad_spline_plot(f1, coeffs1, ks1, nSplinePoints1)
    spline_points1 = sp.spline_points(f1, coeffs1, ks1, nSplinePoints1)
    error = sp.spline_error(f1, spline_points1)

print ("Testing spline suite.")
test_spline()
print ("All tests passed!")

