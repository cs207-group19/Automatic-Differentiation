import sys
sys.path.append('../..')

import DeriveAlive.DeriveAlive as da # This path is subject to change
import numpy as np

# Comment out for testing on Travis CI
import matplotlib.pyplot as plt

# Calculate the coefficients of the quadratic functions
def quad_spline_coeff(f, xMin, xMax, nIntervals):
    # f: function
    # x in [xMin, xMax], N is the number of intervals
    # We will solve for x in y = Ax using np.linalg.solve
    
    h = 1/nIntervals
    ks = []
    for i in np.linspace(xMin, xMax, nIntervals+1):
        k = da.Var(i)
        ks.append(k)
    
    # Construct the quadratic functions
    def a(var):
        return var ** 2
    def b(var):
        return var
    def c(var):
        return da.Var(1)
    
    # Construct y
    y = []
    for i in range(nIntervals):
        y.append(f(ks[i]).val)
        y.append(f(ks[i+1]).val)
    for i in range(nIntervals):
        y.append([0])
    y = np.vstack(y)
    
    # Construct A
    A = np.zeros((3*nIntervals, 3*nIntervals))
    # Constraint 1:
    for i in range(nIntervals):
        A[2*i, 3*i] = a(ks[i]).val
        A[2*i, 3*i+1] = b(ks[i]).val
        A[2*i, 3*i+2] = c(ks[i]).val
        A[2*i+1, 3*i] = a(ks[i+1]).val
        A[2*i+1, 3*i+1] = b(ks[i+1]).val
        A[2*i+1, 3*i+2] = c(ks[i+1]).val
    # Constraint 2:
    for i in range(nIntervals-1):
        A[2*nIntervals+i, 3*i] = a(ks[i+1]).der
        A[2*nIntervals+i, 3*i+1] = b(ks[i+1]).der
        A[2*nIntervals+i, 3*i+3] = -1*a(ks[i+1]).der
        A[2*nIntervals+i, 3*i+4] = -1*b(ks[i+1]).der
    # Constraint 3:
    A[3*nIntervals-1, 1] = 10*b(ks[0]).der
    A[3*nIntervals-1, -3] = -1*a(ks[-1]).der
    A[3*nIntervals-1, -2] = -1*b(ks[-1]).der
    
    coeffs = np.linalg.solve(A, y)
    
    return y, A, coeffs, ks

# Plot the spline and the orignal function
def quad_spline_plot(f, coeffs, ks, nSplinePoints):
    
    fig = plt.figure()
    ax = plt.subplot(111)
    
    # Plot the original function
    fx = []
    fy = []
    for k in ks:
        fx.append(k.val)
        fy.append(f(k).val)
    ax.plot(fx, fy, 'o-', linewidth=2, label='original')
    
    spline_points = []
    # Plot the splines
    for i in range(len(ks)-1):
        a = coeffs[3*i]
        b = coeffs[3*i+1]
        c = coeffs[3*i+2]
        sx = np.linspace(ks[i].val, ks[i+1].val, nSplinePoints)
        sy = a*(sx**2) + b*sx + c
        spline_points.append([sx, sy])
        ax.plot(sx, sy, label=r'$s_{%s}(x)$' % i)
    
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$f(x)$')
    box = ax.get_position()
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    return fig, spline_points

# Calculate spline squared error
def spline_squared_error(f, spline_points):
    
    squared_error = 0
    for spline_point in spline_points:
        xs = spline_point[0]
        original_ys = []
        for x in xs:
            original_y = f(da.Var(x)).val
            original_ys.append(original_y)
        
        squared_error += sum((np.hstack(original_ys) - spline_point[1])**2)
    
    return squared_error
