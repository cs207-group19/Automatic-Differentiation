import DeriveAlive.DeriveAlive as da # This path is subject to change
import numpy as np

# Comment out for testing on Travis CI
import matplotlib.pyplot as plt

# Calculate the coefficients of the quadratic functions
def quad_spline_coeff(f, xMin, xMax, nIntervals):
    """ Constructs the matrix for quadratic spline calculation
        and returns the coefficients of quadratic functions.
        
        Parameters
        ----------
        f: ``DeriveAlive.Var`` objects
           function specified by user.
        
        xMin: float
              left endpoint of the :math:`x` interval
        
        xMax: float
              right endpoint of the :math:`x` interval
        
        nIntervals: integer
                    number of intervals that you want to slice the original function
            
        Returns
        -------
        y: list of floats
           the right hand side of Ax=y
        
        A: numpy.ndarray
            the sqaure matrix in the left hand side of Ax=y
        
        coeffs: list of floats
                coefficients of a_i, b_i, c_i
        
        ks: list of ``DeriveAlive.Var`` objects
            points of interest in the :math:`x` interval as ``DeriveAlive`` objects
        
        """
    
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

# Get the positions of the spline points
def spline_points(f, coeffs, ks, nSplinePoints):
    """ Returns the coefficients of quadratic functions.
        
        Parameters
        ----------
        f: ``DeriveAlive.Var`` objects
            function specified by user.
        
        coeffs: list of floats
                coefficients of a_i, b_i, c_i
        
        ks: list of ``DeriveAlive.Var`` objects
            points of interest in the :math:`x` interval as ``DeriveAlive`` objects
        
        nSplinePoints: integer
                       number of points to draw each spline
        
        Returns
        -------
        spline_points: list of numpy.darrays
                       a list of spline points (x,y) on each s_i
        
        """
    
    spline_points = []

    for i in range(len(ks)-1):
        a = coeffs[3*i]
        b = coeffs[3*i+1]
        c = coeffs[3*i+2]
        sx = np.linspace(ks[i].val, ks[i+1].val, nSplinePoints)
        sy = a*(sx**2) + b*sx + c
        spline_points.append([sx, sy])

    return spline_points

# Plot the spline and the orignal function
def quad_spline_plot(f, coeffs, ks, nSplinePoints):
    """ Returns the coefficients of quadratic functions.
        
        Parameters
        ----------
        f: ``DeriveAlive.Var`` objects
           function specified by user.
        
        coeffs: list of floats
                coefficients of a_i, b_i, c_i
        
        ks: list of ``DeriveAlive.Var`` objects
            points of interest in the :math:`x` interval as ``DeriveAlive`` objects
        
        nSplinePoints: integer
                       number of points to draw each spline
        
        Returns
        -------
        fig: matplotlib.figure
             the plot of :math:`f(x)` and splines
        
        """
    
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

    return fig

# Calculate spline squared error
def spline_error(f, spline_points):
    """ Returns the coefficients of quadratic functions.
        
        Parameters
        ----------
        f: ``DeriveAlive.Var`` objects
           function specified by user.
        
        spline_points: list of numpy.darrays
                       a list of spline points (x,y) on each s_i
        
        Returns
        -------
        error: float
               average absolute error of the spline and the original function on one given interval
        
        """
    
    error = 0
    for spline_point in spline_points:
        xs = spline_point[0]
        original_ys = []
        for x in xs:
            original_y = f(da.Var(x)).val
            original_ys.append(original_y)
        
        error += abs(sum((np.hstack(original_ys) - spline_point[1])) / len(spline_point[0])) ** 1
    
    return error / len(spline_points)
