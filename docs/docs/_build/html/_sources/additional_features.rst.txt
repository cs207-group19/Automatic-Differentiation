=================================
CS 207 Final Project: Milestone 2
=================================

:Author: Group 19: Chen Shi, Stephen Slater, Yue Sun
:Date:   November 2018

.. role:: math(raw)
   :format: html latex
..

Additional features
===================

Root finding
------------

Optimization
------------

Quadratic Splines
-----------------

Background
~~~~~~~~~~

| The ``DeriveAlive`` package can be used to calculate quadratic splines
  since it automatically returns the first derivative of a function at a
  given point.
| We aim to construct a piecewise quadratic spline :math:`s(x)` using
  :math:`N` equally-sized intervals over an interval for :math:`f(x)`.
  Define :math:`h=1/N`, and let :math:`s_{k}(x)` be the spline over the
  range :math:`[kh,(k+1)h]` for :math:`k=0,1,\ldots,N-1`. Each
  :math:`s_k(x)=a_kx^2+b_kx+c_k` is a quadratic, and hence the spline
  has :math:`3N` degrees of freedom in total.
| The spline coefficients satisfy the following constraints:

-  Each :math:`s_k(x)` should match the function values at both of its
   endpoints, so that :math:`s_k(kh)=f(kh)` and
   :math:`s_k( (k+1)h) =f( (k+1)h)`. (Provides :math:`2N` constraints.)

-  At each interior boundary, the spline should be differentiable, so
   that :math:`s_{k-1}(kh)= s_k(kh)` for :math:`k=1,\ldots,N-1`.
   (Provides :math:`N-1` constraints.)

-  Since :math:`f'(x+1)=10f'(x)`, let :math:`s'_{N-1}(1) = 10s'_0(0)`.
   (Provides :math:`1` constraint.)

Since there are :math:`3N` constraints for :math:`3N` degrees of
freedom, there is a unique solution.

Implementation
~~~~~~~~~~~~~~

-  Methods

   -  ``quad_spline_coeff``: calculate the coefficients of quadratic
      splines

      -  input:

         -  ``f``: function of interest

         -  ``xMin``: left endpoint of the :math:`x` interval

         -  ``xMax``: right endpoint of the :math:`x` interval

         -  ``nIntervals``: number of intervals that you want to slice
            the original function

      -  return:

         -  ``y``: the right hand side of :math:`Ax=y`

         -  ``A``: the sqaure matrix in the left hand side of
            :math:`Ax=y`

         -  ``coeffs``: coefficients of :math:`a_i, b_i, c_i`

         -  ``ks``: points of interest in the :math:`x` interval as
            ``DeriveAlive`` objects

   -  ``spline_points``: get the coordinates of points on the
      corresponding splines

      -  input:

         -  ``f``: function of interest

         -  ``coeffs``: coefficients of :math:`a_i, b_i, c_i`

         -  ``ks``: points of interest in the :math:`x` interval as
            ``DeriveAlive`` objects

         -  ``nSplinePoints``: number of points to draw each spline

      -  return:

         -  ``spline_points``: a list of spline points :math:`(x,y)` on
            each :math:`s_i`

   -  ``quad_spline_plot``: plot the original function and the
      corresponding splines

      -  input:

         -  ``f``: function of interest

         -  ``coeffs``: coefficients of :math:`a_i, b_i, c_i`

         -  ``ks``: points of interest in the :math:`x` interval as
            ``DeriveAlive`` objects

         -  ``nSplinePoints``: number of points to draw each spline

      -  return:

         -  ``fig``: the plot of :math:`f(x)` and splines

   -  ``spline_error``: calculate the average absolute error of the
      spline and the original function at one point

      -  input:

         -  ``f``: function of interest

         -  ``spline_points``: a list of spline points :math:`(x,y)` on
            each :math:`s_i`

      -  return:

         -  ``error``: average absolute error of the spline and the
            original function on one given interval

-  External dependencies

   -  ``DeriveAlive``

   -  ``NumPy``

   -  ``matplotlib.pyplot``

Demo
~~~~

Plot the quadratic spline of :math:`f_1(x) = 10^x, x \in [-1, 1]` with
10 intervals.

::

        import spline as sp
        import numpy as np
        import matplotlib.pyplot as plt
      
        xMin1 = -1
        xMax1 = 1
        nIntervals1 = 10
        nSplinePoints1 = 5

        y1, A1, coeffs1, ks1 = sp.quad_spline_coeff(f1, xMin1, xMax1, nIntervals1)
        fig1 = sp.quad_spline_plot(f1, coeffs1, ks1, nSplinePoints1)
        spline_points1 = sp.spline_points(f1, coeffs1, ks1, nSplinePoints1)
        sp.spline_error(f1, spline_points1)

::

        0.0038642295476342416

Drawing with Splines
~~~~~~~~~~~~~~~~~~~~
