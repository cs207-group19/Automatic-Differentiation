
Additional Features
===================

Root finding
------------

Background
~~~~~~~~~~

Newton root finding starts from an initial guess for :math:`x_1` and converges to :math:`x` such that :math:`f(x) = 0`. The algorithm is iterative. At each step :math:`t`, the algorithm finds a line (or plane, in higher dimensions) that is tangent to :math:`f` at :math:`x_t`. The new guess for :math:`x_{t+1}` is where the tangent line crosses the :math:`x`-axis. This generalizes to :math:`m` dimensions.

- Algorithm (univariate case)

for :math:`t` iterations or until step size < ``tol``:
    :math:`x_{t+1} \Leftarrow x_{t} - \frac{f(x_t)}{f'(x_t)}`

- Algorithm (multivariate case)

for :math:`t` iterations or until step size < ``tol``:
    :math:`\textbf{x}_{t+1} \Leftarrow \textbf{x}_t - (J(f)(\textbf{x}_t))^{-1}f(\textbf{x}_t)`

In the multivariate case, :math:`J(f)` is the Jacobian of :math:`f`. If :math:`J(f)` is non-square, we use the pseudoinverse.

Here is an example in the univariate case:



A common application of root finding is in Lagrangian optimization. For example, consider the Lagrangian :math:`\mathcal{L}(\textbf{b}, \lambda)`. One can solve for the weights :math:`\textbf{b}, \lambda` such that :math:`\frac{\partial \mathcal{L}}{\partial b_j} = \frac{\partial \mathcal{L}}{\partial \lambda} = 0`.

Implementation
~~~~~~~~~~~~~~

-  Methods

   -  ``NewtonRoot``: return a root of a function :math:`f: \mathbb{R}^m \Rightarrow \mathbb{R}^1`

      -  input:

         -  ``f``: function of interest, callable. If :math:`f` is a scalar to scalar function, then define :math: `f` as follows:

            .. code-block:: python
               :linenos:

               def f(x):
                   # Use x in function
                   return x ** 2 + np.exp(x)  


            If :math:`f` is a function of multiple scalars (i.e. :math:`\mathbb{R}^m \Rightarrow \mathbb{R}^1`), the arguments to :math:`f` must be passed in
            as a list. In this case, define :math:`f` as follows:

            .. code-block:: python
               :linenos:

               def f(variables):
                   x, y, z = variables
                   return x ** 2 + y ** 2 + z ** 2 + np.sin(x)
         
         -  ``x``: List of da.Var objects. Inital guess for a root of :math:`f`. If :math:`f` is a scalar to scalar function (i.e. :math:`\mathbb{R}^1 \Rightarrow \mathbb{R}^1), and the initial guess for the root is 1, then x = [da.Var(1)]. If :math:`f` is a function of multiple scalars, with initial guess for the root as (1, 2, 3), then define ``x`` as:

            .. code-block:: python
               :linenos:

               x0 = da.Var(1, [1, 0, 0])
               y0 = da.Var(2, [0, 1, 0])
               z0 = da.Var(3, [0, 0, 1])
               x = [x0, y0, z0]               

         -  ``iters``: int, optional, default=2000. The maximum number of iterations to run the Newton root finding algorithm. The algorithm will run for min :math:`(t, iters)` iterations, where :math:`t` is the number of steps until ``tol`` is satisfied.

         -  ``tol``: int or float, optional, default=1e-10. If the size of the update step (L2 norm in the case of :math:`\mathbb{R}^m \Rightarrow \mathbb{R}^1)` is smaller than ``tol``, then the algorithm will add that step and then terminate, even if the number of iterations has not reached ``iters``.

      -  return:

         -  ``root``: da.Var :math:`\in \mathbb{R}^m`. The `val` attribute contains a numpy array of the root that the algorithm found in :math:`min(iters, t)` iterations (:math:`iters, t` defined above). The `der` attribute contains the Jacobian value at the specified root.

         -  ``var_path``: a numpy array (:math:`\mathbb{R}^{n \times m}`), where :math:`n = min(iters, t)` is the number of steps of the algorithm and :math:`m` if the dimension of the root, where rows of the array are steps taken in consecutive order.

         -  ``g_path``: a numpy array (:math:`\mathbb{R}^{n \times 1}`), containing the consecutive steps of the output of :math:`f` at each guess in ``var_path``.

-  External dependencies

   -  ``DeriveAlive``

   -  ``NumPy``

   -  ``matplotlib.pyplot``

Demo
~~~~~

Case 1: :math:`f = sin(x)` with starting point :math:`x_0= \frac{3\pi}{2}`. Note: Newton method is not guaranteed to converge when :math:`f\prime(x_0)= 0`. But in our case, we use flip coin to determine which direction we want to go in order to get stuck at this point, and choose a derivative of :math:`\pm1`.

::

        # define f function
        >>> f_string = 'f(x) = sin(x)'

        >>> def f(x):
                return np.sin(x)

        >>> # Start at 3*pi/2 
        >>> x0 = 3 * np.pi / 2

            # finding the root
        >>> for val in [np.pi - 0.25, np.pi, 1.5 * np.pi, 2 * np.pi - 0.25, 2 * np.pi + 0.25]:
                solution, x_path, y_path = rf.NewtonRoot(f, x0)

            # visualize the trace
        >>> x_lims = -2 * np.pi, 3 * np.pi
        >>> y_lims = -2, 2
        >>> rf.plot_results(f, x_path, y_path, f_string, x_lims, y_lims)

.. image:: images/7_2_3_1.png
   :width: 600

Case 2:  :math:`f = x - \exp(-2\sin(4x)sin(4x)+0.3` with starting point :math:`x_0 = 0`.

::

        # define f function
        f_string = 'f(x) = x - e^{-2 * sin(4x) * sin(4x)} + 0.3'

        >>> def f(x):
                return x - np.exp(-2.0 * np.sin(4.0 * x) * np.sin(4.0 * x)) + 0.3

        # start at 0
        >>> x0 = 0

        # finding the root
        >>> for val in np.arange(-0.75, 0.8, 0.25):
                solution, x_path, y_path = rf.NewtonRoot(f, x0)

        # visualize the trace
        >>> x_lims = -2, 2
        >>> y_lims = -2, 2
        >>> rf.plot_results(f, x_path, y_path, f_string, x_lims, y_lims)

Case 3: :math:`f(x, y) = x^2 + 4y^2-2x^2y +4` with starting points :math:`x_0 =-8.0, y_0 = -5.0`.

::

        # define f function
        >>> f_string = 'f(x, y) = x^2 + 4y^2 -2x^2y + 4'

        >>> def f(variables):
                x, y = variables
                return x ** 2 + 4 * y ** 2 - 2 * (x ** 2) * y + 4

        # start at x0=−8.0,y0= −5
        >>> x0 = -8.0
        >>> y0 = -5.0
        >>> init_vars = [x0, y0]

        # finding the root and visualize the trace
        >>> solution, xy_path, f_path = rf.NewtonRoot(f, init_vars)
        >>> rf.plot_results(f, xy_path, f_path, f_string, threedim=True)

Case 4: :math:`f(x, y, z) = x^2 + y^2 + z^2` with starting points :math:`x_0 =1, y_0 = -2, z_0 = 5`.

::

        # define f function
        >>> f_string = 'f(x, y, z) = x^2 + y^2 + z^2'

        >>> def f(variables):
                x, y, z = variables
                return x ** 2 + y ** 2 + z ** 2 + np.sin(x) + np.sin(y) + np.sin(z)

        # start at 
        >>> x0= 1
        >>> y0= -2
        >>> z0= 5
        >>> init_vars = [x0, y0, z0]

        # finding the root and visualize the trace
        >>> solution, xyz_path, f_path = rf.NewtonRoot(f, init_vars)
        >>> m = len(solution.val)
        >>> rf.plot_results(f, xyz_path, f_path, f_string, fourdim=True) 

Optimization
------------

Background
~~~~~~~~~~

Gradient Descent is used to find the local minimum of a function :math:`f` by taking locally optimum steps in the direction of steepest descent. A common application is in machine learning when a user desires to find optimal weights to minimize a loss function.

Here is a visualization of Gradient Descent on a convex function of 2 variables:

.. image:: images/gradient_descent.png
   :width: 600

BFGS, short for "Broyden–Fletcher–Goldfarb–Shanno algorithm", seeks a stationary point of a function, i.e. where the gradient is zero. In quasi-Newton methods, the Hessian matrix of second derivatives is not computed. Instead, the Hessian matrix is approximated using updates specified by gradient evaluations (or approximate gradient evaluations). 

Here is a pseudocode of the implementation of BFGS.

.. image:: images/bfgs.png
   :width: 600


Implementation
~~~~~~~~~~~~~~

-  Methods

   -  ``GradientDescent``: solve for a local minimum of a function :math:`f: \mathbb{R}^m \Rightarrow \mathbb{R}^1`. If :math:`f` is a convex function, then the local minimum is a global minimum.

      -  input:

         -  ``f``: function of interest, callable. In machine learning applications, this should be the cost function. For example, if solving for optimal weights to minimize a cost function :math:`f`, then :math:`f` can be defined as :math:`\frac{1}{2m}` times the sum of :math:`m` squared residuals.

            If :math:`f` is a scalar to scalar function, then define :math:'f' as follows:

            .. code-block:: python
               :linenos:

               def f(x):
                   # Use x in function
                   return x ** 2 + np.exp(x)   


            If :math:`f` is a function of multiple scalars (i.e. :math:`\mathbb{R}^m \Rightarrow \mathbb{R}^1`), the arguments to :math:`f` must be passed in
            as a list. In this case, define :math:`f` as follows:

            .. code-block:: python
               :linenos:

               def def f(variables):
                   x, y, z = variables
                   return x ** 2 + y ** 2 + z ** 2 + np.sin(x)

         -  ``x``: List of da.Var objects. Inital guess for a root of :math:`f`. If :math:`f` is a scalar to scalar function (i.e. :math:`\mathbb{R}^1 \Rightarrow \mathbb{R}^1)`, and the initial guess for the root is 1, then x = [da.Var(1)]. If :math:`f` is a function of multiple scalars, with initial guess for the root as (1, 2, 3), then define ``x`` as follows:

            .. code-blcok::python
               :linenos:
               >>> x0 = da.Var(1, [1, 0, 0])
               >>> y0 = da.Var(2, [0, 1, 0])
               >>> z0 = da.Var(3, [0, 0, 1])
               >>> x = [x0, y0, z0]

         -  ``iters``: int, optional, default=2000. The maximum number of iterations to run the Newton root finding algorithm. The algorithm will run for min :math:`(t, iters)` iterations, where :math:`t` is the number of steps until ``tol`` is satisfied.

         -  ``tol``: int or float, optional, default=1e-10. If the size of the update step (L2 norm in the case of :math:`\mathbb{R}^m \Rightarrow \mathbb{R}^1)` is smaller than ``tol``, then the algorithm will add that step and then terminate, even if the number of iterations has not reached ``iters``.

      -  return:

         -  ``minimum``: da.Var :math:`\in \mathbb{R}^m`. The `val` attribute contains a numpy array of the minimum that the algorithm found in :math:`min(iters, t)` iterations (:math:`iters, t` defined above). The `der` attribute contains the Jacobian value at the specified root.

         -  ``var_path``: a numpy array (:math:`\mathbb{R}^{n \times m}`), where :math:`n = min(iters, t)` is the number of steps of the algorithm and :math:`m` if the dimension of the minimum, where rows of the array are steps taken in consecutive order.

         -  ``g_path``: a numpy array (:math:`mathbb{R}^{n \times 1}`), containing the consecutive steps of the output of :math:`f` at each guess in ``var_path``.

-  External dependencies

   -  ``DeriveAlive``

   -  ``NumPy``

   -  ``matplotlib.pyplot``

Demo
~~~~

::

        >>> import DeriveAlive.optimize as opt
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

Case 1: Minimize quartic function :math:`f(x) = x^4`. Get stuck in local minimum.

::

        >>> def f(x):
                return x ** 4 + 2 * (x ** 3) - 12 * (x ** 2) - 2 * x + 6

            # Function string to include in plot
        >>> f_string = 'f(x) = x^4 + 2x^3 -12x^2 -2x + 6'

        >>> x0 = 4
        >>> solution, xy_path, f_path = opt.GradientDescent(f, x0, iters=1000, eta=0.002)
        >>> opt.plot_results(f, xy_path, f_path, f_string, x_lims=(-6, 5), y_lims=(-100, 70))

Case 2: Minimize Rosenbrock's function :math:`f(x, y) = 4(y - x^2)^2 + (1 - x)^2`. Global minimum: 0 at :math:`(x,y)=(1, 1)`.

::

        # Rosenbrock function with leading coefficient of 4
        >>> def f(variables):
                x, y = variables
                return 4 * (y - (x ** 2)) ** 2 + (1 - x) ** 2

        # Function string to include in plot
        >>> f_string = 'f(x, y) = 4(y - x^2)^2 + (1 - x)^2'

        >>> x_val, y_val = -6, -6
        >>> init_vars = [x_val, y_val]
        >>> solution, xy_path, f_path = opt.GradientDescent(f, init_vars, iters=25000, eta=0.002)
        >>> opt.plot_results(f, xy_path, f_path, f_string, x_lims=(-7.5, 7.5), threedim=True)



::

        >>> x_val, y_val = -2, 5
        >>> init_vars = [x_val, y_val]
        >>> solution, xy_path, f_path = opt.GradientDescent(f, init_vars, iters=25000, eta=0.002)
        >>> opt.plot_results(f, xy_path, f_path, f_string, x_lims=(-7.5, 7.5), threedim=True)


Case 3: Minimize Easom's function: :math:`f(x, y) = -cos(x)cos(y)exp(-((x - \pi)^2 + (y - \pi)^2))`. Global minimum: -1 at :math:`(x,y)=(\pi, \pi)`.

::

        # Easom's function
        >>> def f(variables):
                x, y = variables
                return -np.cos(x) * np.cos(y) * np.exp(-((x - np.pi) ** 2 + (y - np.pi) ** 2))

        # Function string to include in plot
        >>> f_string = 'f(x, y) = -\cos(x)\cos(y)\exp(-((x-\pi)^2 + (y-\pi)^2))'

        # Initial guess
        >>> x0 = 1.5
        >>> y0 = 1.75
        >>> init_vars = [x0, y0]

        # Visulaize gradient descent
        solution, xy_path, f_path = opt.GradientDescent(f, init_vars, iters=10000, eta=0.3)
        opt.plot_results(f, xy_path, f_path, f_string, threedim=True)

Case 4: Machine Learning application: minimize mean squared error in regression

.. math:: \begin{align}
          \hat{y_i} &= \textbf{w}^\top \textbf{x}_i \\
          MSE(X, y) &= \frac{1}{m} \sum_{i=1}^m (\textbf{w}^\top\textbf{x}_i - y_i)^2

where :math:`\textbf{w}` contains an extra dimension to fit the intercept of the features.
Example dataset
Standardized dataset: 47 homes from Portland, Oregon
Features: area (square feet), number of bedrooms
Output: price (in thousands of dollars)

::

        >>> f = "mse"
        >>> init_vars = [0, 0, 0]

        # Function string to include in plot
        >>> f_string = 'f(w_0, w_1, w_2) = (1/2m)\sum_{i=0}^m (w_0 + w_1x_{i1} + w_2x_{i2} - y_i)^2'

        # Visulaize gradient descent
        >>> solution, w_path, f_path, f = opt.GradientDescent(f, init_vars, iters=2500, data=data)
        >>> print ("Gradient descent optimized weights:\n{}".format(solution.val))
        >>> opt.plot_results(f, w_path, f_path, f_string, x_lims=(-7.5, 7.5), fourdim=True)

Case 5: Find stationary point of :math:`f(x) = sin(x)`. Note: BFGS finds stationary point, which can be maximum, not minimum.

::

        >>> def f(x):
                return np.sin(x)

        >>> f_string = 'f(x) = sin(x)'

        >>> x0 = -1
        >>> solution, x_path, f_path = opt.BFGS(f, x0)
        >>> anim = opt.plot_results(f, x_path, f_path, f_string, x_lims=(-2 * np.pi, 2 * np.pi), y_lims=(-1.5, 1.5), bfgs=True)

Case 6: Find stationary point of Rosenbrock function: :math:`f(x, y) = 4(y - x^2)^2 + (1 - x)^2`. Stationary point: 0 at :math:`(x,y)=(1, 1)`.

::

        >>> def f(variables):
                x, y = variables
                return 4 * (y - (x ** 2)) ** 2 + (1 - x) ** 2

        >>> f_string = 'f(x, y) = 4(y - x^2)^2 + (1 - x)^2'

        >>> x0, y0 = -6, -6
        >>> init_vars = [x0, y0]
        >>> solution, xy_path, f_path = opt.BFGS(f, init_vars, iters=25000)
        >>> xn, yn = solution.val
        >>> anim = opt.plot_results(f, xy_path, f_path, f_string, x_lims=(-7.5, 7.5), y_lims=(-7.5, 7.5), threedim=True, bfgs=True)

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
  
| Example: :math:`f(x) = 10^x, x \in [0,1]`, with :math:`N=10` intervals, 
  the spline coefficients satisfy the following constraints:

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

::

        >>> import DeriveAlive.spline as sp
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

Case 1: Plot the quadratic spline of :math:`f_1(x) = 10^x, x \in [-1, 1]` with
10 intervals.

::

        >>> def f1(var):
                return 10**var

        >>> xMin1 = -1
        >>> xMax1 = 1
        >>> nIntervals1 = 10
        >>> nSplinePoints1 = 5

        >>> y1, A1, coeffs1, ks1 = sp.quad_spline_coeff(f1, xMin1, xMax1, nIntervals1)
        >>> fig1 = sp.quad_spline_plot(f1, coeffs1, ks1, nSplinePoints1)
        >>> spline_points1 = sp.spline_points(f1, coeffs1, ks1, nSplinePoints1)
        >>> sp.spline_error(f1, spline_points1)
        0.0038642295476342416

        >>> fig1

.. image:: images/7_3_3_1.png
  :width: 600       

Case 2: Plot the quadratic spline of :math:`f_2(x) = x^3, x \in [-1, 1]` with 10 intervals.

::

        >>> def f2(var):
                return var**3

        >>> xMin2 = -1
        >>> xMax2 = 1
        >>> nIntervals2 = 10
        >>> nSplinePoints2 = 5

        >>> y2, A2, coeffs2, ks2 = sp.quad_spline_coeff(f2, xMin2, xMax2, nIntervals2)
        >>> fig2 = sp.quad_spline_plot(f2, coeffs2, ks2, nSplinePoints2)
        >>> spline_points2 = sp.spline_points(f2, coeffs2, ks2, nSplinePoints2)
        >>> sp.spline_error(f2, spline_points2)
        0.0074670329670330216

        >>> fig2

.. image:: images/7_3_3_2.png
  :width: 600       

Case 3: Plot the quadratic spline of :math:`f_3(x) = \sin(x), x \in [-1,1]` and :math:`x \in [-\pi, \pi]` with 5 intervals and 10 intervals.

::

        >>> def f3(var):
                return np.sin(var)

        >>> xMin3 = -1
        >>> xMax3 = 1
        >>> nIntervals3 = 5
        >>> nSplinePoints3 = 5

        >>> y3, A3, coeffs3, ks3 = sp.quad_spline_coeff(f3, xMin3, xMax3, nIntervals3)
        >>> fig3 = sp.quad_spline_plot(f3, coeffs3, ks3, nSplinePoints3)
        >>> spline_points3 = sp.spline_points(f3, coeffs3, ks3, nSplinePoints3)
        >>> sp.spline_error(f3, spline_points3)
        0.015578205778177232

        >>> fig3

.. image:: images/7_3_3_3.png
  :width: 600       

::

        >>> xMin4 = -1
        >>> xMax4 = 1
        >>> nIntervals4 = 10
        >>> nSplinePoints4 = 5

        >>> y4, A4, coeffs4, ks4 = sp.quad_spline_coeff(f3, xMin4, xMax4, nIntervals4)
        >>> fig4 = sp.quad_spline_plot(f3, coeffs4, ks4, nSplinePoints4)
        >>> spline_points4 = sp.spline_points(f3, coeffs4, ks4, nSplinePoints4)
        >>> sp.spline_error(f3, spline_points4)
        0.0034954287455489196

        >>> fig4

.. image:: images/7_3_3_4.png
  :width: 600       

.. note:: We can see that the quadratic splines do not work that well with linear-ish functions. While adding more intervals may help to make the approximated splines better.

Casee 4: Here we demonstrate that the more intervals will make the splines approximations better using a :math:`log-log` plot of the absolute average error with respect to :math: \frac{1}{N}` with :math:`f(x) = 10^x, x \in [-\pi, \pi]` at intervals from 5 to 100.

::

        >>> def f(var):
                return 10 ** var

        >>> xMin = -sp.np.pi
        >>> xMax = sp.np.pi
        >>> nIntervalsList = sp.np.arange(1, 50, 1)
        >>> nSplinePoints = 10
        >>> squaredErrorList = []

        >>> for nIntervals in nIntervalsList:
                y, A, coeffs, ks = sp.quad_spline_coeff(f, xMin, xMax, nIntervals)
                spline_points = sp.spline_points(f, coeffs, ks, nSplinePoints)
                error = sp.spline_error(f, spline_points)
                squaredErrorList.append(error)
    
        >>> plt.figure()
    
        >>> coefficients = np.polyfit(np.log10(2*np.pi/nIntervalsList), np.log10(squaredErrorList), 1)
        >>> polynomial = np.poly1d(coefficients)
        >>> ys = polynomial(np.log10(2*np.pi/nIntervalsList))
        >>> plt.plot(np.log10(2*np.pi/nIntervalsList), ys, label='linear fit')
        >>> plt.plot(np.log10(2*np.pi/nIntervalsList), np.log10(squaredErrorList), label='actual error plot')
        >>> plt.xlabel(r'$\log(1/N)$')
        >>> plt.ylabel(r'$\log(average error)$')
        >>> plt.legend()
        >>> plt.title('loglog plot of 1/N vs. average error')
        >>> plt.show()

.. image:: images/7_3_3_5.png
  :width: 600       

::

        >>> beta, alpha = coefficients[0], 10**coefficients[1]
        >>> beta, alpha
        (2.2462166565957835, 11.414027075895813)

.. note:: We can see in the :math:`log-log` plot that the log of absolute average error is proportional to the log of :math:`\frac{1}{N}`, i.e. :math:`E_{1/N} \approx 11.4(\dfrac{1}{N})^{2.25}`. 

Drawing with Splines
~~~~~~~~~~~~~~~~~~~~

| This graph is shipped within ``DeriveAlive`` package as a surprise.

| We want to draw a graph based on the follow 20 functions.

- :math:`f_1(x) = \frac{-1}{0.5^2} x^2 + 1, x \in [-0.5, 0]`

- :math:`f_2(x) = \frac{1}{0.5^2} x^2 - 1, x \in [-0.5, 0]`

- :math:`f_3(x) = \frac{-1}{0.5} x^2 + 1, x \in [0, 0.5]`

- :math:`f_4(x) = \frac{1}{0.5} x^2 - 1, x \in [0, 0.5]`

- :math:`f_6(x) = \frac{-1}{0.5} (x-1.5)^2 + 1, x \in [1, 1.5]`

- :math:`f_7(x) = \frac{1}{0.5} (x-1.5)^2 - 1, x \in [1, 1.5]`

- :math:`f_8(x) = \frac{-1}{0.5} (x-1.5)^2, x \in [1.5, 2]`

- :math:`f_9(x) = \frac{-1}{0.5} (x-1.5)^2 + 1, x \in [1.5, 2]`

- :math:`f_{10}(x) = \frac{1}{0.5} (x-1.5)^2 - 1, x \in [1.5, 2]`

- :math:`f_{11}(x) = \frac{-1}{0.5} (x-3)^2 + 1, x \in [2.5, 3]`

- :math:`f_{12}(x) = \frac{-1}{0.5} (x-3)^2 + 1, x \in [3, 3.5]`

- :math:`f_{13}(x) = 1.5x - 4.75, x \in [2.5, 3.5]`

- :math:`f_{14}(x) = -1, x \in [2.5, 3.5]`

- :math:`f_{15}(x) = \frac{-1}{0.5^2} (x-4.5)^2 + 1, x \in [4, 4.5]`

- :math:`f_{16}(x) = \frac{1}{0.5^2} (x-4.5)^2 - 1, x \in [4, 4.5]`

- :math:`f_{17}(x) = \frac{-1}{0.5^2} (x-4.5)^2 + 1, x \in [4, 4.5]`

- :math:`f_{18}(x) = \frac{1}{0.5^2} (x-4.5)^2 - 1, x \in [4.5, 5]`

- :math:`f_{19}(x) = 1, x \in [5.5, 6.5]`

- :math:`f_{20}(x) = \frac{-1}{(-0.75)^2} (x-6.5)^2 + 1, x \in [5.75, 6.5]`

::

  >>> import surprise
  # We first draw out the start and end points of each function
  >>> surprise.drawPoints()

.. image:: images/7_3_3_6.png
  :width: 600       

::

  # Then we use the spline suite to draw quadratic splines based on the two points
  >>> surprise.drawSpline()

.. image:: images/7_3_3_7.png
  :width: 600       

::

  >>> surprise.drawTogether()

.. image:: images/7_3_3_8.png
  :width: 600       
