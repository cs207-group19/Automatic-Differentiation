How to Use ``DeriveAlive``
==========================

How to install
--------------

| The url to the project is https://pypi.org/project/DeriveAlive/.

   -  Create a virtual environment and activate it

      ::

              # If you don't have virtualenv, install it
              sudo easy_install virtualenv
              # Create virtual environment
              virtualenv env
              # Activate your virtual environment
              source env/bin/activate

   -  Install ``DeriveAlive`` using pip. In the terminal, type:

      ::

              pip install DeriveAlive

   -  Run module tests before beginning.

      ::

              # Navigate to https://pypi.org/project/DeriveAlive/#files
              # Download tar.gz folder, unzip, and enter the folder
              pytest tests


Basic demo
----------

::

      python
      >>> from DeriveAlive import DeriveAlive as da
      >>> from DeriveAlive import rootfinding as rf
      >>> from DeriveAlive import optimize as opt
      >>> from DeriveAlive import spline as sp
      >>> import numpy as np
      >>> import matplotlib.pyplot as plt

Declare Variables
~~~~~~~~~~~~~~~~~

- Denote constants

::

      # None has to be typed, otherwise will be denoted as an R^1 variable
      >>> a = da.Var([1], None)
      >>> a
      Var([1], None)

.. note:: Constant (scalar or vector): User must initialize derivative to 'None'. Otherwise, the variable will be denoted as an  :math:`\mathbb{R}^1` variable with derivative [1].

- Denote scalar variables and functions

::

      # The first way to denote a scalar varibale
      >>> x = da.Var([1])
      >>> x
      Var([1], [1])

      # The second way to denote a scalar variable
      >>> x = da.Var([1], [1])
      >>> x
      Var([1], [1])

      # Denote a scalar function
      >>> f = 2 * x + np.sin(x)
      >>> f
      Var([2.84147098], [2.54030231])

      # Define a callable scalar function:
      >>> def f(x):
              return 2 * x + np.sin(x)
      <function f at 0x116080950>

- Denote vector variables and functions

::

      # Suppose we want to denote variables in R^3
      >>> x = da.Var([1], [1, 0, 0])
      >>> y = da.Var([2], [0, 1, 0])
      >>> z = da.Var([3], [0, 0, 1])

      # Alternatively, users can use the following notation to declare the same variables
      # 'x,y': x denotes the length of the derivative, y denotes the position of the 1
      >>> x = da.Var([1], '3,0')
      >>> y = da.Var([2], '3,1')
      >>> z = da.Var([3], '3,2')

      # Suppose we want to denote an R^3 to R^1 function
      f = x + y + z
      >>> f
      Var([6], [1 1 1])

      # Alternatively, the user can define the R^3 to R^1 function 
      # by explicitly defining a da.Var vector with one entry:

      >>> g = da.Var([2 * x + x * y])
      >>> g
      Var([4], [4 1 0])

      # Suppose we want to denote an R^3 to R^3 function
      >>> f = da.Var([x, y ** 2, z ** 4])
      >>> f
          Values:
          [ 1  4 81],
          Jacobian:
          [[  1   0   0]
           [  0   4   0]
           [  0   0 108]]

      # Suppose we want to denote an R^1 to R^3 function
      >>> x = da.Var([1])
      >>> f = da.Var([x, np.sin(x), np.exp(x-1)])
      >>> f
          Values:
          [1.         0.84147098 1.        ],
          Jacobian:
          [[1.        ]
           [0.54030231]
           [1.        ]]




Demo 1: :math:`\mathbb{R}^1 \rightarrow \mathbb{R}^1`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Consider the case :math:`f(x) = \sin(x) + 5 \tan(x/2)`. We want to calculate the value and the first derivative of :math:`f(x)` at :math:`x=\frac{\pi}{2}`.

::

      # Expect value of 6.0, derivative of 5.0
      >>> x = da.Var([np.pi/2])
      >>> f = np.sin(x) + 5 * np.tan(x/2)
      >>> print(f.val)
      [6.]
      >>> print(f.der)
      [5.]

Demo 2: :math:`\mathbb{R}^m \rightarrow \mathbb{R}^1`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Consider the case :math:`f(x,y) = \sin(x) + \exp(y)`. We want to calculate the value and the jacobian of :math:`f(x,y)` at :math:`x=\frac{\pi}{2}, y=1`.

::

      # Expect value of 3.71828183, jacobian of [0, 2.71828183]
      >>> x = da.Var([np.pi/2], [1, 0])
      >>> y = da.Var([1], [0, 1])
      >>> f = np.sin(x) + np.exp(y)
      >>> print(f.val)
      [3.71828183]
      >>> print(f.der)
      [0.         2.71828183]

Demo 3: :math:`\mathbb{R}^1 \rightarrow \mathbb{R}^n`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Consider the case :math:`f(x) = (\sin(x), x^2)`. We want to calculate the value and the Jacobian of :math:`f(x)` at :math:`x=\frac{\pi}{2}`.

::

      # Expect value of [1. 2.4674011], jacobian of [[0], [3.14159265]]
      >>> x = da.Var([np.pi/2], [1])
      >>> f = da.Var([np.sin(x), x ** 2])
      >>> f
          Values:
          [1.        2.4674011],
          Jacobian:
          [[0.        ]
           [3.14159265]]

Demo 4: :math:`\mathbb{R}^m \rightarrow \mathbb{R}^n`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Consider the case :math:`f(x,y,z) = (\sin(x), 4y + z^3)`. We want to calculate the value and the jacobian of :math:`f(x,y,z)` at :math:`x=\frac{\pi}{2}, y=3, z=-2`.

::
      
      # Expect value of [1, 4], jacobian of [[0 0 0], [0 4 12]]
      >>> x = da.Var([np.pi/2], [1, 0, 0])
      >>> y = da.Var([3], [0, 1, 0])
      >>> z = da.Var([-2], [0, 0, 1])
      >>> f = da.Var([np.sin(x), 4 * y + z ** 3])
      >>> f
      Values:
      [1. 4.],
      Jacobian:
      [[ 0.  0.  0.]
       [ 0.  4. 12.]]

..Note:: Demos for additional features are listed in the corresponding additional features tab.
