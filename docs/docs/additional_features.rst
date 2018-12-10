
.. role:: math(raw)
   :format: html latex
..


Additional features
===================

This package will have two possible additional features (at least one of
which we will implement):

-  Write an application that uses *DeriveAlive* to implement
   optimization methods, like different forms of Newton’s methods for
   optimization.

-  Reverse mode, in which case we will store the Jacobian at each step.

Basic use case
--------------

We demonstrate a possible use case: Newton root finding for
:math:`y=x^2-1` with *DeriveAlive.*

::

      ## Install at command line as in Section 4.4
      pip install DeriveAlive
      python
      >>> import DeriveAlive.DeriveAlive as da
      >>> import numpy as np
      
      # Initial guess: root at 0.5
      # Expect root at 1.0
      >>> x0 = da.Var([0.5])
      >>> f = x0 ** 2 - 1

      # Newton root finding method
      >>> error = 1
      >>> while error > 0.000001:
            x1 = x0 - (f.val / f.der)
            error = da.abs(x1 - x0) / da.abs(x0)
            x0 = x1

      # Expect root x = 1.0
      >>> print (x1.val)
      1.0

