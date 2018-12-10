How to Use *DeriveAlive*
========================

How to install
--------------

| The user can either download from GitHub or use use ``pip`` to install
  the package. Installation details are listed in Section 4.4. After
  installation, the user will need to import our package (see pseudocode
  below). Implicitly, this will import other dependencies (such as
  ``numpy``), since we include those dependencies as imports in our
  module. Then, the user will define an input of type ``Var`` in our
  module. After this, the user can define a function in terms of this
  ``Var``. Essentially, the user will give the initial input :math:`x`
  and then apply :math:`f` to :math:`x` and store the new value and
  derivative with respect to :math:`x` inside :math:`f`. At each step of
  the evaluation, the program will process nodes in the implicit
  computation graph in order, propagating values and derivatives. The
  final output yield another ``Var`` containing :math:`f(x)` and
  :math:`f'(x)`.

Basic demo
----------

For example, consider the case
$f(x) = \sin(x) + 5 \cdot tan(x/2)$. If the user wants to evaluate
:math:`f(x)` at :math:`x = a`, where $a = \pi/2$, the user will
instantiate a ``Var`` object as ``da.Var(np.pi/2)``. Then, the user will
give the initial input :math:`a` and set $y = f(a)$, which stores
:math:`f(a)` and :math:`f'(a)` as attributes inside the ``Var``
:math:`y`. This functionality will propagate throughout the graph with
more variables in a recursive structure, where each evaluation trace
creates a new ``Var``. See the code below for a demonstration. Note that
:math:`\sin(\pi/2) = 1.0` and :math:`\tan(\pi/4) = 1.0`, and their
evaluated derivatives :math:`\cos(\pi/2) = 0` and
:math:`\frac{1}{\cos^2(\pi/4)} = 2`. We assign :math:`y = f(a)` by
explicitly applying the ``Var``-specific operations of :math:`f` to
:math:`a`. The closed form of the derivative is
:math:`f'(x) = \cos(x) + 5 * \frac{1}{\cos^2(x/2)} * \frac{1}{2}`.

::

      # Install at command line using installation instructions in Section 4.4
      pip install DeriveAlive
      python
      >>> import DeriveAlive.DeriveAlive as da
      >>> import numpy as np
      
      # Expect value of 6.0, derivative of 5.0
      >>> x = da.Var([np.pi / 2])
      >>> y = x.sin() + 5 * (x / 2).tan()
      >>> print (y.val, y.der)
      [6.0] [5.0]
