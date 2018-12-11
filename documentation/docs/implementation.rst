Implementation
==============

Forward Mode Implementation
---------------------------

-  Variable domain: The variables are defined as real numbers, hence any
   calculations or results involving complex numbers will be excluded
   from the package.

-  Type of user input: Regardless of the input type (e.g., an int, a float or a
   list or a numpy array), the ``Var`` class will automatically convert
   the input into a numpy array. This will provide flexibility in the
   future for implementing vector to vector functions.

-  Core data structures: The core data structures will be classes, lists
   and numpy arrays.

   -  Classes will help us provide an API for differentiation and custom
      functions, including custom methods for our elementary functions.

   -  Lists will help us maintain the collection of trace variables and
      output functions (in the case of multi-output models) from the
      computation graph in order. For example, if we have a function
      :math:`f(x): \mathbb{R}^1 \rightarrow \mathbb{R}^2`, then we store
      :math:`f = [f1, f2]`, where we have defined :math:`f1` and
      :math:`f2` as functions of :math:`x`, and we simply process the
      functions in order. Depending on the extensions we choose for the
      project, we may use lists to store the parents of each node in the
      graph.

   -  Numpy arrays are the main data structure during the calculation.
      We store the list of derivatives as a numpy array so that we can
      apply entire functions to the array, rather than to each entry
      separately. Each trace ``Var`` has a numpy array of derivatives
      where the length of the array is the number of input variables in
      the function. In the vector-vector case, if we have a function
      :math:`f: \mathbb{R}^m \rightarrow \mathbb{R}^n` or :math:`f: \mathbb{R}^1 \rightarrow \mathbb{R}^n`, we can process
      this as :math:`f = [f_1, f_2, \ldots, f_n]`, where each
      :math:`f_i` is a function
      :math:`f_i: \mathbb{R}^m \rightarrow \mathbb{R}`. Our
      implementation can act as a wrapper over these functions, and we
      can evaluate each :math:`f_i` independently, so long as we define
      :math:`f_i` in terms of the :math:`m` inputs. : Currently, the module 
      supports both scalar to scalar and vector to vector functions.

-  Our implementation plan currently includes 1 class which accounts for
   trace variables and derivatives with respect to each input variable.

   -  ``Var`` class. The class instance itself has two main attributes:
      the value and the evaluated derivatives with respect to each
      input. Within the class we redefine the elementary functions and
      basic algebraic functions, including both evaluation and
      derivation. Since our computation graph includes “trace"
      variables, this class will account for each variable. Similar to a
      dual number, this class structure will allow us easy access to
      necessary attributes of each variable, such as the trace
      evaluation and the evaluated derivative with respect to each input
      variable. This trace table would also be of possible help in
      future project extensions.

API
---

``DeriveAlive.DeriveAlive``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Class attributes and methods:

   -  Attributes in ``Var``: ``self.var``, ``self.der``. To cover
      vector-to-vector cases, we implement our ``self.var`` and
      ``self.der`` as numpy arrays, in order to account for derivatives
      with respect to each input variable. Also the constructor checks
      whether the values and derivatives are integers, floats, or lists,
      and transforms them into numpy arrays automatically.

   -  We have overloaded elementary mathematical operations such as
      addition, subtraction, multiplication, division, sine, pow, log,
      etc. that take in :math:`1` ``Var`` type, or :math:`2` types, or
      :math:`1` and :math:`1` constant, and return a new ``Var`` (i.e.
      the next “trace" variable). All other operations on constants will
      use the standard Python library. In each ``Var``, we will store as
      attributes the value of the variable (which is calculated based on
      the current operation and previous trace variables) and the
      evaluated gradient of the variable with respect to each input
      variable.

   -  Methods in ``Var``:

      -  ``__init__``: initialize a ``Var`` class object, regardless of
         the user input, with values and derivatives stored as numpy
         arrays.

      -  ``__add__``: overload add function to handle addition of
         ``Var`` class objects and addition of and non-\ ``Var``
         objects.

      -  ``__radd__``: preserve addition commutative property.

      -  ``__sub__``: overload subtraction function to handle
         subtraction of ``Var`` class objects and subtraction between
         and non-\ ``Var`` objects.

      -  ``__rsub__``: allow subtraction for :math:`a - \texttt{Var}`
         case where a is a float or an integer.

      -  ``__mul__``: overload multiplication function to handle
         multiplication of ``Var`` class objects and multiplication
         between and non-\ ``Var`` objects.

      -  ``__rmul__``: preserve multiplication commutative property.

      -  ``__truediv__``: overload division function to handle division
         of ``Var`` class objects over floats or integers.

      -  ``__rtruediv__``: allow division for
         :math:`a \div \texttt{Var}` case where :math:`a` is a float or
         an integer.

      -  ``__neg__``: return negated ``Var``.

      -  ``__abs__``: return the absolute value of ``Var``.

      -  ``__eq__``: return ``True`` if two ``Var`` objects have the
         same value and derivative, ``False`` otherwise.

      -  ``__pow__``, ``__rpow__``, ``pow``: extend power functions to
         ``Var`` class objects.

      -  ``log``: extend logarithmic functions to ``Var`` class objects.

      -  ``exp``: extend exponential functions to ``Var`` class objects.

      -  ``sin``, ``cos``, ``tan``: extend trigonometric functions to
         ``Var`` class objects.

      -  ``arcsin``, ``arccos``, ``arctan``: extend inverse
         trigonometric functions to ``Var`` class objects.

      -  ``sinh``, ``cosh``, ``tanh``: extend hyperbolic functions to
         ``Var`` class objects.

-  External dependencies:

   -  ``NumPy`` - This provides an API for a large collection of
      high-level mathematical operations. In addition, it provides
      support for large, multi-dimensional arrays and matrices.

   -  ``doctest`` - This module searches for pieces of text that look
      like interactive Python sessions (typically within the
      documentation of a function), and then executes those sessions to
      verify that they work exactly as shown.

   -  ``pytest`` - This is an alternative, more Pythonic way of writing
      tests, making it easy to write small tests, yet scales to support
      complex functional testing. We plan to use this for a
      comprehensive test suite.

   -  | ``setuptools`` - This package allows us to create a package out
        of our project for easy distribution. See more information on
        packaging instructions here:
      | https://packaging.python.org/tutorials/packaging-projects/.

   -  Test suites: Travis CI, Coveralls

-  Elementary functions

   -  Our explanation of our elementary functions is included in the
      “Class attributes and methods" section above. For the elementary
      functions, we defined our own custom methods within the ``Var``
      class so that we can calculate, for example, the :math:`\sin(x)`
      of a variable :math:`x` using a package such as ``numpy``, and
      also store the proper gradient (:math:`\cos(x)dx`) to propagate
      the gradients forward. For example, consider a scalar function
      where ``self.val`` contains the current evaluation trace and
      ``self.der`` is a numpy array of the derivative of the current
      trace with respect to the input. When we apply :math:`\sin`, we
      propagate as follows:

      ::

           def sin(self):
                      val = np.sin(self.val)
                      der = np.cos(self.val) * self.der
                      return Var(val, der)
                  

      The structure of each elementary function is that it calculates
      the new value (based on the operation) and the new derivative, and
      then returns a new ``Var`` with the updated arguments.

``DeriveAlive.optimization``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``DeriveAlive.root_finding``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


``DeriveAlive.spline``
~~~~~~~~~~~~~~~~~~~~~~

