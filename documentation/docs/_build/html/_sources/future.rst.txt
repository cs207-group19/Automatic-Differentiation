Future
======

Currently, our ``DeriveAlive``  can handle scalar to scalar, scalar to vector, vector to scalar and vector to vector functions.  The further improvement for the software can be expected as follows:

Input Format Update
-------------------

Now our users have to define each input variable themselves when performing vector to vector or vector to scalar functions, e.g they would like to use ``DeriveAlive`` for :math:`f(x,y,z)`, they have to define:

::

	    >>> x = da.Var([2], [1, 0, 0])
	    >>> y = da.Var([3], [0, 1, 0])
	    >>> y = da.Var([4], [0, 0, 1])


It is obvious that it wouldn’t be convenient for our users to do this when they have high dimension day (i.e. 100 dimensions, they have to input [1, 0, ….,0], …, 100 times!). In the future, we would like them to define in this way:

::

      >>> x = da.Var([2],’3, 1’)
      >>> y = da.Var([3],’3, 2’)
      >>> z = da.Var([4],’3, 3’)


where, the first element in the string is the dimension of the input vector (here 100), and the second element is the position of the ‘1’ in the initialized Jacobian matrix.

This improvement will make our software more user-friendly and easier to work with.

Module Extension
----------------

Now our ``DeriveAlive`` can work perfectly with the forward mode, we are expecting the implement the reverse mode also. This improvement will allow our users to play with Neural Network models using backpropagation.

We also want to extend the quadratic spline suite to a cubic spline suite or even higher order splines, which would utilize higher order derivatives to be implemented using autodifferentiation.
