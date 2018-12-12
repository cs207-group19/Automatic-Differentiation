Future
======

Currently, our ``DeriveAlive``  can handle scalar to scalar, scalar to vector, vector to scalar and vector to vector functions.  The further improvement for the software can be expected as follows:


Module Extension
----------------

- Reverse mode. Now that our ``DeriveAlive`` can work perfectly with the forward mode, we are expecting to implement the reverse mode as well. This improvement will allow our users to play with custom Neural Network models using backpropagation.

- Hessian. By calculating and storing the second derivatives in a Hessian matrix, we can make use of more applications of automatic differentiation that use second derivatives, such as Newton optimization and cubic splines.

- Higher-order splines (cubic). We also want to extend the quadratic spline suite to a cubic spline suite or even higher order splines, which would utilize higher order derivatives to be implemented using autodifferentiation. We would also like to allow users to draw any custom plots with this module.
