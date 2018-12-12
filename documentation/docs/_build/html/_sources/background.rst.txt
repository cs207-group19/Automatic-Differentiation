
Background
==========

The chain rule, gradient (Jacobian), computational graph, elementary
functions and several numerical methods serve as the mathematical
cornerstone for this software. The mathematical concepts here come from
CS 207 Lectures 9 and 10 on Autodifferentiation.

The Chain Rule
--------------

| The chain rule is critical to AD, since the derivative of the function
  with respect to the input is dependent upon the derivative of each
  trace in the evaluation with respect to the input.
| If we have :math:`h(u(x))` then the derivative of :math:`h` with
  respect to :math:`x` is:
| 

  .. math:: \frac{\partial h}{\partial x} =\frac{\partial h}{\partial u} \cdot \frac{\partial u}{\partial x}

| If we have another argument :math:`h(u, v)` where :math:`u` and
  :math:`v` are both functions of :math:`x`, then the derivative of
  :math:`h(x)` with respect to :math:`x` is:
| 

  .. math:: \frac{\partial h}{\partial x} =\frac{\partial h}{\partial u} \cdot \frac{\partial u}{\partial x} + \frac{\partial h}{\partial v} \cdot \frac{\partial v}{\partial x}

Gradient and Jacobian
---------------------

If we have :math:`x\in\mathbb{R}^{m}` and function
:math:`h\left(u\left(x\right),v\left(x\right)\right)`, we want to
calculate the gradient of :math:`h` with respect to :math:`x`:

.. math:: \nabla_{x} h = \frac{\partial h}{\partial u}\nabla_x u + \frac{\partial h}{\partial v} \nabla_x v

In the case where we have a function
:math:`h(x): \mathbb{R}^m \rightarrow \mathbb{R}^n`, we write the
Jacobian matrix as follows, allowing us to store the gradient of each
output with respect to each input.

.. math:: J =       \begin{bmatrix}
					  \frac{\partial h_1}{\partial x_1} & 
					    \frac{\partial h_1}{\partial x_2} & \ldots &
					    \frac{\partial h_1}{\partial x_m} \\[1ex] 
					  \frac{\partial h_2}{\partial x_1} & 
					    \frac{\partial h_2}{\partial x_2} & \ldots &
					    \frac{\partial h_2}{\partial x_m} \\[1ex]
					    \vdots & \vdots & \ddots  & \vdots \\[1ex]
					  \frac{\partial h_n}{\partial x_1} & 
					    \frac{\partial h_n}{\partial x_2} & \ldots &
					    \frac{\partial h_n}{\partial x_m}
					\end{bmatrix}

In general, if we have a function :math:`g\left(y\left(x\right)\right)`
where :math:`y\in\mathbb{R}^{n}` and :math:`x\in\mathbb{R}^{m}`. Then
:math:`g` is a function of possibly :math:`n` other functions, each of
which can be a function of :math:`m` variables. The gradient of
:math:`g` is now given by

.. math:: \nabla_{x}g = \sum_{i=1}^{n}{\frac{\partial g}{\partial y_{i}}\nabla_x y_{i}\left(x\right)}.

The Computational Graph
-----------------------

Let us visualize what happens during the evaluation trace. The following 
example is based on Lectures 9 and 10.
Consider the function:

.. math:: f\left(x\right) = x - \exp\left(-2\sin^{2}\left(4x\right)\right)

If we want to evaluate :math:`f` at the point :math:`x`, we construct a
graph where the input value is :math:`x` and the output is :math:`y`.
Each input variable is a node, and each subsequent operation of the
execution trace applies an operation to one or more previous nodes (and
creates a node for constants when applicable).

.. image:: images/computationgraph.png
  :width: 600       

As we execute :math:`f(x)` in the “forward mode", we can propagate not
only the sequential evaluations of operations in the graph given
previous nodes, but also the derivatives using the chain rule.

Elementary functions
--------------------

An elementary function is built up of a finite combination of constant
functions, field operations :math:`(+, -, \times, \div)`, algebraic,
exponential, trigonometric, hyperbolic and logarithmic functions and
their inverses under repeated compositions. Below is a table of some
elementary functions and examples that we will include in our
implementation.

======================== ============
Elementary Functions     Example
======================== ============
        powers           :math:`x^2`
        roots            :math:`\sqrt{x}`
     exponentials        :math:`e^{x}`  
      logarithms         :math:`\log(x)`  
    trigonometrics       :math:`\sin(x)` 
 inverse trigonometrics  :math:`\arcsin(x)` 
     hyperbolics         :math:`\sinh(x)`
======================== ============

.. note:: Background for additional features, `Newton's root finding method <https://cs-207-final-project-group-19.readthedocs.io/en/latest/additional_features.html#background>`_, `Gradient Descent & BFGS <https://cs-207-final-project-group-19.readthedocs.io/en/latest/additional_features.html#id1>`_, `quadratic splines <https://cs-207-final-project-group-19.readthedocs.io/en/latest/additional_features.html#id3>`_,  can be found in `Additional Features<https://cs-207-final-project-group-19.readthedocs.io/en/latest/additional_features.html#>`_.
