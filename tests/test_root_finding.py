# Test suite for optimization suite
# Harvard University, CS 207, Fall 2018
# Authors: Chen Shi, Stephen Slater, Yue Sun

import DeriveAlive as da
import numpy as np

# User-defined constant
x_const = da.Var([2], None)

# Example of applying a user-defined function to a da.Var
# print (f(x_const))
# print (f(x_var))

# Integration test
def f(var):
	return (var - 1) ** 2 - 1
x_var = da.Var(2)
y_var = da.Var(0)
# Roots are at 0 and 2
output = NewtonRoot(f, x_var)
np.testing.assert_array_equal(output.val, [2.0])
np.testing.assert_array_equal(output.der, [2.0])
output2 = NewtonRoot(f, y_var)
np.testing.assert_array_equal(output2.val, [0.0])
np.testing.assert_array_equal(output2.der, [-2.0])