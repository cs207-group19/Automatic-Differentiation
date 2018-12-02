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
# Regression tests for Newton root finding for scalars 
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

# Regression tests for Newton root finding for scalars using a more complex function
x_var = da.Var(0.1)# Initial guess
def f(x):
    return x - np.exp(-2.0 * np.sin(4.0*x) * np.sin(4.0*x))

output = NewtonRoot(f, x_var)
np.testing.assert_array_equal(np.round(output.val,4), [0.2474])
np.testing.assert_array_equal(np.round(output.der,4), [2.8164])

# Regression tests for Newton root finding for vectors 
x = da.Var(np.pi/2, [1, 0, 0])
y = da.Var(np.pi/3, [0, 1, 0])
z = da.Var(2.0, [0, 0, 1])

#def F(x):
#        return da.Var(
#            [x[0]**2 - x[1] + x[0]* np.cos(np.pi*x[0]),
#             x[0]*x[1] + np.exp(-x[1]) - x[0]**2])

#expected = np.array([1, 0])
#x1=da.Var([2],[1, 0])
#x2=da.Var([-1],[0, 1])
#x = da.Var([x1, x2])

def F(x1, x2):
        return da.Var(
            [x1 ** 2 - x2 + x1 * np.cos(np.pi*x1),
             x1 * x2 + np.exp(-x2) - x1 ** 2])
x1=da.Var([2],[1, 0])
x2=da.Var([-1],[0, 1])
x = da.Var([x1, x2])
output = NewtonRoot(F(x1, x2), x)


