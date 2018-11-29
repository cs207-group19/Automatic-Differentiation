# Optimization Suite for DeriveAlive module
# Harvard University, CS 207, Fall 2018
# Authors: Chen Shi, Stephen Slater, Yue Sun

import DeriveAlive as da
import numpy as np

# User-defined constant
x_const = da.Var([2], None)

# User-defined variable
x_var = da.Var([2])

def f(var):
	return var ** 2

# Example of applying a user-defined function to a da.Var
print (f(x_const))
print (f(x_var))


def Newton(f, x0, tol=0.0001, iters=100):
	# Run Newton's method
	pass

def SteepestDescent(f, x0, tol=0.0001, iters=100):
	pass


def QuasiNewton(f, x0, tol=0.0001, iters=100):
	pass


def Secant(f, x0, tol=0.0001, iters=100):
	pass

