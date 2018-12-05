# Optimization Suite for DeriveAlive module
# Harvard University, CS 207, Fall 2018
# Authors: Chen Shi, Stephen Slater, Yue Sun

import DeriveAlive as da
import numpy as np

# User-defined variable
x_var = da.Var([2.00])
y_var = da.Var([0])
z_var = da.Var([1])

def f(var):
	return (var - 1) ** 2 - 1

def NewtonRoot(f, x0, tol=1e-7, iters=1000):
	is_vec_input = len(x0.val) > 1

	print ("inside NewtonRoot, init guess is:\n{}".format(x0))

	# Run Newton's root-finding method
	x = x0
	
	# Check if initial guess is a root
	g = f(x)
	if np.array_equal(g.val, np.zeros((g.val.shape))):
		return da.Var(x.val, g.der)

	for i in range(iters):
		g = f(x)
		if np.array_equal(g.der, np.zeros((g.der.shape))):
			g = g + tol * x0
		step = da.Var(g.val / g.der, None)

		# If step size is below tolerance, then no need to update step
		cond = np.linalg.norm(step.val) if is_vec_input else abs(step)
		print ("condition: {}".format(cond))
		if cond < tol:
			print ("Reached tol in {} iterations".format(i))
			break
		print ("x is:\n{}".format(x))
		x = x - step
	else:
		print ("Reached {} iterations without satisfying tolerance.".format(iters))

	return da.Var(x.val, g.der)


def NewtonOptimization(f, x0, tol=1e-7, iters=1000):
	pass


def SteepestDescent(f, x0, tol=1e-7, iters=1000):
	pass

# for x in range(-1, 5):
# 	x_var = da.Var(x)
# 	print ("\n")
# 	print (NewtonRoot(f, x_var))


# Attempt at vector case
# x = da.Var(1, [1, 0])
# y = da.Var(1, [0, 1])
# init_guess = da.Var([x, y])

# h = vec ** 2
# x = vec.val[0]
# y = vec.val[1]
# return da.Var(x ** 2 + y ** 2, [vec.der[:, 0] + vec.der[:, 1]])


def r2_to_r2():
	init_guess = da.Var([1, 2], [1, 1])

	def z(vec):
		return vec ** 2

	print ("\n\nTRYING R2 -> R2 CASE. EXPECT ROOT AT (0, 0) FOR Z(X, Y) = [X**2, Y**2].\n")
	print (NewtonRoot(z, init_guess))


def r2_to_r1():
	# init_guess = da.Var([1, 2], [1, 1])
	x = vec.val[0]
	y = vec.val[1]
	init_guess = [x, y]

	def z(vec):
		return da.Var(vec.val[0] + vec.val[1], [vec.der[0] + vec.der[1]])

	def z(vec):
		return vec[0] + vec[1]
		return x + y

	print ("\n\nTRYING R2 -> R1 CASE. EXPECT ROOT AT (0, 0) FOR Z(X, Y) = X + Y.\n")
	print (NewtonRoot(z, init_guess))


# r2_to_r2()
r2_to_r1()