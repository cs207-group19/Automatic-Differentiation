# Optimization Suite for DeriveAlive module
# Harvard University, CS 207, Fall 2018
# Authors: Chen Shi, Stephen Slater, Yue Sun

import DeriveAlive.DeriveAlive as da
import numpy as np
import matplotlib.pyplot as plt

def _get_unit_vec(length, pos):
	arr = np.zeros(length)
	arr[pos] = 1
	return arr

def _NewtonRootVector(f, var_list, iters=2000, tol=1e-10, der_shift=1):
	# print ("\n\n\nNew function")
	# print ("init guess: {}".format(var_list))

	# Number of variables
	m = len(var_list)
	vars_path = []
	g_path = []

	for i in range(iters):
		g = f(var_list)
		values = np.array([x_i.val for x_i in var_list])
		values_flat = np.reshape(values, [-1])
		vars_path.append(values_flat)
		g_path.append(g.val)

		# Check if guess is a root
		if np.array_equal(g.val, np.zeros((g.val.shape))):
			return da.Var(values_flat, g.der), vars_path, g_path

		# If derivative is extremely close to 0, set to +1 or -1 as a form of random restart
		# This avoids making an entry of the new guess vector at, e.g., 1e10
		if np.linalg.norm(g.der) < tol:
			g.der = np.ones(g.der.shape) * (der_shift if np.random.random() < 0.5 else -der_shift)

		if len(g.der.shape) == 1:
			g_der_pinv = np.linalg.pinv(np.expand_dims(g.der, 0))

		step = g_der_pinv * g.val
		values = values - step
		values_flat = np.reshape(values, [-1])
		var_list = [da.Var(v_i, _get_unit_vec(m, i)) for i, v_i in enumerate(values_flat)]

		# If step size is below tolerance, no need to continue
		cond = np.linalg.norm(step)
		if cond < tol:
			# print ("Reached tol in {} iterations".format(i))
			break
	else:
		print ("Reached {} iterations without satisfying tolerance.".format(iters))
	
	return da.Var(values_flat, g.der), vars_path, g_path


def NewtonRoot(f, x, iters=2000, tol=1e-10, der_shift=1):
	is_vec_input = len(x) > 1
	if is_vec_input:
		return _NewtonRootVector(f, x)
	else:
		x = x[0]

	x_path = []
	g_path = []

	# Run Newton's root-finding method
	for i in range(iters):
		g = f(x)
		x_path.append(x.val)
		g_path.append(g.val)

		# Check if guess is a root
		if np.array_equal(g.val, np.zeros((g.val.shape))):
			return da.Var(x.val, g.der), x_path, g_path

		# If derivative is extremely close to 0, set to +1 or -1 as a form of random restart
		# This avoids making a new guess at, e.g., x + 1e10
		if np.linalg.norm(g.der) < tol:
			g.der = np.ones(g.der.shape) * (der_shift if np.random.random() < 0.5 else -der_shift)

		# Take step and include in path
		step = da.Var(g.val / g.der, None)
		x = x - step

		# Avoid using abs(step) in case guess is at 0, because derivative is not continuous
		cond = np.linalg.norm(step.val) if is_vec_input else -1 * step if step < 0 else step
		
		# If step size is below tolerance, no need to continue
		if cond < tol:
			# print ("Reached tol in {} iterations".format(i))
			break
	else:
		print ("Reached {} iterations without satisfying tolerance.".format(iters))

	return da.Var(x.val, g.der), x_path, g_path


def NewtonOptimization(f, x0, tol=1e-7, iters=1000):
	pass


def SteepestDescent(f, x0, tol=1e-7, iters=1000):
	pass
