# Optimization Suite for DeriveAlive module
# Harvard University, CS 207, Fall 2018
# Authors: Chen Shi, Stephen Slater, Yue Sun

import DeriveAlive.DeriveAlive as da
import numpy as np

def _get_unit_vec(length, pos):
	arr = np.zeros(length)
	arr[pos] = 1
	return arr


def _GradientDescentVector(f, var_list, tol=1e-10, iters=10000, eta=0.01):
	# Number of variables
	m = len(var_list)

	# Initial step
	g = f(var_list)
	values_flat = np.reshape(np.array([x_i.val for x_i in var_list]), [-1])
	var_path = [values_flat]
	g_path = [g.val]

	for i in range(iters):
		# Take step in direction of steepest descent
		step = eta * g.der
		values_flat = values_flat - step
		var_list = [da.Var(v_i, _get_unit_vec(m, i)) for i, v_i in enumerate(values_flat)]
		g = f(var_list)
		var_path.append(values_flat)
		g_path.append(g.val)

		# If step size is below tolerance, no need to continue
		cond = np.linalg.norm(step)
		if cond < tol:
			# print ("Reached tol in {} iterations".format(i + 1))
			break
	else:
		# print ("Reached {} iterations without satisfying tolerance.".format(iters))
		pass
	
	minimum = da.Var(values_flat, g.der)
	var_path = np.reshape(np.concatenate((var_path)), [-1, m])
	g_path = np.concatenate(g_path)
	return minimum, var_path, g_path


def GradientDescent(f, x, tol=1e-10, iters=10000, eta=0.01, data=[]):
	if isinstance(x, list):
		# Number of variables
		m = len(x)

		# Convert to da.Var type
		if m > 1:
			for i in range(len(x)):
				if not isinstance(x[i], da.Var):
					x[i] = da.Var(x[i], _get_unit_vec(m, i))
			return _GradientDescentVector(f, x, iters=iters, eta=eta)
		x = x[0]
	if not isinstance(x, da.Var):
		x = da.Var(x)

	# Initial step
	g = f(x)
	var_path = [x.val]
	g_path = [g.val]

	for i in range(iters):
		# Take step in direction of steepest descent
		step = da.Var(eta * g.der, None)
		x = x - step
		g = f(x)
		var_path.append(x.val)
		g_path.append(g.val)

		# If step size is below tolerance, no need to continue
		cond = -step if step < 0 else step
		if cond < tol:
			# print ("Reached tol in {} iterations".format(i + 1))
			break
	else:
		# print ("Reached {} iterations without satisfying tolerance.".format(iters))
		pass

	minimum = da.Var(x.val, g.der)
	var_path = np.reshape(np.concatenate((var_path)), [-1])
	g_path = np.concatenate(g_path)
	return minimum, var_path, g_path
