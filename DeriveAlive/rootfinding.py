# Optimization Suite for DeriveAlive module
# Harvard University, CS 207, Fall 2018
# Authors: Chen Shi, Stephen Slater, Yue Sun

import DeriveAlive.DeriveAlive as da
import numpy as np

def _get_unit_vec(length, pos):
	arr = np.zeros(length)
	arr[pos] = 1
	return arr

def _NewtonRootVector(f, var_list, iters=2000, tol=1e-10, der_shift=1):
	# Number of variables
	m = len(var_list)

	# Initial step
	g = f(var_list)
	values = np.array([x_i.val for x_i in var_list])
	values_flat = np.reshape(values, [-1])
	var_path = [values_flat]
	g_path = [g.val]

	for i in range(iters):
		# Check if guess is a root
		if np.array_equal(g.val, np.zeros((g.val.shape))):
			break

		# If derivative is extremely close to 0, set to +1 or -1 as a form of random restart
		# This avoids making an entry of the new guess vector at, e.g., 1e10
		if np.linalg.norm(g.der) < tol:
			g.der = np.ones(g.der.shape) * (der_shift if np.random.random() < 0.5 else -der_shift)

		# Appropriate shape for taking pseudoinverse
		if len(g.der.shape) == 1:
			g_der_pinv = np.linalg.pinv(np.expand_dims(g.der, 0))

		# Take step in direction of steepest descent
		step = g_der_pinv * g.val
		values = values - step
		values_flat = np.reshape(values, [-1])
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
	
	root = da.Var(values_flat, g.der)
	var_path = np.reshape(np.concatenate((var_path)), [-1, m])
	g_path = np.concatenate(g_path)
	return root, var_path, g_path


def NewtonRoot(f, x, iters=2000, tol=1e-10, der_shift=1):
	if isinstance(x, list):
		# Number of variables
		m = len(x)

		# Convert to da.Var type
		if m > 1:
			for i in range(m):
				if not isinstance(x[i], da.Var):
					x[i] = da.Var(x[i], _get_unit_vec(m, i))
			return _NewtonRootVector(f, x, iters=iters, tol=tol, der_shift=der_shift)
		x = x[0]
	if not isinstance(x, da.Var):
		x = da.Var(x)

	# Initial step	
	g = f(x)
	var_path = [x.val]
	g_path = [g.val]

	# Run Newton's root-finding method
	for i in range(iters):
		# Check if guess is a root
		if np.array_equal(g.val, np.zeros((g.val.shape))):
			break

		# If derivative is extremely close to 0, set to +1 or -1 as a form of random restart
		# This avoids making a new guess at, e.g., x + 1e10
		if np.linalg.norm(g.der) < tol:
			g.der = np.ones(g.der.shape) * (der_shift if np.random.random() < 0.5 else -der_shift)

		# Take step and include in path
		step = da.Var(g.val / g.der, None)
		x = x - step
		g = f(x)
		var_path.append(x.val)
		g_path.append(g.val)

		# Avoid using abs(step) in case guess is at 0, because derivative is not continuous
		cond = -step if step < 0 else step
		
		# If step size is below tolerance, no need to continue
		if cond < tol:
			# print ("Reached tol in {} iterations".format(i + 1))
			break
	else:
		# print ("Reached {} iterations without satisfying tolerance.".format(iters))
		pass

	root = da.Var(x.val, g.der)
	var_path = np.reshape(np.concatenate((var_path)), [-1])
	g_path = np.concatenate(g_path)
	return root, var_path, g_path
