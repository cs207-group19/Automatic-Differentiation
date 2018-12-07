# Optimization Suite for DeriveAlive module
# Harvard University, CS 207, Fall 2018
# Authors: Chen Shi, Stephen Slater, Yue Sun

import DeriveAlive.DeriveAlive as da
import numpy as np
import matplotlib.pyplot as plt

def NewtonRoot(f, x, iters=2000, tol=1e-7, der_shift=1):
	is_vec_input = len(x.val) > 1

	# Run Newton's root-finding method
	for i in range(iters):
		g = f(x)

		if i == 0:
			x_path = [x.val]
			y_path = [g.val]

		# Check if initial guess is a root
		if np.array_equal(g.val, np.zeros((g.val.shape))):
			return da.Var(x.val, g.der), x_path, y_path

		# If derivative is extremely close to 0, set to +1 or -1 as a form of random restart
		# This avoids making a new guess at, e.g., x + 1e10
		if np.linalg.norm(g.der) < tol:
			g.der = np.ones(g.der.shape) * (der_shift if np.random.random() < 0.5 else -der_shift)

		# Take step and include in path
		step = da.Var(g.val / g.der, None)
		x = x - step
		x_path.append(x.val)
		y_path.append(g.val)

		# Avoid using abs(step) in case root is at 0, because derivative is not continuous
		cond = np.linalg.norm(step.val) if is_vec_input else -1 * step if step < 0 else step
		
		# If step size is below tolerance, no need to continue
		if cond < tol:
			# print ("Reached tol in {} iterations".format(i))
			break
	else:
		print ("Reached {} iterations without satisfying tolerance.".format(iters))

	return da.Var(x.val, g.der), x_path, y_path


def NewtonOptimization(f, x0, tol=1e-7, iters=1000):
	pass


def SteepestDescent(f, x0, tol=1e-7, iters=1000):
	pass
