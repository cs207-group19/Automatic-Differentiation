# Optimization Suite for DeriveAlive module
# Harvard University, CS 207, Fall 2018
# Authors: Chen Shi, Stephen Slater, Yue Sun

import DeriveAlive.DeriveAlive as da
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def _get_unit_vec(length, pos):
	'''Internal function to compute a unit vector, used for intializing derivatives.'''
	arr = np.zeros(length)
	arr[pos] = 1
	return arr

def _NewtonRootVector(f, var_list, iters=2000, tol=1e-10, der_shift=1):
	'''Internal function that may get called from the API GradientDescent method.

	See documentation of rootfinding.NewtonRoot.
	'''

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


def NewtonRoot(f, x, tol=1e-10, iters=2000, der_shift=1):
	""" Run the gradient descent algorithm to minimize a function f.
		
		Parameters
		----------
		f: callable (function), or string
		   if :math:`f` is a vector to scalar function, it must take as argument a single list
		   of variables (of type int, float, or ``DeriveAlive.Var``). If :math:`f` is a scalar
		   to scalar function, the input can be of type int, float, or ``DeriveAlive.Var``, but 
		   should not be a list.
		
		x: int, float, ``DeriveAlive.Var`` object, or list of ``DeriveAlive.Var`` (multivariate case).
		   initial guess/starting point for root.

		iters: integer, optional (iters=10000)
			   maximum number of iterations to run the algorithm.
		
		tol: float, optional (default=1e-10)
			 this denotes the stopping criterion/tolerance for the algorithm. 
			 The algorithm terminates if the next absolute step size is less than :math:`tol`. 
			 In the multivariate case, the step size is determined by taking the L2 norm.
		
		der_shift: int or float, optional (default=1)
				   this serves as a random restart value for the derivative in the case that a
				   root guess has derivative 0, since the iteration would require division by 0.
				   A large der_shift (i.e. 1) provides a quicker route to convergence than a small
				   der_shift (e.g., 1e-4), since the iteration step divides by the derivative, and
				   would predict extreme values for :math:`x_{t+1}` if der_shift were small.
			
		Returns
		-------
		root: ``DeriveAlive.Var``
				 contains the optimal scalar or vector solution and the derivative at that point.
		
		var_path: numpy.ndarray (min(iters, t), m)
				  contains the path of the input variable(s) throughout the min(iters, t) steps,
				  where iters is defined above and t is the number of steps needed to satisfy tol.
		
		g_path: numpy.ndarray (min(iters, t),)
				contains the path of the evaluations of :math:`f` throughout the min(iters, t) steps.
		
		"""

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


def plot_results(f, var_path, f_path, f_string, x_lims=None, y_lims=None, num_points=100, 
				 threedim=False, fourdim=False, animate=False, speed=500):
	""" Plot the results of NewtonRoot. Use the return values (var_path, f_path) from
	    NewtonRoot when plotting your results to show the steps of the algorithm.
		
		Parameters
		----------
		f: callable (function)
		   if :math:`f` is a vector to scalar function, it must take as argument a single list
		   of variables (of type int, float, or ``DeriveAlive.Var``). If :math:`f` is a scalar
		   to scalar function, the input can be of type int, float, or ``DeriveAlive.Var``, but 
		   should not be a list.
		
		var_path: numpy.ndarray (min(iters, t), m), where m is dimension of input and t is the
				  number of iterations required to satisfy tolerance.
		   		  The rows represent the consecutive steps of the algorithm towards a solution.

		f_path: numpy.ndarray (min(iters, t), 1), where t is the number of iterations required to 
				satisfy tolerance. The rows represent the consecutive values of the objective function
				for each step of the algorithm.

		f_string: str
				  a string denoting the equation whose roots will be solved. Supports LaTeX code,
				  i.e. f_string = 'f(x, y) = x^2 + y^2' will look nice in title of the plot.
		
		x_lims: length-2 iterable, optional (default=None)
			 	this denotes the lower and upperbound values to be plotted on the x-axis. If not
			 	specified, the function will plot over the range (-10, 10).

		y_lims: length-2 iterable, optional (default=None)
			 	this denotes the lower and upperbound values to be plotted on the y-axis. If not
			 	specified, the function will plot over the range (-10, 10).	

		num_points: int, optional (default=100)
			 	this denotes the number of points to plot on the x-axis and y-axis intervals, respectively.		 
		
		threedim: bool, optional (default=False)
				  whether to plot in three dimensions (user must specify a function f(x, y)).

		fourdim: bool, optional (default=False)
				 whether to plot in four dimensions (user must specify a function (f(x, y, z))).

		animate: bool, optional (default=False)
				 whether to animate plots in 2D and 3D.

		speed: int, optional (default=500)
		       number of milliseconds between each additional point in an animated plot.
			
		Returns
		-------
		None

		"""
	if x_lims:
		x_min, x_max = x_lims
		x_range = np.linspace(x_min, x_max, num_points)
	else:
		x_range = np.linspace(-10, 10, num_points)

	if y_lims:
		y_min, y_max = y_lims
	
	# Plot function and path of points
	if threedim:
		xs, ys = var_path[:, 0], var_path[:, 1]
		fs = f_path
		x0, y0 = np.round(xs[0], 4), np.round(ys[0], 4)
		xn, yn = np.round(xs[-1], 4), np.round(ys[-1], 4)

		fig = plt.figure(figsize=(10, 6))
		ax = fig.add_subplot(111, projection='3d')			
		ax.set_title(r'Finding root(s) of ${}$'.format(f_string))

		# Plot surface
		X, Y = np.meshgrid(x_range, x_range)
		zs = np.array([f([x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))])
		Z = zs.reshape(X.shape)
		ax.plot_surface(X, Y, Z, color='0.5', alpha=0.5)	

		# Plot start and end
		ax.scatter(xs[0], ys[0], fs[0], c='red', marker='o', linewidth=6, label='start: {}'.format((x0, y0)))
		ax.scatter(xs[-1], ys[-1], fs[-1], c='green', marker='o', linewidth=6, label='end: {}'.format((xn, yn)))
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_zlabel('f(x,y)')
		
		if animate and len(var_path) > 2:
			line, = ax.plot(xs, ys, fs, 'b-o', label='path')

			def update(num, x, y, z, line):
			    line.set_data(x[:num], y[:num])
			    line.set_3d_properties(z[:num])
			    if x_lims and y_lims:
			    	line.axes.axis([x_min, x_max, y_min, y_max])
			    return line,

			ani = animation.FuncAnimation(fig, update, len(var_path), fargs=[xs, ys, f_path, line],
			                              interval=speed, blit=True, repeat_delay=500, repeat=True)

		else:
			ax.plot(xs, ys, fs, 'b-o', label='path')

		ax.legend(loc='upper left', bbox_to_anchor=(0, 0.5))
		plt.show()

	elif fourdim:
		if animate:
			print ("Sorry, animation is not supported for 4D plots.")

		# Other cmaps: cm.PiYG, cm.tab20c, cm.twilight, cm.nipy_spectral
		cm_type = cm.RdYlGn_r 
		fig = plt.figure(figsize=(10, 6))
		ax = fig.add_subplot(111, projection='3d')
		xs, ys, zs = var_path[:, 0], var_path[:, 1], var_path[:, 2]
		fs = f_path
		x0, y0, z0 = np.round(xs[0], 4), np.round(ys[0], 4), np.round(zs[0], 4)
		xn, yn, zn = np.round(var_path[-1], 4)
		ax.scatter(xs, ys, zs, c=fs, cmap=cm_type)
		ax.plot3D(xs, ys, zs, '-')
		ax.scatter(xs[0], ys[0], zs[0], c='red', marker='o', linewidth=6, label='start: {}'.format((x0, y0, z0)))
		ax.scatter(xs[-1], ys[-1], zs[-1], c='green', marker='o', linewidth=6, label='end: {}'.format((xn, yn, zn)))

		m = cm.ScalarMappable(cmap=cm_type)
		m.set_array(fs)
		cbar = plt.colorbar(m)
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_zlabel('z')
		ax.set_title(r'Finding root(s) of ${}$'.format(f_string))

		# Place legend
		ax.legend(loc='upper left', bbox_to_anchor=(0, 0.85))
		plt.show()

	else:
		x0, f0 = np.round(var_path[0], 4), np.round(f_path[0], 4)
		xn, fn = np.round(var_path[-1], 4), np.round(f_path[-1], 4)

		if animate and len(var_path) > 2:
			fig, ax = plt.subplots()
			ax.plot(x_range, np.zeros(num_points), '0.7')
			ax.set_title(r'Finding root(s) of ${}$'.format(f_string))
			ax.plot(x_range, f(x_range), 'k-', label=r'${}$'.format(f_string))
			plt.plot(x0, f0, 'ro', label='start: {}'.format(x0))
			plt.plot(xn, fn, 'go', label='end: {}'.format(xn))

			line, = ax.plot(var_path, f_path, 'b-o', label='path')

			def update(num, x, y, line):
			    line.set_data(x[:num], y[:num])
			    if x_lims and y_lims:
			    	line.axes.axis([x_min, x_max, y_min, y_max])
			    return line,

			ani = animation.FuncAnimation(fig, update, len(var_path), fargs=[var_path, f_path, line],
			                              interval=200, blit=True, repeat_delay=500, repeat=True)

			ax.legend()
			ax.set_xlabel('x')
			ax.set_ylabel('f(x)')
			plt.show()
		else:
			plt.figure()
			plt.plot(x_range, np.zeros(num_points), '0.7')
			plt.title(r'Finding root(s) of ${}$'.format(f_string))
			plt.plot(x_range, f(x_range), 'k-', label=r'${}$'.format(f_string))
			plt.plot(var_path, f_path, 'b-o', label='path')
			plt.plot(x0, f0, 'ro', label='start: {}'.format(x0))
			plt.plot(xn, fn, 'go', label='end: {}'.format(xn))	

			# Plot details
			if x_lims:
				plt.xlim(x_min, x_max)
			if y_lims:
				plt.ylim(y_min, y_max)
			plt.xlabel('x')
			plt.ylabel('f(x)')
			plt.legend()
			plt.show()
