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


def _GradientDescentVector(f, var_list, iters=10000, tol=1e-10, eta=0.01, valid_data=False):
	'''Internal function that may get called from the API GradientDescent method.

	See documentation of optimize.GradientDescent.
	'''

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
	return (minimum, var_path, g_path, f) if valid_data else (minimum, var_path, g_path)


def GradientDescent(f, x, iters=10000, tol=1e-10, eta=0.01, data=[]):
	""" Run the gradient descent algorithm to minimize a function f.
		
		Parameters
		----------
		f: callable (function), or string
		   if :math:`f` is a vector to scalar function, it must take as argument a single list
		   of variables (of type int, float, or ``DeriveAlive.Var``). If :math:`f` is a scalar
		   to scalar function, the input can be of type int, float, or ``DeriveAlive.Var``, but 
		   should not be a list.
		   If :math:`f` is a string, it must be "mse", in which case the user must also specify
		   a dataset for which the method will optimize according to the mean squared error.
		
		x: int, float, or ``DeriveAlive.Var`` object
		   initial guess/starting point for minimum

		iters: integer, optional (iters=10000)
			   maximum number of iterations to run the algorithm.
		
		tol: float, optional (default=1e-10)
			 this denotes the stopping criterion/tolerance for the algorithm. 
			 The algorithm terminates if the next absolute step size is less than :math:`tol`. 
			 In the multivariate case, the step size is determined by taking the L2 norm.
		
		eta: float, optional (default=0.01)
			 learning rate for gradient descent algorithm.

		data: numpy.ndarray (m x n), optional (default=[])
			  dataset on which to run the gradient descent with mean squared error objective function.
			  There must be at least two columns (one for X features, one for y outputs), and the 
			  features should be standardized. The final column should contain the outputs y. There
			  are m datapoints and n - 1 features. Do not include a bias term column. 
			  The algorithm will solve for n optimal weights.
			
		Returns
		-------
		minimum: ``DeriveAlive.Var``
				 contains the optimal scalar or vector solution and the derivative at that point.
		
		var_path: numpy.ndarray (min(iters, t), m)
				  contains the path of the input variable(s) throughout the min(iters, t) steps,
				  where iters is defined above and t is the number of steps needed to satisfy tol.
		
		g_path: numpy.ndarray (min(iters, t),)
				contains the path of the objective function :math:`f` throughout the min(iters, t) steps.

		f: callable (function)
		   this value is returned if the user passed in a valid dataset and specified the "mse" objective,
		   referring to mean squared error.
		
		"""

	# Handle dataset input
	valid_data = False
	if isinstance(f, str):
		assert f == 'mse' and len(data) > 0 and len(data[0] >= 2)
		valid_data = True
		num_points = len(data)
		num_features = len(data[0]) - 1
		num_weights = num_features + 1

		# User must have data in at least one dimensions, which requires 2 weights (one for bias)
		assert num_features == len(x) - 1 if isinstance(x, list) else 2

		# Initialize weights to 0
		weights = [da.Var(x[i], _get_unit_vec(num_weights, i)) for i in range(num_weights)]

		y = data[:, -1]
		X = data[:, :-1]

		def f(weights):
			return (1 / (2 * num_points)) * sum([(weights[0] + np.dot(weights[1:], X[i]) - y[i]) ** 2 for i in range(num_points)])

	if isinstance(x, list):
		# Number of variables
		m = len(x)

		# Convert to da.Var type
		if m > 1:
			for i in range(len(x)):
				if not isinstance(x[i], da.Var):
					x[i] = da.Var(x[i], _get_unit_vec(m, i))
			return _GradientDescentVector(f, x, iters=iters, tol=tol, eta=eta, valid_data=valid_data)
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
	return (minimum, var_path, g_path, f) if valid_data else (minimum, var_path, g_path)


def plot_results(f, var_path, f_path, f_string, x_lims=None, y_lims=None, num_points=100, 
				 threedim=False, fourdim=False, animate=False, speed=1):
	""" Plot the results of GradientDescent. Use the return values (var_path, f_path) from
	    GradientDescent when plotting your results to show the steps of the algorithm.
		
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
				  a string denoting the equation to be minimized. Supports LaTeX code,
				  i.e. f_string = 'f(x, y) = x^2 + y^2' will look nice in title of the plot.
		
		x_lims: length-2 iterable, optional (default=None)
			 	this denotes the lower and upperbound values to be plotted on the x-axis. If not
			 	specified, the function will plot over the range (-10, 10).

		y_lims: length-2 iterable, optional (default=None)
			 	this denotes the lower and upperbound values to be plotted on the y-axis. If not
			 	specified, the function will plot over the range (-10, 10).	

		num_points: int, optional (default=100)
			 	    this denotes the number of points to plot on the x-axis and y-axis intervals, 
			 	    respectively.		 
		
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
		ax.set_title(r'Finding minimum of ${}$'.format(f_string))

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
		xn, yn, zn = np.round(var_path[-1], 4)
		x0, y0, z0 = np.round(xs[0], 4), np.round(ys[0], 4), np.round(zs[0], 4)
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
		ax.set_title(r'Minimizing ${}$'.format(f_string))

		# Place legend
		ax.legend(loc='upper left', bbox_to_anchor=(0, 0.85))
		plt.show()

		# Plot objective function
		plt.figure()
		plt.plot(f_path, label='Loss')
		plt.title('Loss function vs. number of iterations')
		plt.xlabel('Iterations')
		plt.ylabel('Loss function')
		plt.show()

	else:
		x0, f0 = np.round(var_path[0], 4), np.round(f_path[0], 4)
		xn, fn = np.round(var_path[-1], 4), np.round(f_path[-1], 4)

		if animate and len(var_path) > 2:
			fig, ax = plt.subplots()
			ax.set_title(r'Finding minimum of ${}$'.format(f_string))
			ax.plot(x_range, f(x_range), 'k-', label=r'${}$'.format(f_string))
			ax.plot(x0, f0, 'ro', label='start: {}'.format(x0))
			ax.plot(xn, fn, 'go', label='end: {}'.format(xn))

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
			plt.title(r'Finding minimum of ${}$'.format(f_string))
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
