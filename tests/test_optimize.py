# Integration tests for root_finding suite
# Harvard University, CS 207, Fall 2018
# Authors: Chen Shi, Stephen Slater, Yue Sun

import sys
sys.path.append('../')

import DeriveAlive.optimize as opt
import DeriveAlive.DeriveAlive as da
import numpy as np

# Comment out for testing on Travis CI
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.animation as animation

def _assert_decreasing(nums):
	for i in range(1, len(nums)):
		assert nums[i] <= nums[i - 1]

def plot_results(f, x_path, f_path, f_string, x_lims=None, y_lims=None, num_points=100, 
				 threedim=False, fourdim=False, animate=False, speed=1):
	if x_lims:
		x_min, x_max = x_lims
		x_range = np.linspace(x_min, x_max, num_points)
	else:
		x_range = np.linspace(-10, 10, num_points)

	if y_lims:
		y_min, y_max = y_lims
	
	# Plot function and path of points
	if threedim:
		xs, ys = x_path[:, 0], x_path[:, 1]
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
		
		if animate and len(x_path) > 2:
			line, = ax.plot(xs, ys, fs, 'b-o', label='path')

			def update(num, x, y, z, line):
			    line.set_data(x[:num], y[:num])
			    line.set_3d_properties(z[:num])
			    if x_lims and y_lims:
			    	line.axes.axis([x_min, x_max, y_min, y_max])
			    return line,

			ani = animation.FuncAnimation(fig, update, len(x_path), fargs=[xs, ys, f_path, line],
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
		xs, ys, zs = x_path[:, 0], x_path[:, 1], x_path[:, 2]
		fs = f_path
		xn, yn, zn = np.round(x_path[-1], 4)
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
		x0, f0 = np.round(x_path[0], 4), np.round(f_path[0], 4)
		xn, fn = np.round(x_path[-1], 4), np.round(f_path[-1], 4)

		if animate and len(x_path) > 2:
			fig, ax = plt.subplots()
			ax.set_title(r'Finding minimum of ${}$'.format(f_string))
			ax.plot(x_range, f(x_range), 'k-', label=r'${}$'.format(f_string))
			ax.plot(x0, f0, 'ro', label='start: {}'.format(x0))
			ax.plot(xn, fn, 'go', label='end: {}'.format(xn))

			line, = ax.plot(x_path, f_path, 'b-o', label='path')

			def update(num, x, y, line):
			    line.set_data(x[:num], y[:num])
			    if x_lims and y_lims:
			    	line.axes.axis([x_min, x_max, y_min, y_max])
			    return line,

			ani = animation.FuncAnimation(fig, update, len(x_path), fargs=[x_path, f_path, line],
			                              interval=200, blit=True, repeat_delay=500, repeat=True)

			ax.legend()
			ax.set_xlabel('x')
			ax.set_ylabel('f(x)')
			plt.show()
		else:
			plt.figure()
			plt.title(r'Finding minimum of ${}$'.format(f_string))
			plt.plot(x_range, f(x_range), 'k-', label=r'${}$'.format(f_string))
			plt.plot(x_path, f_path, 'b-o', label='path')
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

def test_GradientDescent():
	'''Test gradient descent optimization method to find local minima.'''

	def case_1():
		'''Simple quadratic function with minimum at x = 0.'''

		def f(x):
			return x ** 2

		f_string = 'f(x) = x^2'

		for x_val in [-4, -2, 0, 2, 4]:
			x0 = da.Var(x_val)
			solution, x_path, f_path = opt.GradientDescent(f, x0)
			# plot_results(f, x_path, f_path, f_string)

			assert np.allclose(solution.val, [0])
			assert np.allclose(solution.der, [0])

	def case_2():

		def f(x):
			return (x - 1) ** 2 + 3

		f_string = 'f(x) = (x - 1)^2 + 3'

		for x_val in [-3, -1, 1, 3, 5]:
			# Test initial guess without using da.Var type
			solution, x_path, f_path = opt.GradientDescent(f, x_val)
			# plot_results(f, x_path, f_path, f_string)

			assert np.allclose(solution.val, [1])
			assert np.allclose(solution.der, [0])


	def case_3():

		def f(x):
			return np.sin(x)

		f_string = 'f(x) = sin(x)'

		for x_val in [-3, -1, 1, 3]:
			x0 = da.Var(x_val)
			solution, x_path, f_path = opt.GradientDescent(f, x0)
			# plot_results(f, x_path, f_path, f_string)

			multiple_of_three_halves_pi = (solution.val  % (2 * np.pi))
			np.testing.assert_array_almost_equal(multiple_of_three_halves_pi, 1.5 * np.pi)				
			np.testing.assert_array_almost_equal(solution.der, [0])


	def case_4():
		'''Minimize Rosenbrock function: f(x, y) = 100(y - x^2)^2 + (1 - x)^2.

		The global minimum is 0 when (x, y) = (1, 1).
		'''

		def f(variables):
			x, y = variables
			return 4 * (y - (x ** 2)) ** 2 + (1 - x) ** 2

		f_string = 'f(x, y) = 4(y - x^2)^2 + (1 - x)^2'

		for x_val, y_val in [[-6., -6], [2., 3.], [-2., 5.]]:
			x0 = da.Var(x_val, [1, 0])
			y0 = da.Var(y_val, [0, 1])
			init_vars = [x0, y0]
			solution, xy_path, f_path = opt.GradientDescent(f, init_vars, iters=25000, eta=0.002)
			xn, yn = solution.val
			# plot_results(f, xy_path, f_path, f_string, x_lims=(-7.5, 7.5), threedim=True)

			minimum = [1, 1]
			assert np.allclose([xn, yn], minimum)

			# dfdx = 2(200x^3 - 200xy + x - 1)
			# dfdy = 200(y - x^2)
			der_x = 2 * (8 * (xn ** 3) - 8 * xn * yn + xn - 1)
			der_y = 8 * (yn - (xn ** 2))
			assert np.allclose(solution.der, [der_x, der_y])

	def case_5():
		'''Minimize bowl function: f(x, y) = x^2 + y^2 + 2.

		The global minimum is 2 when (x, y) = (0, 0).
		'''

		def f(variables):
			x, y = variables
			return x ** 2 + y ** 2 + 2

		f_string = 'f(x, y) = x^2 + y^2 + 2'

		for x_val, y_val in [[2., 3.], [-2., 5.]]:
			x0 = da.Var(x_val, [1, 0])
			y0 = da.Var(y_val, [0, 1])
			init_vars = [x0, y0]
			solution, xy_path, f_path = opt.GradientDescent(f, init_vars)
			xn, yn = solution.val
			# plot_results(f, xy_path, f_path, f_string, x_lims=(-7.5, 7.5), threedim=True)

			minimum = [0, 0]
			assert np.allclose([xn, yn], minimum)

			der = [0, 0]
			assert np.allclose(solution.der, der)
	
	def case_6():
		'''Minimize cost function of ML regression demo.'''

		def generate_loss_function(num_features, dataset):
			sizes = [] # living area
			bedrooms = [] # number of bedrooms
			prices = [] # price

			with open(dataset, "r") as data_file:
				houses = data_file.readlines()
				
				for house in houses:
					nums = house.split(",")
					sizes.append(float(nums[0]))
					bedrooms.append(float(nums[1]))
					prices.append(float(nums[2]) / 1000)

				sizes = np.array(sizes)
				bedrooms = np.array(bedrooms)
				prices = np.array(prices)

			# Number of data points
			m = len(sizes)

			# Mean squared error
			def f(variables):
				w0, w1, w2 = variables
				array = [(w0 + w1 * sizes[i] + w2 * bedrooms[i] - prices[i]) ** 2 for i in range(m)]
				return (1 / (2 * m)) * sum(array)

			return f 

		f = generate_loss_function(2, "normalized.txt")

		f_string = 'f(w_0, w_1, w_2) = (1/2m)\sum_{i=0}^m (w_0 + w_1x_{i1} + w_2x_{i2} - y_i)^2'

		for w0_val, w1_val, w2_val in [[0., 0., 0.]]:
			# Test initial guess without using da.Var type
			init_vars = [w1_val, w1_val, w2_val]
			m = len(init_vars)
			solution, w_path, f_path = opt.GradientDescent(f, init_vars, iters=2000, eta=.01)
			w0n, w1n, w2n = solution.val
			# plot_results(f, w_path, f_path, f_string, x_lims=(-7.5, 7.5), fourdim=True)

			# Ensure that loss function is weakly decreasing
			_assert_decreasing(f_path)

	def case_7():
		'''Minimize Easom's function.'''

		def f(variables):
			x, y = variables
			return -np.cos(x) * np.cos(y) * np.exp(-((x - np.pi) ** 2 + (y - np.pi) ** 2))

		f_string = 'f(x, y) = -\cos(x)\cos(y)\exp(-((x-\pi)^2 + (y-\pi)^2))'

		for x_val, y_val in [[1.5, 1.75]]:
			x0 = da.Var(x_val, [1, 0])
			y0 = da.Var(y_val, [0, 1])
			init_vars = [x0, y0]
			solution, xy_path, f_path = opt.GradientDescent(f, init_vars, iters=10000, eta=0.3)
			# plot_results(f, xy_path, f_path, f_string, threedim=True)

			# Ensure that loss function is weakly decreasing
			_assert_decreasing(f_path)

	def case_8():
		'''Minimize Himmelblau's function.'''

		def f(variables):
			x, y = variables
			return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

		f_string = 'f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2'

		for x_val, y_val in [[5, 5]]:
			x0 = da.Var(x_val, [1, 0])
			y0 = da.Var(y_val, [0, 1])
			init_vars = [x0, y0]
			solution, xy_path, f_path = opt.GradientDescent(f, init_vars, iters=10000, eta=0.01)
			# plot_results(f, xy_path, f_path, f_string, threedim=True)

			# Ensure that loss function is weakly decreasing
			_assert_decreasing(f_path)

	case_1()
	case_2()
	case_3()
	case_4()
	case_5()
	case_6()
	case_7()
	case_8()
	print ("All gradient descent tests passed!")

print ("Testing optimization suite.")
test_GradientDescent()
