# Integration tests for root_finding suite
# Harvard University, CS 207, Fall 2018
# Authors: Chen Shi, Stephen Slater, Yue Sun

import sys
sys.path.append('../')

import DeriveAlive.root_finding as rf
import DeriveAlive.DeriveAlive as da
import numpy as np

# Comment out for testing on Travis CI
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def plot_results(f, x_path, f_path, f_string, x_lims=None, y_lims=None, num_points=50, 
				 threedim=False, fourdim=False, gd=False):
	if x_lims:
		x_min, x_max = x_lims
		x_range = np.linspace(x_min, x_max, num_points)
	else:
		x_range = np.linspace(-10, 10, num_points)

	if y_lims:
		y_min, y_max = y_lims
	
	x0, f0 = x_path[0], f_path[0]
	xn, fn = np.round(x_path[-1], 4), f_path[-1]

	# Plot function and path of points
	
	if threedim:
		# 3D plot
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		xs = np.array([x for x, _ in x_path])
		ys = np.array([y for _, y in x_path])
		fs = np.array([z[0] for z in f_path])
		xn, yn = np.round(xs[-1], 4), np.round(ys[-1], 4)

		# For Gradient Descent, plot surface
		if gd:
			X, Y = np.meshgrid(x_range, x_range)
			zs = np.array([f([x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))])
			Z = zs.reshape(X.shape)
			# plt.contour(X, Y, Z)
			ax.plot_surface(X, Y, Z, color='0.5', alpha=0.5)
			ax.set_title('Finding minimum of {}'.format(f_string))
		else:
			ax.set_title('Finding root(s) of {}'.format(f_string))

		ax.plot(xs, ys, fs, 'b-o', label='path')
		ax.scatter(xs[0], ys[0], fs[0], c='red', marker='o', linewidth=6, label='start: {}'.format((xs[0], ys[0])))
		ax.scatter(xs[-1], ys[-1], fs[-1], c='green', marker='o', linewidth=6, label='end: {}'.format((xn, yn)))
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_zlabel('f(x, y)')
		ax.legend(loc=6)
		plt.show()

	elif fourdim:
		# 4D plot
		cm_type = cm.RdYlGn_r  # cm.PiYG  # m.tab20c  # cm.twilight  # cm.nipy_spectral
		fig = plt.figure(figsize=(10, 6))
		ax = fig.add_subplot(111, projection='3d')
		xs, ys, zs = x_path[:, 0], x_path[:, 1], x_path[:, 2]
		fs = f_path
		xn, yn, zn = np.round(x_path[-1], 4)
		ax.scatter(xs, ys, zs, c=fs, cmap=cm_type)
		ax.plot3D(xs, ys, zs, '-')
		ax.scatter(xs[0], ys[0], zs[0], c='red', marker='o', linewidth=6, label='start: {}'.format((xs[0], ys[0], zs[0])))
		ax.scatter(xs[-1], ys[-1], zs[-1], c='green', marker='o', linewidth=6, label='end: {}'.format((xn, yn, zn)))

		# ax.scatter(x, y, z, c=c, cmap=cm.nipy_spectral)  # cm.PiYG_r)  # plt.hot())
		m = cm.ScalarMappable(cmap=cm_type)  # PiYG_r)
		m.set_array(fs)
		cbar = plt.colorbar(m)
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_zlabel('z')
		ax.legend(loc=(0,0))
		ax.set_title('Finding root(s) of {}'.format(f_string))
		plt.show()

	else:
		plt.figure()
		if gd:
			plt.title('Finding minimum of {}'.format(f_string))
		else:
			plt.plot(x_range, np.zeros(num_points), '0.7')
			plt.title('Finding root(s) of {}'.format(f_string))

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

def test_NewtonRoot_r1_to_r1():
	'''Integration tests for scalar to scalar functions.'''

	def case_1():
		'''Find root of quadratic function that is also a global minimum.'''
		def f(x):
			return x ** 2

		x_lims = -4, 4
		y_lims = -4, 4
		f_string = 'f(x) = x^2'

		for val in np.arange(-2, 2.1, 1):
			x0 = [da.Var(val)]
			solution, x_path, y_path = rf.NewtonRoot(f, x0)

			# plot_results(f, x_path, y_path, f_string, x_lims, y_lims)

			root = [0]
			der = [0]
			np.testing.assert_array_almost_equal(solution.val, root, decimal=4)
			np.testing.assert_array_almost_equal(solution.der, der, decimal=4)

	def case_2():
		'''Find root of concave up quadratic function.'''
		def f(x):
			return (x - 1) ** 2 - 1

		x_lims = -4, 6
		y_lims = -2, 8
		f_string = 'f(x) = (x - 1)^2 - 1'

		for val in range(-1, 4):
			x0 = [da.Var(val)]
			solution, x_path, y_path = rf.NewtonRoot(f, x0)

			# plot_results(f, x_path, y_path, f_string, x_lims, y_lims)

			root_1, der_1 = [0], [-2]
			root_2, der_2 = [2], [2]
			assert ((np.allclose(solution.val, root_1) and np.allclose(solution.der, der_1)) or 
					(np.allclose(solution.val, root_2) and np.allclose(solution.der, der_2)))			

	def case_3():
		'''Find roots of concave down quadratic function.'''
		def f(x):
			return -((x + 3) ** 2) + 2

		x_lims = -8, 2
		y_lims = -6, 6
		f_string = 'f(x) = -(x+3)^2 + 2'

		for val in range(-7, 2, 2):
			x0 = [da.Var(val)]
			solution, x_path, y_path = rf.NewtonRoot(f, x0)

			# plot_results(f, x_path, y_path, f_string, x_lims, y_lims)

			root_1 = [-3 + np.sqrt(2)]
			root_2 = [-3 - np.sqrt(2)]
			assert (np.allclose(solution.val, root_1) or np.allclose(solution.val, root_2))
			
	def case_4():
		'''Find roots of cubic function.'''
		def f(x):
			return (x - 4) ** 3 - 3

		x_lims = 0, 8
		y_lims = -12, 12
		f_string = 'f(x) = (x - 4)^3 - 3'

		for val in range(2, 7):
			x0 = [da.Var(val)]
			solution, x_path, y_path = rf.NewtonRoot(f, x0)

			# plot_results(f, x_path, y_path, f_string, x_lims, y_lims)

			root = [4 + np.cbrt(3)]
			np.testing.assert_array_almost_equal(solution.val, root)	

	def case_5():
		'''Find roots of sinusoidal wave.'''
		def f(x):
			return np.sin(x)

		x_lims = -2 * np.pi, 3 * np.pi
		y_lims = -2, 2
		f_string = 'f(x) = sin(x)'

		for val in [np.pi - 0.25, np.pi, 1.5 * np.pi, 2 * np.pi - 0.25, 2 * np.pi + 0.25]:
			x0 = [da.Var(val)]
			solution, x_path, y_path = rf.NewtonRoot(f, x0)

			# plot_results(f, x_path, y_path, f_string, x_lims, y_lims)	

			root_multiple_of_pi = (solution.val / np.pi) % 1
			np.testing.assert_array_almost_equal(root_multiple_of_pi, [0.])	

	def case_6():
		'''Find roots of complicated scalar function.'''
		x_var = da.Var(0.1)
		def f(x):
			return x - np.exp(-2.0 * np.sin(4.0 * x) * np.sin(4.0 * x)) + 0.3

		x_lims = -2, 2
		y_lims = -2, 2
		f_string = 'f(x) = x - e^{-2 * sin(4x) * sin(4x)} + 0.3'

		for val in np.arange(-0.75, 0.8, 0.25):
			x0 = [da.Var(val)]
			solution, x_path, y_path = rf.NewtonRoot(f, x0)

			# plot_results(f, x_path, y_path, f_string, x_lims, y_lims)	

			root = [0.166402]
			der = [4.62465]
			np.testing.assert_array_almost_equal(solution.val, root, decimal=4)
			np.testing.assert_array_almost_equal(solution.der, der, decimal=4)		


	case_1()
	case_2()
	case_3()
	case_4()
	case_5()
	case_6()
	print ("All Newton root scalar to scalar tests passed!")


def test_NewtonRoot_rm_to_r1():
	'''Integration tests for Newton root finding for functions of vectors to scalars.'''

	def case_1():
		'''Find a root of f(x, y) = x + y, i.e x = -y'''
		def f(variables):
			x, y = variables
			return x + y

		f_string = 'f(x, y) = x + y'

		for x_val, y_val in [[-4, 3], [8, 1]]:
			x0 = da.Var(x_val, [1, 0])
			y0 = da.Var(y_val, [0, 1])
			init_vars = [x0, y0]
			solution, xy_path, f_path = rf.NewtonRoot(f, init_vars)
			xn, yn = solution.val

			# plot_results(f, xy_path, f_path, f_string, threedim=True)	

			# root: x = -y
			der = [1, 1]
			np.testing.assert_array_almost_equal(xn, -yn)
			np.testing.assert_array_almost_equal(solution.der, der)

	def case_2():
		'''Find a root of z(x, y) = x^2 - y^2, i.e. x = +-y.'''

		def f(variables):
			x, y = variables
			return x ** 2 - y ** 2

		f_string = 'f(x, y) = x^2 - y^2'

		for x_val, y_val in [[-4, 2], [12, -1]]:
			x0 = da.Var(x_val, [1, 0])
			y0 = da.Var(y_val, [0, 1])
			init_vars = [x0, y0]
			solution, xy_path, f_path = rf.NewtonRoot(f, init_vars)
			xn, yn = solution.val

			# plot_results(f, xy_path, f_path, f_string, threedim=True)	

			# root: x = +-y
			der = [2 * xn, -2 * yn]
			assert (np.allclose(xn, yn) or np.allclose(xn, -yn))
			assert np.allclose(solution.der, der)

	def case_3():
		'''Find global mininum and root at (0, 0) for f(x, y) = x^2 + y^2.'''

		def f(variables):
			x, y = variables
			return x ** 2 + y ** 2

		f_string = 'f(x, y) = x^2 + y^2'

		for x_val, y_val in [[2, 5], [-1, 3]]:
			x0 = da.Var(x_val, [1, 0])
			y0 = da.Var(y_val, [0, 1])
			init_vars = [x0, y0]
			solution, xy_path, f_path = rf.NewtonRoot(f, init_vars)
			xn, yn = solution.val

			# plot_results(f, xy_path, f_path, f_string, threedim=True)	

			# root: x = y = 0
			der = [0, 0]
			assert (np.allclose(xn, 0) and np.allclose(xn, 0))
			assert np.allclose(solution.der, der)

	def case_4():
		'''Complicated function.'''

		def f(variables):
			x, y = variables
			return x ** 2 + 4 * y ** 2 - 2 * (x ** 2) * y + 4

		f_string = 'f(x, y) = x^2 + 4y^2 -2x^2y + 4'

		for x_val, y_val in [[2, 12], [-5, -4]]:
			x0 = da.Var(x_val, [1, 0])
			y0 = da.Var(y_val, [0, 1])
			init_vars = [x0, y0]
			solution, xy_path, f_path = rf.NewtonRoot(f, init_vars)
			xn, yn = solution.val

			# plot_results(f, xy_path, f_path, f_string, threedim=True)	
			
			# root: x = +- 2(sqrt(y^2 + 1))/sqrt(2y - 1)
			value = 2 * np.sqrt(yn ** 2 + 1) / (np.sqrt(2 * yn - 1))
			assert (np.allclose(xn, value) or np.allclose(xn, -value))

			# dfdx = x(2 - 4y)
			# dfdy = 8y - 2x^2
			der_x = xn * (2 - 4 * yn)
			der_y = 8 * yn - (2 * (xn ** 2))
			assert np.allclose(solution.der, [der_x, der_y])
		
	def case_5():
		
		def f(variables):
			x, y = variables
			return (x - 1) ** 2 + (y + 1) ** 2

		f_string = 'f(x, y) = (x - 1)^2 + (y + 1)^2'

		for x_val, y_val in [[2, 10], [-5, -4]]:
			x0 = da.Var(x_val, [1, 0])
			y0 = da.Var(y_val, [0, 1])
			init_vars = [x0, y0]
			solution, xy_path, f_path = rf.NewtonRoot(f, init_vars)
			xn, yn = solution.val

			# plot_results(f, xy_path, f_path, f_string, threedim=True)	

			# root: x = 1 +- sqrt(-(y + 1)^2)
			inner = -(yn + 1) ** 2
			inner = 0 if abs(inner) < 1e-6 else inner
			value = np.sqrt(inner)
			root_1 = 1 + value
			root_2 = 1 - value
			assert (np.allclose(xn, root_1) or np.allclose(xn, root_2))

			# dfdx = 2(x - 1)
			# dfdy = 2(y + 1)
			der = [2 * (xn - 1), 2 * (yn + 1)]
			assert np.allclose(solution.der, der)				

	def case_6():
		'''Find the roots of a function from R^3 to R^1.'''

		def f(variables):
			x, y, z = variables
			return x ** 2 + y ** 2 + z ** 2

		f_string = 'f(x, y, z) = x^2 + y^2 + z^2'

		for x_val, y_val, z_val in [[1, -2, 5], [20, 15, -5]]:
			x0 = da.Var(x_val, [1, 0, 0])
			y0 = da.Var(y_val, [0, 1, 0])
			z0 = da.Var(z_val, [0, 0, 1])
			init_vars = [x0, y0, z0]
			solution, xyz_path, f_path = rf.NewtonRoot(f, init_vars)
			m = len(solution.val)
			xn, yn, zn = solution.val

			concat = np.reshape(np.concatenate(xyz_path), [-1, m])
			f_path = np.concatenate(f_path)
			# plot_results(f, concat, f_path, f_string, fourdim=True)

			root = [0, 0, 0]
			assert np.allclose(solution.val, root)

			# dfdx = 2x
			# dfdy = 2y
			# dfdz = 2z
			der = [2 * xn, 2 * yn, 2 * zn]
			assert np.allclose(solution.der, der)	

	def case_7():
		'''Find the roots of a more complicated function from R^3 to R^1.'''

		def f(variables):
			x, y, z = variables
			# return np.exp(x + (y + 1.5) ** 2 + (z - 2) ** 2 - 3)
			return (x + y + z) ** 2 - 3

		# f_string = 'f(x, y, z) = exp(x + (y + 1.5)^2 + (z - 2)^2 - 3)'
		f_string = 'f(x, y, z) = (x + y + z)^2 - 3'

		for x_val, y_val, z_val in [[1, -2, 5], [20, 15, -5]]:
			# print ("starting from [{}, {}, {}]".format(x_val, y_val, z_val))
			x0 = da.Var(x_val, [1, 0, 0])
			y0 = da.Var(y_val, [0, 1, 0])
			z0 = da.Var(z_val, [0, 0, 1])
			init_vars = [x0, y0, z0]
			solution, xyz_path, f_path = rf.NewtonRoot(f, init_vars)
			m = len(solution.val)
			xn, yn, zn = solution.val
			# print ("Solution:\n{}".format(solution))
			# print ("Evaluated solution:\n{}".format(f([xn, yn, zn])))

			concat = np.reshape(np.concatenate(xyz_path), [-1, m])
			f_path = np.concatenate(f_path)
			# plot_results(f, concat, f_path, f_string, fourdim=True)

			# root: z = -y -z +- sqrt(3)
			root_1 = -yn - zn - np.sqrt(3)
			root_2 = -yn - zn + np.sqrt(3)
			assert (np.allclose(xn, root_1) or np.allclose(xn, root_2))

			# dfdx = 2(x + y + z)
			# dfdy = 2(x + y + z)
			# dfdz = 2(x + y + z)
			value = 2 * (xn + yn + zn)
			der = [value, value, value]
			assert np.allclose(solution.der, der)	

	def case_8():
		'''Find the roots of a more complicated function from R^3 to R^1.'''

		def f(variables):
			x, y, z = variables
			return x ** (y ** 2) - z ** 2

		f_string = 'f(x, y, z) = x^(y^2) - z^2'

		for x_val, y_val, z_val in [[1, -2, 5], [2, 4, -5]]:
			# print ("starting from [{}, {}, {}]".format(x_val, y_val, z_val))
			x0 = da.Var(x_val, [1, 0, 0])
			y0 = da.Var(y_val, [0, 1, 0])
			z0 = da.Var(z_val, [0, 0, 1])
			init_vars = [x0, y0, z0]
			solution, xyz_path, f_path = rf.NewtonRoot(f, init_vars)
			m = len(solution.val)
			xn, yn, zn = solution.val
			# print ("Solution:\n{}".format(solution))
			# print ("Evaluated solution:\n{}".format(f([xn, yn, zn])))

			concat = np.reshape(np.concatenate(xyz_path), [-1, m])
			f_path = np.concatenate(f_path)
			# plot_results(f, concat, f_path, f_string, fourdim=True)

			# root: x = +- (zn^2)^(1 / (yn^2))
			root_1 = (zn ** 2) ** (1 / (yn ** 2))
			root_2 = -root_1
			assert (np.allclose(xn, root_1) or np.allclose(xn, root_2))

			# dfdx = y^2 * x^{y^2 - 1}
			# dfdy = 2yx^{y^2} * log(x)
			# dfdz = -2z
			der = [yn ** 2 * (xn ** ((yn ** 2) - 1)), 
				   2 * yn * (xn ** (yn ** 2)) * np.log(xn), 
				   -2 * zn]

			# TODO: add check for derivative once __pow__ is updated.
			# assert np.allclose(solution.der, der)			

	case_1()
	case_2()
	case_3()
	case_4()
	case_5()
	case_6()
	case_7()
	# case_8()
	print ("All Newton root vector to scalar cases passed!")


def test_GradientDescent():
	'''Test gradient descent optimization method to find local minima.'''

	def case_1():
		'''Simple quadratic function with minimum at x = 0.'''

		def f(x):
			return x ** 2

		f_string = 'f(x) = x^2'

		for x_val in [-4, -2, 0, 2, 4]:
			x0 = da.Var(x_val)
			solution, x_path, f_path = rf.GradientDescent(f, x0)
			# print ("Solution:\n{}".format(solution))

			# plot_results(f, x_path, f_path, f_string, show_zero=False)

			assert np.allclose(solution.val, [0])
			assert np.allclose(solution.der, [0])

	def case_2():

		def f(x):
			return (x - 1) ** 2 + 3

		f_string = 'f(x) = (x - 1)^2 + 3'

		for x_val in [-3, -1, 1, 3, 5]:
			x0 = da.Var(x_val)
			solution, x_path, f_path = rf.GradientDescent(f, x0)
			# print ("Solution:\n{}".format(solution))

			# plot_results(f, x_path, f_path, f_string, gd=True)

			assert np.allclose(solution.val, [1])
			assert np.allclose(solution.der, [0])


	def case_3():

		def f(x):
			return np.sin(x)

		f_string = 'f(x) = sin(x)'

		for x_val in [-3, -1, 1, 3]:
			x0 = da.Var(x_val)
			solution, x_path, f_path = rf.GradientDescent(f, x0)
			# print ("Solution:\n{}".format(solution))

			# plot_results(f, x_path, f_path, f_string, gd=True)

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

		for x_val, y_val in [[2., 3.], [-2., 5.]]:
			x0 = da.Var(x_val, [1, 0])
			y0 = da.Var(y_val, [0, 1])
			init_vars = [x0, y0]
			solution, xy_path, f_path = rf.GradientDescent(f, init_vars)
			xn, yn = solution.val
			# print ("Solution:\n{}".format(solution))

			# plot_results(f, xy_path, f_path, f_string, x_lims=(-7.5, 7.5), threedim=True, gd=True)

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
			solution, xy_path, f_path = rf.GradientDescent(f, init_vars)
			xn, yn = solution.val
			# print ("Solution:\n{}".format(solution))

			# plot_results(f, xy_path, f_path, f_string, x_lims=(-7.5, 7.5), threedim=True, gd=True)

			minimum = [0, 0]
			assert np.allclose([xn, yn], minimum)

			der = [0, 0]
			assert np.allclose(solution.der, der)
	

	case_1()
	case_2()
	case_3()
	case_4()
	case_5()
	print ("All gradient descent tests passed!")


	# #def F(x):
	# #        return da.Var(
	# #            [x[0]**2 - x[1] + x[0]* np.cos(np.pi*x[0]),
	# #             x[0]*x[1] + np.exp(-x[1]) - x[0]**2])

	# #expected = np.array([1, 0])
	# #x1=da.Var([2],[1, 0])
	# #x2=da.Var([-1],[0, 1])
	# #x = da.Var([x1, x2])

	# def F(x1, x2):
	#         return da.Var(
	#             [x1 ** 2 - x2 + x1 * np.cos(np.pi*x1),
	#              x1 * x2 + np.exp(-x2) - x1 ** 2])
	# x1=da.Var([2],[1, 0])
	# x2=da.Var([-1],[0, 1])
	# x = da.Var([x1, x2])
	# output = NewtonRoot(F(x1, x2), x)

	# init_guess = da.Var([1, 2], [1, 1])
	# x = vec.val[0]
	# y = vec.val[1]
	# init_guess = [x, y]

	# def z(variables):
	# 	x, y = variables
	# 	# return da.Var(vec.val[0] + vec.val[1], [vec.der[0] + vec.der[1]])
	# 	return x + y

	# def z(vec):
	# 	return vec[0] + vec[1]
	# 	return x + y


# def test_r2_to_r2():
# 	init_guess = da.Var([1, 2], [1, 1])

# 	def z(vec):
# 		return vec ** 2

# 	print ("\n\nTRYING R2 -> R2 CASE. EXPECT ROOT AT (0, 0) FOR Z(X, Y) = [X**2, Y**2].\n")
# 	print (rf.NewtonRoot(z, init_guess))

print ("Testing root finding and optimization suite.")
test_NewtonRoot_r1_to_r1()
test_NewtonRoot_rm_to_r1()
# test_r2_to_r2()
test_GradientDescent()
print ("All tests passed!")

