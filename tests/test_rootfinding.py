# Integration tests for root_finding suite
# Harvard University, CS 207, Fall 2018
# Authors: Chen Shi, Stephen Slater, Yue Sun

import sys
sys.path.append('../')

import DeriveAlive.rootfinding as rf
import DeriveAlive.DeriveAlive as da
import numpy as np


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
			rf.plot_results(f, x_path, y_path, f_string, x_lims, y_lims)

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

			if val == 1:
				fig = rf.plot_results(f, x_path, y_path, f_string, x_lims, y_lims)

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
			rf.plot_results(f, x_path, y_path, f_string, x_lims, y_lims)

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
			rf.plot_results(f, x_path, y_path, f_string, x_lims, y_lims)

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
			rf.plot_results(f, x_path, y_path, f_string, x_lims, y_lims)	

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
			rf.plot_results(f, x_path, y_path, f_string, x_lims, y_lims)	

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
			rf.plot_results(f, xy_path, f_path, f_string, threedim=True)	

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
			rf.plot_results(f, xy_path, f_path, f_string, threedim=True)	

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
			rf.plot_results(f, xy_path, f_path, f_string, threedim=True)	

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

		for x_val, y_val in [[-8, -5], [2, 12], [-5, -4]]:
			# Test initial guess without using da.Var type
			init_vars = [x_val, y_val]
			solution, xy_path, f_path = rf.NewtonRoot(f, init_vars)
			xn, yn = solution.val
			rf.plot_results(f, xy_path, f_path, f_string, threedim=True, speed=25)	
			
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
			rf.plot_results(f, xy_path, f_path, f_string, threedim=True)	

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
			init_vars = [x_val, y_val, z_val]
			solution, xyz_path, f_path = rf.NewtonRoot(f, init_vars)
			m = len(solution.val)
			xn, yn, zn = solution.val
			rf.plot_results(f, xyz_path, f_path, f_string, fourdim=True)

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
			return (x + y + z) ** 2 - 3

		f_string = 'f(x, y, z) = (x + y + z)^2 - 3'

		for x_val, y_val, z_val in [[1, -2, 5], [20, 15, -5]]:
			init_vars = [x_val, y_val, z_val]
			solution, xyz_path, f_path = rf.NewtonRoot(f, init_vars)
			m = len(solution.val)
			xn, yn, zn = solution.val
			rf.plot_results(f, xyz_path, f_path, f_string, fourdim=True)

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
			x0 = da.Var(x_val, [1, 0, 0])
			y0 = da.Var(y_val, [0, 1, 0])
			z0 = da.Var(z_val, [0, 0, 1])
			init_vars = [x0, y0, z0]
			solution, xyz_path, f_path = rf.NewtonRoot(f, init_vars)
			m = len(solution.val)
			xn, yn, zn = solution.val
			rf.plot_results(f, xyz_path, f_path, f_string, fourdim=True)

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


print ("Testing root finding and optimization suite.")
test_NewtonRoot_r1_to_r1()
test_NewtonRoot_rm_to_r1()
