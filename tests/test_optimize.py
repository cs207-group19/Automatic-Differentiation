# Integration tests for root_finding suite
# Harvard University, CS 207, Fall 2018
# Authors: Chen Shi, Stephen Slater, Yue Sun

import sys
sys.path.append('../')

import DeriveAlive.optimize as opt
import DeriveAlive.DeriveAlive as da
import numpy as np
import matplotlib.pyplot as plt


def _assert_decreasing(nums, eps=0.01):
	''' Ensure that loss is monotonically decreasing, but allow for cases where
	the algorithm jumps over a minimum or oscillates within eps.
	'''
	
	for i in range(1, len(nums)):
		assert (nums[i] <= nums[i - 1] or nums[i] - eps <= nums[i - 1])


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
			opt.plot_results(f, x_path, f_path, f_string, hide=True)

			assert np.allclose(solution.val, [0])
			assert np.allclose(solution.der, [0])

		plt.close('all')

	def case_2():

		def f(x):
			return (x - 1) ** 2 + 3

		f_string = 'f(x) = (x - 1)^2 + 3'

		for x_val in [-3, -1, 1, 3, 5]:
			# Test initial guess without using da.Var type
			solution, x_path, f_path = opt.GradientDescent(f, x_val)
			opt.plot_results(f, x_path, f_path, f_string, x_lims=(-3, 5), y_lims=(2, 10), animate=True, hide=True)

			assert np.allclose(solution.val, [1])
			assert np.allclose(solution.der, [0])

		plt.close('all')		


	def case_3():

		def f(x):
			return np.sin(x)

		f_string = 'f(x) = sin(x)'

		for x_val in [-3, -1, 1, 3]:
			# Test case when 1D input is a list
			x0 = [da.Var(x_val)]
			solution, x_path, f_path = opt.GradientDescent(f, x0)
			opt.plot_results(f, x_path, f_path, f_string, x_lims=(-2 * np.pi, 2 * np.pi), y_lims=(-1.5, 1.5), hide=True)

			multiple_of_three_halves_pi = (solution.val  % (2 * np.pi))
			np.testing.assert_array_almost_equal(multiple_of_three_halves_pi, 1.5 * np.pi)				
			np.testing.assert_array_almost_equal(solution.der, [0])

		plt.close('all')		

	def case_4():
		'''Minimize Rosenbrock function: f(x, y) = 4(y - x^2)^2 + (1 - x)^2.

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
			opt.plot_results(f, xy_path, f_path, f_string, x_lims=(-7.5, 7.5), threedim=True, hide=True)

			minimum = [1, 1]
			assert np.allclose([xn, yn], minimum)

			# dfdx = 2(200x^3 - 200xy + x - 1)
			# dfdy = 200(y - x^2)
			der_x = 2 * (8 * (xn ** 3) - 8 * xn * yn + xn - 1)
			der_y = 8 * (yn - (xn ** 2))
			assert np.allclose(solution.der, [der_x, der_y])

		plt.close('all')

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
			opt.plot_results(f, xy_path, f_path, f_string, x_lims=(-7.5, 7.5), threedim=True, animate=True, hide=True)

			minimum = [0, 0]
			assert np.allclose([xn, yn], minimum)

			der = [0, 0]
			assert np.allclose(solution.der, der)

		plt.close('all')
	
	def case_6():

		'''Minimize cost function of ML regression demo.

		Note that we hardcode the dataset, since Travis CI does not recognize .txt files in the repository.
		'''

		# Standardized dataset: number of bedrooms, number of bathrooms, price
		dataset = [0.1300098691,-0.2236751872,399900.0000000000,
		-0.5041898382,-0.2236751872,329900.0000000000,
		0.5024763638,-0.2236751872,369000.0000000000,
		-0.7357230647,-1.5377669118,232000.0000000000,
		1.2574760154,1.0904165374,539900.0000000000,
		-0.0197317285,1.0904165374,299900.0000000000,
		-0.5872397999,-0.2236751872,314900.0000000000,
		-0.7218814044,-0.2236751872,198999.0000000000,
		-0.7810230438,-0.2236751872,212000.0000000000,
		-0.6375731100,-0.2236751872,242500.0000000000,
		-0.0763567023,1.0904165374,239999.0000000000,
		-0.0008567372,-0.2236751872,347000.0000000000,
		-0.1392733400,-0.2236751872,329999.0000000000,
		3.1172918237,2.4045082621,699900.0000000000,
		-0.9219563121,-0.2236751872,259900.0000000000,
		0.3766430886,1.0904165374,449900.0000000000,
		-0.8565230089,-1.5377669118,299900.0000000000,
		-0.9622229602,-0.2236751872,199900.0000000000,
		0.7654679091,1.0904165374,499998.0000000000,
		1.2964843307,1.0904165374,599000.0000000000,
		-0.2940482685,-0.2236751872,252900.0000000000,
		-0.1417900055,-1.5377669118,255000.0000000000,
		-0.4991565072,-0.2236751872,242900.0000000000,
		-0.0486733818,1.0904165374,259900.0000000000,
		2.3773921652,-0.2236751872,573900.0000000000,
		-1.1333562145,-0.2236751872,249900.0000000000,
		-0.6828730891,-0.2236751872,464500.0000000000,
		0.6610262907,-0.2236751872,469000.0000000000,
		0.2508098133,-0.2236751872,475000.0000000000,
		0.8007012262,-0.2236751872,299900.0000000000,
		-0.2034483104,-1.5377669118,349900.0000000000,
		-1.2591894898,-2.8518586364,169900.0000000000,
		0.0494765729,1.0904165374,314900.0000000000,
		1.4298676025,-0.2236751872,579900.0000000000,
		-0.2386816274,1.0904165374,285900.0000000000,
		-0.7092980769,-0.2236751872,249900.0000000000,
		-0.9584479619,-0.2236751872,229900.0000000000,
		0.1652431861,1.0904165374,345000.0000000000,
		2.7863503098,1.0904165374,549000.0000000000,
		0.2029931687,1.0904165374,287000.0000000000,
		-0.4236565421,-1.5377669118,368500.0000000000,
		0.2986264579,-0.2236751872,329900.0000000000,
		0.7126179335,1.0904165374,314000.0000000000,
		-1.0075229393,-0.2236751872,299000.0000000000,
		-1.4454227371,-1.5377669118,179900.0000000000,
		-0.1870899846,1.0904165374,299900.0000000000,
		-1.0037479410,-0.2236751872,239500.0000000000]

		dataset = np.reshape(dataset, [-1, 3])
		dataset[:, 2] = dataset[:, 2] / 1000.
		f_string = 'f(w_0, w_1, w_2) = (1/2m)\sum_{i=0}^m (w_0 + w_1x_{i1} + w_2x_{i2} - y_i)^2'

		solution, w_path, f_path, f = opt.GradientDescent("mse", [0, 0, 0], data=dataset)
		opt.plot_results(f, w_path, f_path, f_string, x_lims=(-7.5, 7.5), fourdim=True, animate=True, hide=True)
		_assert_decreasing(f_path)

		plt.close('all')

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
			opt.plot_results(f, xy_path, f_path, f_string, threedim=True, hide=True)

			# Ensure that loss function is weakly decreasing
			_assert_decreasing(f_path)

		plt.close('all')

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
			opt.plot_results(f, xy_path, f_path, f_string, threedim=True, hide=True)

			# Ensure that loss function is weakly decreasing
			_assert_decreasing(f_path)

		plt.close('all')

	case_1()
	case_2()
	case_3()
	case_4()
	case_5()
	case_6()
	case_7()
	case_8()
	print ("All gradient descent tests passed!")


def test_BFGS():
	'''Test gradient descent optimization method to find local minima.'''

	def case_1():
		'''Simple quadratic function with stationary point at x = 0.'''

		def f(x):
			return x ** 2

		f_string = 'f(x) = x^2'

		for x_val in [-4, -2, 0, 2, 4]:
			x0 = da.Var(x_val)
			solution, x_path, f_path = opt.BFGS(f, x0)
			opt.plot_results(f, x_path, f_path, f_string, bfgs=True, hide=True)

			assert np.allclose(solution.val, [0])
			assert np.allclose(solution.der, [0])

		plt.close('all')

	def case_2():
		'''Stationary point (minimum) at x = 1.'''

		def f(x):
			return (x - 1) ** 2 + 3

		f_string = 'f(x) = (x - 1)^2 + 3'

		for x_val in [-3, -1, 1, 3, 5]:
			# Test initial guess without using da.Var type
			solution, x_path, f_path = opt.BFGS(f, x_val)
			opt.plot_results(f, x_path, f_path, f_string, x_lims=(-3, 5), y_lims=(2, 10), animate=True, bfgs=True, hide=True)

			assert np.allclose(solution.val, [1])
			assert np.allclose(solution.der, [0])

		plt.close('all')


	def case_3():
		'''Find stationary point of f(x) = sin(x).'''

		def f(x):
			return np.sin(x)

		f_string = 'f(x) = sin(x)'

		for x_val in [-3, -1, 1, 3]:
			# Test case when 1D input is a list
			x0 = [da.Var(x_val)]
			solution, x_path, f_path = opt.BFGS(f, x0)
			opt.plot_results(f, x_path, f_path, f_string, x_lims=(-2 * np.pi, 2 * np.pi), y_lims=(-1.5, 1.5), bfgs=True, hide=True)

			# BFGS finds a stationary point, which are every pi offset from pi/2
			first = solution.val - (np.pi / 2)
			multiple_of_one_half_pi = ((abs(solution.val) - (np.pi / 2)) % (np.pi))
			np.testing.assert_array_almost_equal(multiple_of_one_half_pi, [0])				
			np.testing.assert_array_almost_equal(solution.der, [0])

		plt.close('all')


	def case_4():
		'''Find stationary point of Rosenbrock function: f(x, y) = 4(y - x^2)^2 + (1 - x)^2.

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
			solution, xy_path, f_path = opt.BFGS(f, init_vars, iters=25000)
			xn, yn = solution.val
			opt.plot_results(f, xy_path, f_path, f_string, x_lims=(-7.5, 7.5), y_lims=(-7.5, 7.5), threedim=True, animate=True, bfgs=True, hide=True)

			minimum = [1, 1]
			assert np.allclose([xn, yn], minimum)

			# dfdx = 2(200x^3 - 200xy + x - 1)
			# dfdy = 200(y - x^2)
			der_x = 2 * (8 * (xn ** 3) - 8 * xn * yn + xn - 1)
			der_y = 8 * (yn - (xn ** 2))
			assert np.allclose(solution.der, [der_x, der_y])

		plt.close('all')

	def case_5():
		'''Find stationary point of bowl function: f(x, y) = x^2 + y^2 + 2.

		The global minimum is 2 when (x, y) = (0, 0).
		'''

		def f(variables):
			x, y = variables
			return x ** 2 + y ** 2 + 2

		f_string = 'f(x, y) = x^2 + y^2 + 2'

		for x_val, y_val in [[2., 3.], [-2., 5.]]:
			x0 = x_val
			y0 = y_val
			init_vars = [x0, y0]
			solution, xy_path, f_path = opt.BFGS(f, init_vars)
			xn, yn = solution.val
			opt.plot_results(f, xy_path, f_path, f_string, x_lims=(-7.5, 7.5), y_lims=(-7.5, 7.5), threedim=True, animate=True, bfgs=True, hide=True)

			minimum = [0, 0]
			assert np.allclose([xn, yn], minimum)

			der = [0, 0]
			assert np.allclose(solution.der, der)

		plt.close('all')
	
	def case_6():
		'''Find stationary point of Easom's function. Stationary points include (pi, pi), (pi/2, pi/2).'''

		def f(variables):
			x, y = variables
			return -np.cos(x) * np.cos(y) * np.exp(-((x - np.pi) ** 2 + (y - np.pi) ** 2))

		f_string = 'f(x, y) = -\cos(x)\cos(y)\exp(-((x-\pi)^2 + (y-\pi)^2))'

		x0 = 2.5
		y0 = 2.5
		init_vars = [x0, y0]
		solution, xy_path, f_path = opt.BFGS(f, init_vars, iters=10000)
		opt.plot_results(f, xy_path, f_path, f_string, threedim=True, bfgs=True, hide=True)

		xn, yn = solution.val
		assert (np.allclose(xn, np.pi) and np.allclose(yn, np.pi))

		plt.close('all')

	def case_7():
		'''Find stationary point of Himmelblau's function.'''

		def f(variables):
			x, y = variables
			return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

		f_string = 'f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2'

		x0 = 5
		y0 = 5
		init_vars = [x0, y0]
		solution, xy_path, f_path = opt.BFGS(f, init_vars, iters=10000)
		opt.plot_results(f, xy_path, f_path, f_string, x_lims=(-10, 10), y_lims=(-10, 10), threedim=True, animate=True, bfgs=True, hide=True)

		xn, yn = solution.val
		assert ((np.allclose(xn, 3) and np.allclose(yn, 2)) or (
				 np.allclose(xn, -2.805118) and np.allclose(yn, 3.131312)) or (
				 np.allclose(xn, -3.779310) and np.allclose(yn, -3.283186)) or (
				 np.allclose(xn, 3.584428) and np.allclose(yn, -1.848126)))

		plt.close('all')

	case_1()
	case_2()
	case_3()
	case_4()
	case_5()
	case_6()
	case_7()
	print ("All BFGS tests passed!")


print ("Testing optimization suite.")
test_GradientDescent()
test_BFGS()
