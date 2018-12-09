# Test suite for DeriveAlive module

# These lines should be included for Travis/Coverall
import sys
sys.path.append('../')

import DeriveAlive.DeriveAlive as da
import numpy as np
import math

def test_DeriveAlive_Var():
	'''Test constructor of Var class to ensure proper variable initializations.'''

	def test_scalar_without_bracket():
		x = da.Var(1)
		assert x.val == [1]
		assert type(x.val) == np.ndarray
		assert x.der == [1]

	def test_scalar_with_bracket():
		x = da.Var([1])
		assert x.val == [1]
		assert type(x.val) == np.ndarray
		assert x.der == [1]

	def test_vector_input():
		x = da.Var(2, [1, 0])
		y = da.Var(3, [0, 1])
		f = da.Var([x ** 2, y ** 2, x * y])

	def test_with_preset_der():
		der = 3.5
		x = da.Var(2, der)
		assert x.val == [2]
		assert x.der == [der]
		assert type(x.der) == np.ndarray

	def test_repr():
		x = da.Var(7.0)
		f = x.pow(2) + 3 * x
		assert f.val == [70]
		assert f.der == [17]
		assert repr(f) == 'Var({}, {})'.format(f.val, f.der)

	# Run tests within test_DeriveAliveVar
	test_scalar_without_bracket()
	test_scalar_with_bracket()
	test_vector_input()
	test_with_preset_der()
	test_repr()


def test_DeriveAlive_comparisons():
	def test_eq():
		x = da.Var(3)
		y = da.Var(3)
		z = da.Var(5)

		# Ensure that dunder method is called appropriately
		assert x.__eq__(y)
		assert x == y
		assert not (x == z)

		f = 2 * x + x / 3
		g = (3 * x) / 9 + x * 2
		assert f.__eq__(g)
		assert f == g

	def test_ne():
		x = da.Var(3)
		z = da.Var(5)

		# Ensure that dunder method is called appropriately
		assert x.__ne__(z)
		assert x != z

		f = 2 * x + x / 4
		g = x / 3 + x * 2
		assert f.__ne__(g)
		assert f != g

	def test_lt():
		x = da.Var(3)
		z = da.Var(5)
		assert x < z
		assert x < 4
		assert not (x < 3)
		assert not (x < 2)
		assert 2 < x
		assert not (z < x)

	def test_le():
		# Scalar comparisons
		x = da.Var(3)
		z = da.Var(5)
		assert x <= z
		assert x <= 4
		assert x <= 3
		assert not (x <= 2)	
		assert 2 <= x
		assert 3 <= x
		assert not (z <= x)

	def test_gt():
		x = da.Var(3)
		z = da.Var(5)
		assert z > x
		assert z > 4
		assert not (x > 3)
		assert x > 2
		assert not (2 > x)
		assert not (x > z)

	def test_ge():
		x = da.Var(3)
		z = da.Var(5)
		assert z >= x
		assert 4 >= x
		assert 3 >= x
		assert not (2 >= x)
		assert x >= 2
		assert x >= 3
		assert not (x >= z)

	test_eq()
	test_ne()
	test_lt()
	test_le()
	test_gt()
	test_ge()


def test_DeriveAlive_scalar_functions():
	'''Test scalar functions split up by operation type.'''

	def test_neg():
		x = da.Var(3.0)
		f = -x
		assert f.val == [-3.0]
		assert f.der == [-1.0]

		# Negate operator applied after the power
		f2 = -x ** 2
		assert f2.val == [-9.0]
		assert f2.der == [-6.0]

	def test_abs():
		# Negative value
		x = da.Var(-4.0)
		f = abs(x)
		assert f.val == [4.0]
		assert f.der == [-1.0]

		# Positive value
		y = da.Var(3.0)
		f = abs(y)
		assert f.val == [3.0]
		assert f.der == [1.0]

		# Zero
		with np.testing.assert_raises(ValueError):
			z = da.Var(0)
			f = abs(z)

	def test_constant():
		x = da.Var(5.0, None)
		f = x
		assert f == da.Var(5.0, None)

	def test_add():
		x = da.Var(2.0)
		f = x + 2
		assert f == da.Var(4.0, 1.0)

		f2 = x + 2 + x + 3 + x + 2 + x
		assert f2 == da.Var(15.0, 4.0)

		f3 = x - 3 + x - x + 2 - x
		assert f3 == da.Var(-1.0, 0.0)

		# State not modified
		assert x == da.Var(2.0, 1.0)

	def test_radd():
		x = da.Var(5.0)
		f = 2 + x
		assert f == da.Var(7.0, 1.0)

		f2 = 3 + x + x + 5 + x
		assert f2 == da.Var(23.0, 3.0)

		f3 = 5 + x + (6 + x)
		assert f3 == da.Var(21.0, 2.0)

		# State not modified
		assert x == da.Var(5.0, 1.0)

	def test_sub():
		x = da.Var(5.0)
		f = x - 3
		assert f == da.Var(2.0, 1.0)

		f2 = x - x
		assert f2 == da.Var(0.0, 0.0)

		f3 = x - x - x - x
		assert f3 == da.Var(-10.0, -2.0)

		f4 = x - 4 - 3
		assert f4 == da.Var(-2.0, 1.0)

		# State not modified
		assert x == da.Var(5.0, 1.0)

	def test_rsub():
		y = da.Var(3)
		f = 4 - y
		assert f == da.Var(1.0, -1.0)

		f2 = 3 - y
		assert f2 == da.Var(0.0, -1.0)

		f3 = 4 - 3 - y - 2
		assert f3 == da.Var(-4.0, -1.0)

		# State not modified
		assert y == da.Var(3.0, 1.0)

	def test_mul():
		x = da.Var(3.0)
		f = x * 2
		assert f == da.Var(6.0, 2)

		f2 = x * 3 + x * 4 + x * x * x
		assert f2 == da.Var(48.0, 34.0)

	def test_rmul():
		x = da.Var(5.0)
		f = 2 * x
		assert f == da.Var(10, 2)

		x2 = da.Var(0.0)
		f2 = 5 * x2
		assert f2 == da.Var(0.0, 5.0)

		# Derivative at x is 24 * x + 4
		f3 = 3 * x * 4 * x + 4 * x
		assert f3 == da.Var(320.0, 124)

	def test_truediv():
		# Expect value of 1.5, derivative of 0.5
		x = da.Var(3.0)
		f = x / 2
		assert np.round(f.val, 2) == [1.5]
		assert np.round(f.der, 2) == [0.5]

		# Constant of 1 has derivative of 0
		f2 = x / x
		assert f2 == da.Var(1.0, 0.0)

		f3 = (x / 2) * (1 / x)
		assert f3 == da.Var(0.5, 0.0)

		with np.testing.assert_raises(ZeroDivisionError):
			f4 = x / 0

	def test_rtruediv():
		# Expect value of 0.5, derivative of -0.25
		x = da.Var(2.0)
		f = 1 / x
		assert np.round(f.val, 2) == [0.5]
		assert np.round(f.der, 2) == [-0.25]

		f2 = 2 / (x * x)
		assert f2 == da.Var(0.50, -0.50)

		f3 = 2 / x / x
		assert f3 == f2

		with np.testing.assert_raises(ZeroDivisionError):
			zero = da.Var(0)
			f4 = 2 / zero

		f5 = 0 / x
		assert f5 == da.Var(0.0, 0.0)

	def test_sin():
		# Expect value of 18.42, derivative of 6.0
		x = da.Var(np.pi / 2)
		f = 3 * 2 * np.sin(x) + 2 * x + 4 * x + 3
		assert np.round(f.val, 2) == [18.42]
		assert np.round(f.der, 2) == [6.0]

		x2 = da.Var(np.pi)
		f2 = 3 * np.sin(x2) +  3
		assert np.round(f2.val, 2) == [3.0]
		assert f2.der == [-3.0]

	def test_cos():
		# Expect value of -10pi, derivative of 0.0 (because of -sin(pi))
		x = da.Var(np.pi)
		f = 5 * np.cos(x)+ np.cos(x) * 5
		assert f.val == [-10]
		assert abs(f.der) <= 1e-14

	def test_tan():
		# Expect value of 9.0, derivative of 12.0
		x = da.Var(np.pi / 4)
		f = 3 * 2 * np.tan(x) + 3
		assert np.round(f.val, 2) == [9.0]
		assert np.round(f.der, 2) == [12.0]

		x2 = da.Var(np.pi / 4)
		f2 = 3 * 2 * np.tan(x2) + 3
		assert f2.val == 9.0
		assert np.round(f2.der, 2) == [12.0]

		# Tangent is undefined for multiples of pi/2 >= pi/2
		with np.testing.assert_raises(ValueError):
			x3 = da.Var(3 * np.pi / 2)
			f3 = np.tan(x3)

	def test_arcsin():
		x = da.Var(0)
		f = np.arcsin(x)
		assert f.val == [0.0]
		assert f.der == [1.0]

		# Domain of arcsin(x) is -1 <= x <= 1
		with np.testing.assert_raises(ValueError):
			x = da.Var(-1.01)
			np.arcsin(x)

	def test_arccos():
		x = da.Var(0)
		f = np.arccos(x)
		assert f.val == [np.pi / 2]
		assert f.der == [-1.0]

		# Domain of arccos(x) is -1 <= x <= 1
		with np.testing.assert_raises(ValueError):
			x = da.Var(1.01)
			np.arccos(x)

	def test_arctan():
		x = da.Var(0.5)
		f = np.arctan(x)
		assert np.round(f.val, 2) == [0.46]
		assert f.der == [0.8]

	def test_sinh():
		x = da.Var(-1.0)
		f = np.sinh(x)
		assert np.round(f.val, 2) == [-1.18]
		assert np.round(f.der, 2) == [1.54]

	def test_cosh():
		x = da.Var(0.5)
		f = np.cosh(x)
		assert np.round(f.val, 2) == [1.13]
		assert np.round(f.der, 2) == [0.52]

	def test_tanh():
		x = da.Var(0.75)
		f = np.tanh(x)
		assert np.round(f.val, 2) == [0.64]
		assert np.round(f.der, 2) == [0.60]

	def test_pow():
		def method_version():
			'''This is for .pow() method check'''

			# Expect value of 64.0, derivative of 48.0
			x = da.Var(4.0)
			f = x.pow(3)
			assert np.round(f.val, 2) == [64.0]
			assert np.round(f.der, 2) == [48.0]
			assert f == x * x * x
		
			# Divides by zero when computing derivative (0^{-1/2} term)
			with np.testing.assert_raises(ZeroDivisionError):
				zero = da.Var(0)
				f_zero = zero.pow(1 / 2)

			x2 = da.Var(4.0)
			f2 = x2.pow(1 / 2)
			assert np.round(f2.val, 2) == [2.0]
			assert np.round(f2.der, 2) == [0.25]

			with np.testing.assert_raises(ValueError):
				x3 = da.Var(-2)
				f3 = x3.pow(1 / 2)

			x4 = da.Var(4.0)
			f4 = x4.pow(3)
			assert f4.val == [64.0]
			assert f4.der == [48.0]

		
		def dunder_version():
			'''Compute same tests using ** notation'''

			# Expect value of 64.0, derivative of 48.0
			x = da.Var(4.0)
			f = x ** 3
			assert np.round(f.val, 2) == [64.0]
			assert np.round(f.der, 2) == [48.0]
			assert f == x * x * x
		
			# Divides by zero when computing derivative (0^{-1/2} term)
			with np.testing.assert_raises(ZeroDivisionError):
				zero = da.Var(0)
				f_zero = zero ** (1 / 2)

			x2 = da.Var(4.0)
			f2 = x2 ** (1 / 2)
			assert np.round(f2.val, 2) == [2.0]
			assert np.round(f2.der, 2) == [0.25]

			with np.testing.assert_raises(ValueError):
				x3 = da.Var(-2)
				f3 = x3 ** (1 / 2)

			x4 = da.Var(4.0)
			f4 = x4 ** 3
			assert f4.val == [64.0]
			assert f4.der == [48.0]
		
		method_version()
		dunder_version()

	def test_rpow():
		x = da.Var(4)
		f = 2 ** x
		assert f.val == [16.0]
		assert np.round(f.der, 2) == [11.09]

		with np.testing.assert_raises(ZeroDivisionError):
			zero = da.Var(-1)
			f_zero = 0 ** zero  
			
		x1 = da.Var(0)
		f1 = 0 ** x1
		assert f1.val == [1]
		assert f1.der == [0]

		x2 = da.Var(3)
		f2 = 0 ** x2
		assert f2.val == [0]
		assert f2.der == [0]
		
		with np.testing.assert_raises(ValueError):
			neg = da.Var(2)
			f_neg = (-2) ** neg

	def test_sqrt():
		x = da.Var(16)
		f = np.sqrt(x)
		assert f == da.Var(4, 1 / 8)

		y = da.Var(0)
		with np.testing.assert_raises(ZeroDivisionError):
			g = np.sqrt(y)

	def test_log():
		# Expect value of 1.0, derivative of 0.04342944819032518
		x = da.Var(10)
		f = x.log(10)
		assert np.round(f.val, 2) == [1.0]
		assert np.round(f.der, 4) == [0.0434]

		with np.testing.assert_raises(ValueError):
			x2 = da.Var(0)
			f2 = x2.log(2)

	def test_exp():
		# Expect value of 2.718281828459045, derivative of 2.718281828459045
		x = da.Var(1)
		f = np.exp(x)
		assert np.round(f.val, 2) == [2.72]
		assert np.round(f.der, 2) == [2.72]

		f2 = np.exp(2 * x)
		assert f2 == da.Var(np.exp(2), 2 * np.exp(2))

	# Run tests within test_DeriveAlive_scalar_functions()
	test_neg()
	test_abs()
	test_constant()
	test_add()
	test_radd()
	test_sub()
	test_rsub()
	test_mul()
	test_rmul()
	test_truediv()
	test_rtruediv()
	test_sin()
	test_cos()
	test_tan()
	test_arcsin()
	test_arccos()
	test_arctan()
	test_sinh()
	test_cosh()
	test_tanh()
	test_pow()
	test_rpow()
	test_sqrt()
	test_log()
	test_exp()


def test_DeriveAlive_Vec():
	'''Test constructor of Vec class to ensure proper variable initializations.'''	

	def test_vector_input():
		x = da.Var(2, [1, 0])
		y = da.Var(3, [0, 1])
		f = da.Var([x ** 2, y ** 2, x * y])
		np.testing.assert_array_equal(x.val, np.array([2]))
		np.testing.assert_array_equal(x.der, np.array([1, 0]))
		np.testing.assert_array_equal(y.val, np.array([3]))
		np.testing.assert_array_equal(y.der, np.array([0, 1]))
		np.testing.assert_array_equal(f.val, np.array([4, 9, 6]))
		np.testing.assert_array_equal(f.der, np.array([[4, 0], [0, 6], [3, 2]]))

		a = da.Var([1, 2, 3, 4], None)
		b = da.Var([1, x, y, 4])
		c = da.Var([1, x, 3, x])
		d = np.array([5, 6, 7, 8])
		np.testing.assert_array_equal(a.val, np.array([1, 2, 3, 4]))
		np.testing.assert_array_equal(a.der, np.array([None]))
		np.testing.assert_array_equal(b.val, np.array([1, 2, 3, 4]))
		np.testing.assert_array_equal(b.der, np.array([[0, 0], [1, 0], [0, 1], [0, 0]]))
		np.testing.assert_array_equal(c.val, np.array([1, 2, 3, 2]))
		np.testing.assert_array_equal(c.der, np.array([[0, 0], [1, 0], [0, 0], [1, 0]]))

		# Operations between vectors of scalars
		add_scalars = a + a
		sub_scalars = a - a
		mul_scalars = a * a
		div_scalars = a / a
		np.testing.assert_array_equal(add_scalars.val, np.array([2, 4, 6, 8]))
		np.testing.assert_array_equal(add_scalars.der, np.array(None))
		np.testing.assert_array_equal(sub_scalars.val, np.array([0, 0, 0, 0]))
		np.testing.assert_array_equal(sub_scalars.der, np.array(None))
		np.testing.assert_array_equal(mul_scalars.val, np.array([1, 4, 9, 16]))
		np.testing.assert_array_equal(mul_scalars.der, np.array(None))
		np.testing.assert_array_equal(div_scalars.val, np.array([1, 1, 1, 1]))
		np.testing.assert_array_equal(div_scalars.der, np.array(None))

		# Operations between vector of scalars and vector containing Vars
		add_mix = a + b
		add_mix_rev = b + a
		sub_mix = a - b
		sub_mix_rev = b - a
		mul_mix = a * b
		mul_mix_rev = b * a
		div_mix = a / b
		div_mix_rev = b / a

		np.testing.assert_array_equal(add_mix.val, add_scalars.val)
		np.testing.assert_array_equal(add_mix.der, np.array([[0, 0], [1, 0], [0, 1], [0, 0]]))
		np.testing.assert_array_equal(add_mix_rev.val, add_mix.val)
		np.testing.assert_array_equal(add_mix_rev.der, add_mix.der)
		np.testing.assert_array_equal(sub_mix.val, sub_scalars.val)
		np.testing.assert_array_equal(sub_mix.der, np.array([[0, 0], [-1, 0], [0, -1], [0, 0]]))
		np.testing.assert_array_equal(sub_mix_rev.val, sub_mix.val)
		np.testing.assert_array_equal(sub_mix_rev.der, -sub_mix.der)
		np.testing.assert_array_equal(mul_mix.val, mul_scalars.val)
		np.testing.assert_array_equal(mul_mix.der, np.array([[0, 0], [2, 0], [0, 3], [0, 0]]))
		np.testing.assert_array_equal(mul_mix_rev.val, mul_mix.val)
		np.testing.assert_array_equal(mul_mix_rev.der, mul_mix.der)
		np.testing.assert_array_equal(div_mix.val, div_scalars.val)
		np.testing.assert_array_equal(div_mix.der, np.array([[0, 0], [-1 / 2, 0], [0, -1 / 3], [0, 0]]))
		np.testing.assert_array_equal(div_mix_rev.val, div_mix.val)
		np.testing.assert_array_equal(div_mix_rev.der, np.array([[0, 0], [1 / 2, 0], [0, 1 / 3], [0, 0]]))

	test_vector_input()

def test_DeriveAlive_vector_functions_m_to_1():
	'''Test vector functions from m dimensions to 1 dimension, split up by operation type.'''
	
	def test_neg():
		x = da.Var(3.0, [1, 0, 0])
		y = da.Var(1.0, [0, 1, 0])
		z = da.Var(2.0, [0, 0, 1])
		f = x ** 2 + y ** 3 + x * z
		f1 = -f
		np.testing.assert_array_equal(f1.val, np.array([-16.]))
		np.testing.assert_array_equal(f1.der, np.array([-8., -3., -3.]))

	def test_abs():
		x = da.Var(-3.0, [1, 0, 0])
		y = da.Var(-1.0, [0, 1, 0])
		z = da.Var(-2.0, [0, 0, 1])
		f = x ** 3 + y ** 3 + x * z	
		f1 = abs(f)
		np.testing.assert_array_equal(f1.val, np.array([22.]))
		
	def test_constant():
		x = da.Var(3.0, None)
		y = da.Var(1.0, None)
		f = x ** 3 + y ** 3
		np.testing.assert_array_equal(f.val, np.array([28.]))
		np.testing.assert_array_equal(f.der, np.array(None))

	def test_add():
		x = da.Var(3.0, [1, 0, 0])
		y = da.Var(1.0, [0, 1, 0])
		z = da.Var(2.0, [0, 0, 1])
		f1 = x ** 3+ y ** 3+ x * z	
		f2 = x ** 2 + y * z - x * z
		f = f1 + f2	
		np.testing.assert_array_equal(f.val, np.array([39.]))
		np.testing.assert_array_equal(f.der,np.array([33.,  5.,  1.]))

		ff = f1 + 2	
		np.testing.assert_array_equal(ff.val, np.array([36.]))
		np.testing.assert_array_equal(ff.der,np.array([29.,  3.,  3.]))

	def test_radd():
		x = da.Var(3.0, [1, 0, 0])
		y = da.Var(1.0, [0, 1, 0])
		z = da.Var(2.0, [0, 0, 1])
		f1 = x ** 3 + y ** 3 + x * z
		f = 2 + f1
		np.testing.assert_array_equal(f.val, np.array([36.]))
		np.testing.assert_array_equal(f.der,np.array([29.,  3.,  3.]))
		
		ff = 2 + f1 + 5 + x
		np.testing.assert_array_equal(ff.val, np.array([44.]))
		np.testing.assert_array_equal(ff.der,np.array([30.,  3.,  3.]))


	def test_sub():
		x = da.Var(3.0, [1, 0, 0])
		y = da.Var(1.0, [0, 1, 0])
		z = da.Var(2.0, [0, 0, 1])
		f1 = x ** 3 + y ** 3 + x * z
		f2 = x ** 2 + y * z + -x * z
		f = f1 - f2	
		np.testing.assert_array_equal(f.val, np.array([29.]))
		np.testing.assert_array_equal(f.der,np.array([25.,  1.,  5.]))
		
		ff = f1 - 2	
		np.testing.assert_array_equal(ff.val, np.array([32.]))
		np.testing.assert_array_equal(ff.der,np.array([29.,  3.,  3.]))

	def test_rsub():
		x = da.Var(3.0, [1, 0, 0])
		y = da.Var(1.0, [0, 1, 0])
		z = da.Var(2.0, [0, 0, 1])
		f1 = x ** 3 + y ** 3 + x * z
		f = 2 - f1
		np.testing.assert_array_equal(f.val, np.array([-32.]))
		np.testing.assert_array_equal(f.der,np.array([-29.,  -3.,  -3.]))
		
		ff = 2 - f1 - 5 - x
		np.testing.assert_array_equal(ff.val, np.array([-40.]))
		np.testing.assert_array_equal(ff.der,np.array([-30.,  -3.,  -3.]))	

	def test_mul():
		x = da.Var(3.0, [1, 0, 0])
		y = da.Var(1.0, [0, 1, 0])
		z = da.Var(2.0, [0, 0, 1])
		f1 = x + y ** 2 + x * z	
		f2 = x * y - y * z
		f = f1 * f2	
		np.testing.assert_array_equal(f.val, np.array([10.]))
		np.testing.assert_array_equal(f.der,np.array([13., 12., -7.]))
		
		ff = f1 * 2	
		np.testing.assert_array_equal(ff.val, np.array([20.]))
		np.testing.assert_array_equal(ff.der,np.array([6., 4., 6.]))

	def test_rmul():
		x = da.Var(3.0, [1, 0, 0])
		y = da.Var(1.0, [0, 1, 0])
		z = da.Var(2.0, [0, 0, 1])
		f1 = x + y ** 2 + x * z	
		f = 2 * f1
		np.testing.assert_array_equal(f.val, np.array([20.]))
		np.testing.assert_array_equal(f.der,np.array([6., 4., 6.]))
		
		ff = 2 * f1 * 2 * x
		np.testing.assert_array_equal(ff.val, np.array([120.]))
		np.testing.assert_array_equal(ff.der,np.array([76., 24., 36.]))	

	def test_truediv():
		x = da.Var(3.0, [1, 0, 0])
		y = da.Var(1.0, [0, 1, 0])
		z = da.Var(2.0, [0, 0, 1])
		f = x + y ** 2 + x * z	
		f1 = f / 2
		np.testing.assert_array_equal(f1.val, np.array([5.]))
		np.testing.assert_array_equal(f1.der,np.array([1.5, 1.,  1.5]))	

		f2 = f / f
		np.testing.assert_array_equal(f2.val, np.array([1.]))
		np.testing.assert_array_equal(f2.der,np.array([0., 0., 0.]))

		f3 = f/z
		np.testing.assert_array_equal(f3.val, np.array([5.]))
		np.testing.assert_array_equal(f3.der, np.array([1.5,  1.,  -1.]))

		with np.testing.assert_raises(ZeroDivisionError):
			f4 = f / 0	

		has_zero = da.Var([0, 1, 2])
		with np.testing.assert_raises(ZeroDivisionError):
			f5 = f / has_zero

	def test_rtruediv():
		x = da.Var(3.0, [1, 0, 0])
		y = da.Var(1.0, [0, 1, 0])
		z = da.Var(2.0, [0, 0, 1])
		f = x + y ** 2 + x * z
		f1 = 3 / f
		np.testing.assert_array_equal(np.round(f1.val,2), np.array([0.3]))
		np.testing.assert_array_equal(f1.der, np.array([-0.09, -0.06, -0.09]))		

	def test_sin():
		x = da.Var(np.pi / 2, [1, 0])
		y = da.Var(0, [0, 1])
		f = 3 * np.sin(x) + 2 * np.sin(y)
		np.testing.assert_array_equal(np.round(f.val,2), np.array([3.]))
		np.testing.assert_array_equal(np.round(f.der,2), np.array([0., 2.]))

	def test_cos():
		x = da.Var(np.pi / 2, [1, 0])
		y = da.Var(0, [0, 1])
		f = 3 * np.cos(x) + 2 * np.cos(y)
		np.testing.assert_array_equal(np.round(f.val,2), np.array([2.]))
		np.testing.assert_array_equal(np.round(f.der,2), np.array([-3., 0.]))

	def test_tan():
		x = da.Var(np.pi / 4, [1, 0])
		y = da.Var(np.pi / 3, [0, 1])
		f = 3 * np.tan(x) + 2 * np.tan(y) 
		np.testing.assert_array_equal(np.round(f.val,2), np.array([6.46]))
		np.testing.assert_array_equal(np.round(f.der,2), np.array([6., 8.]))

		with np.testing.assert_raises(ValueError):
			z = da.Var(3 * np.pi / 2)
			f1 = np.tan(z) + np.tan(x)

	def test_arcsin():
		x = da.Var(0, [1, 0])
		y = da.Var(0, [0, 1])
		f = 3 * np.arcsin(x) + 2 * np.arcsin(y)
		np.testing.assert_array_equal(np.round(f.val,2), np.array([0]))
		np.testing.assert_array_equal(np.round(f.der,2), np.array([3., 2]))
		
		# test on the boundary -1 and 1
		z1 = da.Var(1, [1, 0])
		z2 = da.Var(-1, [0, 1])
		f_z = 3 * np.arcsin(z1) + 2 * np.arcsin(z2)
		np.testing.assert_array_equal(np.round(f_z.val,2), np.array([1.57]))
		np.testing.assert_array_equal(np.round(f_z.der,2), np.array([np.nan, np.nan]))

        # test out of range [-1, 1]
		with np.testing.assert_raises(ValueError):
			x = da.Var(-1.01, [1, 0])
			f = 3 * np.arcsin(x) + 2 * np.arcsin(y)

	def test_arccos():
		x = da.Var(0, [1, 0])
		y = da.Var(0.5, [0, 1])
		f = 3 * np.arccos(x) + 2 * np.arccos(y)
		np.testing.assert_array_equal(np.round(f.val,2), np.array([6.81]))
		np.testing.assert_array_equal(np.round(f.der,2), np.array([-3., -2.31]))

		# test on the boundary -1 and 1
		z1 = da.Var(1, [1, 0])
		z2 = da.Var(-1, [0, 1])
		f_z = 3 * np.arccos(z1) + 2 * np.arccos(z2)
		np.testing.assert_array_equal(np.round(f_z.val,2), np.array([6.28]))
		np.testing.assert_array_equal(np.round(f_z.der,2), np.array([np.nan, np.nan]))

        # test out of range [-1, 1]
		with np.testing.assert_raises(ValueError):
			x = da.Var(1.01, [1, 0])
			f = 3 * np.arccos(x) + 2 * np.arccos(y)

	def test_arctan():
		x = da.Var(0.5, [1, 0])
		y = da.Var(np.pi/2, [0, 1])
		f = 3 * np.arctan(x) + 2 * np.arctan(y)
		np.testing.assert_array_equal(np.round(f.val,2), np.array([3.4]))
		np.testing.assert_array_equal(np.round(f.der,2), np.array([2.4, 0.58]))

	def test_sinh():
		x = da.Var(-1, [1, 0])
		y = da.Var(0, [0, 1])
		f = 3 * np.sinh(x) + 2 * np.sinh(y)
		np.testing.assert_array_equal(np.round(f.val,2), np.array([-3.53]))
		np.testing.assert_array_equal(np.round(f.der,2), np.array([4.63, 2.]))

	def test_cosh():
		x = da.Var(-1, [1, 0])
		y = da.Var(0, [0, 1])
		f = 3 * np.cosh(x) + 2 * np.cosh(y)
		np.testing.assert_array_equal(np.round(f.val,2), np.array([6.63]))
		np.testing.assert_array_equal(np.round(f.der,2), np.array([-3.53, 0.]))

	def test_tanh():
		x = da.Var(-1, [1, 0])
		y = da.Var(0, [0, 1])
		f = 3 * np.tanh(x) + 2 * np.tanh(y)
		np.testing.assert_array_equal(np.round(f.val,2), np.array([-2.28]))
		np.testing.assert_array_equal(np.round(f.der,2), np.array([1.26, 2.]))

	def test_pow():
		def method_version():
			'''test .pow() method'''
			x = da.Var(3.0, [1, 0, 0])
			y = da.Var(1.0, [0, 1, 0])
			z = da.Var(2.0, [0, 0, 1])
			f = x + y+ z
			f1 = f.pow(2)
			np.testing.assert_array_equal(f1.val, np.array([36.]))
			np.testing.assert_array_equal(np.round(f1.der,2), np.array([12., 12., 12.]))

			# Divides by zero when computing derivative (0^{-1/2} term)
			with np.testing.assert_raises(ZeroDivisionError):
				f2 = x - y - z
				f_zero = f2.pow(1 / 2)	

			with np.testing.assert_raises(ValueError):	
				f3 = x - y - x * z
				f_negative = f3.pow(1 / 2)

				f4 = -f3
				f5 = f4.pow(1/2)
				np.testing.assert_array_equal(f5.val, np.array([2.]))
				np.testing.assert_array_equal(np.round(f5.der,2), np.array([0.25, 0.25, 0.75]))      
       
        # Compute same tests using ** notation
		def dunder_version():
			x = da.Var(3.0, [1, 0, 0])
			y = da.Var(1.0, [0, 1, 0])
			z = da.Var(2.0, [0, 0, 1])
			f = x + y+ z
			f1 = f ** 2
			np.testing.assert_array_equal(f1.val, np.array([36.]))
			np.testing.assert_array_equal(np.round(f1.der,2), np.array([12., 12., 12.]))

			with np.testing.assert_raises(ZeroDivisionError):
				f2 = x - y - z
				f_zero = f2 ** (1 / 2)	

			with np.testing.assert_raises(ValueError):	
				f3 = x - y - x*z
				f_negative = f3 ** (1 / 2)

				f4 = -f3
				f5 = f4 ** (1/2)
				np.testing.assert_array_equal(f5.val, np.array([2.]))
				np.testing.assert_array_equal(np.round(f5.der,2), np.array([0.25, 0.25, 0.75]))

			method_version()
			dunder_version()

	def test_rpow():
		x = da.Var(3.0, [1, 0, 0])
		y = da.Var(1.0, [0, 1, 0])
		z = da.Var(2.0, [0, 0, 1])
		f = x + y + z
		f1 = 2 ** f
		np.testing.assert_array_equal(f1.val, np.array([64.]))
		np.testing.assert_array_equal(np.round(f1.der,2), np.array([44.36, 44.36, 44.36]))

	def test_sqrt():
		x = da.Var(3.0, [1, 0, 0])
		y = da.Var(1.0, [0, 1, 0])
		z = da.Var(2.0, [0, 0, 1])
		f = x + y + z
		f1 = np.sqrt(f)
		np.testing.assert_array_equal(np.round(f1.val,2), np.array([2.45]))
		np.testing.assert_array_equal(np.round(f1.der,2), np.array([0.2, 0.2, 0.2]))

		with np.testing.assert_raises(ZeroDivisionError):
			f2 = x - 2 * y - z
			f2_1 = 0 ** f2	

		with np.testing.assert_raises(ValueError):
			f2_2 = (-2) ** f2		

		f3 = x - y - z	
		f3_1 = 0 ** f3
		np.testing.assert_array_equal(f3_1.val, np.array([1.]))
		np.testing.assert_array_equal(f3_1.der, np.array([0.]))

	def test_log():
		x = da.Var(3.0, [1, 0, 0])
		y = da.Var(1.0, [0, 1, 0])
		z = da.Var(2.0, [0, 0, 1])
		f = x + y + z
		f1 = f.log(10)
		np.testing.assert_array_equal(np.round(f1.val,2), np.array([0.78]))
		np.testing.assert_array_equal(np.round(f1.der,2), np.array([0.07, 0.07, 0.07]))

		with np.testing.assert_raises(ValueError):
			f2 = x - y - z
			f3 = f2.log(2)

	def test_exp():
		x = da.Var(0, [1, 0, 0])
		y = da.Var(1.0, [0, 1, 0])
		z = da.Var(2.0, [0, 0, 1])
		f = x + y + z
		f1 = np.exp(f)
		np.testing.assert_array_equal(np.round(f1.val,2), np.array([20.09]))
		np.testing.assert_array_equal(np.round(f1.der,2), np.array([20.09, 20.09, 20.09]))


	# Run tests within test_DeriveAlive_vector_functions_m_to_1()
	test_neg()
	test_abs()
	test_constant()
	test_add()
	test_radd()
	test_sub()
	test_rsub()
	test_mul()
	test_rmul()
	test_truediv()
	test_rtruediv()
	test_sin()
	test_cos()
	test_tan()
	test_arcsin()
	test_arccos()
	test_arctan()
	test_sinh()
	test_cosh()
	test_tanh()
	test_pow()
	test_rpow()
	test_sqrt()
	test_log()
	test_exp()


def test_DeriveAlive_vector_functions_1_to_n():
	'''Test vector functions from 1 dimension to n dimensions, split up by operation type.'''

	def test_neg():
		x = da.Var(3.0, [1])
		f = da.Var([x, x ** 2, x ** 3])
		f1 = -f
		np.testing.assert_array_equal(f1.val, np.array([ -3.,  -9., -27.]))
		np.testing.assert_array_equal(f1.der, np.array([[-1.],
			                                            [-6.],
			                                            [-27.]]))

	def test_abs():
		x = da.Var(-3.0, [1])
		f = da.Var([x, x ** 2, x ** 3])
		f1 = abs(f)
		np.testing.assert_array_equal(f1.val, np.array([ 3.,  9., 27.]))
		np.testing.assert_array_equal(f1.der, np.array([[-1.],
														[-6],
														[-27.]]))

	def test_constant():
		x = da.Var(3.0)
		f = da.Var([x ** 3, x ** 2])
		np.testing.assert_array_equal(f.val, np.array([27.,  9.]))
		np.testing.assert_array_equal(f.der, np.array([[27.],
													   [ 6.]]))

	def test_add():
		x = da.Var(3.0, [1])
		f1 = da.Var([x ** 3, x ** 2, x])	
		f2 = da.Var([x ** 2, x, x * 2])
		f = f1 + f2	
		np.testing.assert_array_equal(f.val, np.array([36.,  12.,  9.]))
		np.testing.assert_array_equal(f.der,np.array([[33.],
                                                      [ 7.],
                                                      [ 3.]]))
		ff = f1 + 2	
		np.testing.assert_array_equal(ff.val, np.array([29.,  11.,  5.]))
		np.testing.assert_array_equal(ff.der,np.array([[27.],
                                                       [ 6.],
                                                       [ 1.]]))

	def test_radd():
		x = da.Var(3.0, [1])
		f1 = da.Var([x ** 3, x ** 2, x])
		f = 2 + f1
		np.testing.assert_array_equal(f.val, np.array([29.,  11.,  5.]))
		np.testing.assert_array_equal(f.der,np.array([[27.],
                                                      [ 6.],
                                                      [ 1.]]))
		ff = 2 + f1 + 5 + x
		np.testing.assert_array_equal(ff.val, np.array([37.,  19.,  13.]))
		np.testing.assert_array_equal(ff.der,np.array([[28.],
                                                       [ 7.],
                                                       [ 2.]]))

	def test_sub():
		x = da.Var(3.0, [1])
		f1 = da.Var([x ** 3, x ** 2, x])	
		f2 = da.Var([x ** 2, x, x * 2])
		f = f1 - f2	
		np.testing.assert_array_equal(f.val, np.array([18.,  6.,  -3.]))
		np.testing.assert_array_equal(f.der,np.array([[21.],
                                                      [ 5.],
                                                      [-1.]]))
		ff = f1 - 2	
		np.testing.assert_array_equal(ff.val, np.array([25.,  7.,  1.]))
		np.testing.assert_array_equal(ff.der,np.array([[27.],
                                                       [ 6.],
                                                       [ 1.]]))

	def test_rsub():
		x = da.Var(3.0, [1])
		f1 = da.Var([x ** 3, x ** 2, x])
		f = f1 - 2	
		np.testing.assert_array_equal(f.val, np.array([25.,  7.,  1.]))
		np.testing.assert_array_equal(f.der,np.array([[27.],
                                                      [ 6.],
                                                      [ 1.]]))

		ff = 2 - f1 - 5 - x	
		np.testing.assert_array_equal(ff.val, np.array([-33., -15., -9.]))
		np.testing.assert_array_equal(ff.der,np.array([[-28.],
                                                       [ -7.],
                                                       [ -2.]]))

	def test_mul():
		x = da.Var(3.0, [1])
		f1 = da.Var([x ** 3, x ** 2, x])
		f2 = da.Var([x ** 2, x, x * 2])
		f = f1 * f2
		np.testing.assert_array_equal(f.val, np.array([243.,  27.,  18.]))
		np.testing.assert_array_equal(f.der, np.array([[405.],
                                                       [ 27.],
                                                       [ 12.]]))
		ff = 2 * f1 * 5 * x
		np.testing.assert_array_equal(ff.val, np.array([810., 270.,  90.]))
		np.testing.assert_array_equal(ff.der,np.array([[1080.],
                                                       [ 270.],
                                                       [ 60.]]))

	def test_rmul():
		x = da.Var(3.0, [1])
		f1 = da.Var([x ** 3, x ** 2, x])
		f2 = da.Var([x ** 2, x, x * 2])
		f = f1 * 2	
		np.testing.assert_array_equal(f.val, np.array([54., 18.,  6.]))
		np.testing.assert_array_equal(f.der,np.array([[54.],
                                                      [12.],
                                                      [ 2.]]))

		ff = 2 * f1 * 5 * x
		np.testing.assert_array_equal(ff.val, np.array([810., 270.,  90.]))
		np.testing.assert_array_equal(ff.der, np.array([[1080.],
                                                        [ 270.],
                                                        [ 60.]]))

	def test_truediv():
		x = da.Var(3.0, [1])
		z = da.Var([x ** 3, x ** 2, x])
		w = da.Var([x, x, x])
		s = z / w
		np.testing.assert_array_equal(s.val, np.array([9, 3, 1]))
		np.testing.assert_array_equal(s.der, np.array([[6.],
													   [1.],
													   [0.]]))

		a = da.Var([ 1., x, x, 4.])
		b = da.Var([ 1., 2., 3., 4.], None)
		c = da.Var([ 3., 3., 3., 3.], None)

		f = c / a
		np.testing.assert_array_equal(f.val, np.array([ 3., 1., 1., 3 / 4]))
		np.testing.assert_array_equal(f.der, np.array([[ 0.],
													   [-1 / 3],
													   [-1 / 3],
													   [ 0.]]))

		g = c / b
		np.testing.assert_array_equal(g.val, np.array([ 3., 3 / 2, 1., 3 / 4]))
		np.testing.assert_array_equal(g.der, np.array(None))

		has_zero = da.Var([ 0., 1., 2.])
		with np.testing.assert_raises(ZeroDivisionError):
			f = z / has_zero

	def test_rtruediv():
		x = da.Var(3.0, [1])
		z = da.Var([x ** 3, x ** 2, x])
		a = da.Var([ 1., x, x, 4.])
		b = da.Var([ 1., 2., 3., 4.], None)

		f = 3 / a
		np.testing.assert_array_equal(f.val, np.array([ 3., 1., 1., 3 / 4]))
		np.testing.assert_array_equal(f.der, np.array([[ 0.],
													   [-1 / 3],
													   [-1 / 3],
													   [ 0.]]))

		g = 3 / b
		np.testing.assert_array_equal(g.val, np.array([3, 3 / 2, 1, 3 / 4]))
		np.testing.assert_array_equal(g.der, np.array(None))

		has_zero = da.Var([ 0., 1., 2.])
		with np.testing.assert_raises(ZeroDivisionError):
			f = z / has_zero

	def test_sin():
		x = da.Var(np.pi / 2, [1])
		f = da.Var([np.sin(x), np.sin(x) + 1, np.sin(x) ** 2])
		np.testing.assert_array_equal(np.round(f.val, 2), np.array([ 1., 2., 1.]))
		np.testing.assert_array_equal(np.round(f.der, 2), np.array([[ 0.],
                                                                    [ 0.],
                                                                    [ 0.]]))

	def test_cos():
		x = da.Var(np.pi / 2, [ 1.])
		f = da.Var([np.cos(x) + 1, np.cos(x), np.cos(x) ** 2])
		np.testing.assert_array_equal(np.round(f.val, 2), np.array([ 1., 0., 0.]))
		np.testing.assert_array_equal(np.round(f.der, 2), np.array([[-1.],
                                                                    [-1.],
                                                                    [ 0.]]))

	def test_tan():
		x = da.Var(np.pi / 3, [1])
		f = da.Var([np.tan(x) + 1, np.tan(x), np.tan(x) ** 2])
		np.testing.assert_array_equal(np.round(f.val, 2), np.array([ 2.73, 1.73, 3. ]))
		np.testing.assert_array_equal(np.round(f.der, 2), np.array([[ 4.],
                                                                    [ 4.],
                                                                    [ 13.86]]))

	def test_arcsin():
		x = da.Var(1, [1])
		f = da.Var([np.arcsin(x), np.arcsin(x) + 1, np.arcsin(x) ** 2])
		np.testing.assert_array_equal(np.round(f.val, 2), np.array([1.57, 2.57, 2.47]))
		np.testing.assert_array_equal(np.round(f.der, 2), np.array([[np.nan],
                                                                    [np.nan],
                                                                    [np.nan]]))	
		
		# test on the boundary -1 and 1
		z1 = da.Var(-1, [1])
		f_z = da.Var([np.arcsin(z1) + np.arcsin(z1), np.arcsin(z1) ** 2 + 1])
		np.testing.assert_array_equal(np.round(f_z.val, 2), np.array([-3.14, 3.47]))
		np.testing.assert_array_equal(np.round(f_z.der, 2), np.array([[np.nan],
                                                                      [np.nan]]))

	def test_arccos(): 
		x = da.Var(1, [1])
		f = da.Var([np.arccos(x), np.arccos(x) + 1, np.arccos(x) ** 2])  
		np.testing.assert_array_equal(np.round(f.val, 2), np.array([ 0., 1., 0.]))
		np.testing.assert_array_equal(np.round(f.der, 2), np.array([[np.nan],
                                                                    [np.nan],
                                                                    [np.nan]]))	

    	# test on the boundary -1 and 1
		z1 = da.Var(-1, [1])
		f_z = da.Var([np.arccos(z1) + np.arccos(z1), np.arccos(z1) ** 2 + 1])
		np.testing.assert_array_equal(np.round(f_z.val, 2), np.array([ 6.28, 10.87]))
		np.testing.assert_array_equal(np.round(f_z.der, 2), np.array([[np.nan],
                                                                     [np.nan]]))

	def test_arctan():
		x = da.Var(0.5, [1])
		f = da.Var([3 * np.arctan(x), 2 * np.arctan(x)])
		np.testing.assert_array_equal(np.round(f.val, 2), np.array([1.39, 0.93]))
		np.testing.assert_array_equal(np.round(f.der, 2), np.array([[2.4],
                                                                    [1.6]]))

	def test_sinh():
		x = da.Var(-1, [1])
		f = da.Var([3 * np.sinh(x), 2 * np.sinh(x)])
		np.testing.assert_array_equal(np.round(f.val, 2), np.array([-3.53, -2.35]))
		np.testing.assert_array_equal(np.round(f.der, 2), np.array([[4.63],
                                                                    [3.09]]))

	def test_cosh():
		x = da.Var(-1, [1])
		f = da.Var([3 * np.cosh(x), 2 * np.cosh(x)])
		np.testing.assert_array_equal(np.round(f.val, 2), np.array([4.63, 3.09]))
		np.testing.assert_array_equal(np.round(f.der, 2), np.array([[-3.53],
                                                                    [-2.35]]))

	def test_tanh():
		x = da.Var(-1, [1])
		f = da.Var([3 * np.tanh(x), 2 * np.tanh(x)])
		np.testing.assert_array_equal(np.round(f.val, 2), np.array([-2.28, -1.52]))
		np.testing.assert_array_equal(np.round(f.der, 2), np.array([[1.26],
                                                                    [0.84]]))

	def test_pow():
		def method_version():
			'''test the .pow() method'''
			x = da.Var(3.0, [1])
			f = da.Var([2 * x, x - 3, x ** 2])
			f1 = f.pow(2)
			np.testing.assert_array_equal(f1.val, np.array([ 36.,  0., 81.]))
			np.testing.assert_array_equal(np.round(f1.der, 2), np.array([[ 24.],
                                                                         [  0.],
                                                                         [108.]]))

			# Divides by zero when computing derivative (0^{-1/2} term)
			with np.testing.assert_raises(ZeroDivisionError):
				f_zero = f.pow(1 / 2)

			with np.testing.assert_raises(ValueError):	
				f3 = da.Var([2 * x, x + 1, -x ** 2])
				f_negative = f3.pow(1 / 2)

			f4 = abs(f3)
			f5 = f4.pow(1/2)
			np.testing.assert_array_equal(np.round(f5.val, 2), np.array([ 2.45, 2. , 3. ]))
			np.testing.assert_array_equal(np.round(f5.der, 2), np.array([[0.41],
                                                                         [0.25],
                                                                         [1.  ]]))      
       
        # Compute same tests using ** notation
		def dunder_version():
			x = da.Var(3.0, [1])
			f = da.Var([2 * x, x - 3, x ** 2])
			f1 = f.pow(2)
			np.testing.assert_array_equal(f1.val, np.array([ 36.,  0., 81.]))
			np.testing.assert_array_equal(np.round(f1.der, 2), np.array([[ 24.],
                                                                         [  0.],
                                                                         [108.]]))

			with np.testing.assert_raises(ZeroDivisionError):
				f_zero = f ** (1 / 2)	

			with np.testing.assert_raises(ValueError):	
				f3 = da.Var([2 * x, x + 1, -x ** 2])
				f_negative = f3 ** (1 / 2)

			f4 = abs(f3)
			f5 = f4 ** (1 / 2)
			np.testing.assert_array_equal(np.round(f5.val, 2), np.array([ 2.45, 2. , 3. ]))
			np.testing.assert_array_equal(np.round(f5.der, 2), np.array([[0.41],
                                                                         [0.25],
                                                                         [1.  ]])) 

		method_version()
		dunder_version()

	def test_rpow():
		x = da.Var(3.0, [1])
		f = da.Var([2 * x, x - 1, x ** 2])
		f1 = 2 ** f
		np.testing.assert_array_equal(np.round(f1.val, 2), np.array([64., 4., 512.]))
		np.testing.assert_array_equal(np.round(f1.der, 2), np.array([[  88.72],
																	 [   2.77],
																	 [2129.35]])) # x: array([ 44.36,   2.77, 354.89])

	def test_sqrt(): 
		x = da.Var(3.0, [1])
		f = da.Var([2 * x, x + 1, x ** 2])
		f1 = np.sqrt(f)
		np.testing.assert_array_equal(np.round(f1.val, 2), np.array([2.45, 2., 3.  ]))
		np.testing.assert_array_equal(np.round(f1.der, 2), np.array([[0.41],
                                                                     [0.25],
                                                                     [ 1. ]]))
		f2 = da.Var([2 * x, x + 1, -x ** 2])
		with np.testing.assert_raises(ZeroDivisionError):
			f2_1 = 0 ** f2	

		with np.testing.assert_raises(ValueError):
			f2_2 = (-2) ** f2		


	def test_log():
		x = da.Var(3.0, [1])
		f = da.Var([2 * x, x + 1, x ** 2])
		f1 = f.log(10)
		np.testing.assert_array_equal(np.round(f1.val, 2), np.array([ 0.78, 0.6 , 0.95]))
		np.testing.assert_array_equal(np.round(f1.der, 2), np.array([[0.14],
                                                                     [0.11],
                                                                     [0.29]]))

		with np.testing.assert_raises(ValueError):
			f2 = da.Var([2 * x, x - 3, x ** 2])
			f3 = f2.log(2)

	def test_exp():
		x = da.Var(3.0, [1])
		f = da.Var([2 * x, x + 1, x ** 2])
		f1 = np.exp(f)
		np.testing.assert_array_equal(np.round(f1.val, 2), np.array([ 403.43,   54.60,  8103.08 ]))
		np.testing.assert_array_equal(np.round(f1.der, 2), np.array([[  806.86],
                                                                     [   54.60],
                                                                     [48618.50]]))

	# Run tests within test_DeriveAlive_vector_functions_1_to_n()
	test_neg()
	test_abs()
	test_constant()
	test_add()
	test_radd()
	test_sub()
	test_rsub()
	test_mul()
	test_rmul()
	test_truediv()
	test_rtruediv()
	test_sin()
	test_cos()
	test_tan()
	test_arcsin()
	test_arccos()
	test_arctan()
	test_sinh()
	test_cosh()
	test_tanh()
	test_pow()
	test_rpow()
	test_sqrt()
	test_log()
	test_exp()


def test_DeriveAlive_vector_functions_m_to_n():
	'''Test vector functions from m dimensions to n dimensions, split up by operation type.'''
    
	def test_neg():
		x = da.Var(3.0, [1, 0, 0])
		y = da.Var(1.0, [0, 1, 0])
		z = da.Var(2.0, [0, 0, 1])
		f = da.Var([x ** 2, y ** 3, x * z])
		f1 = -f
		np.testing.assert_array_equal(f1.val, np.array([-9, -1, -6]))
		np.testing.assert_array_equal(f1.der, np.array([[-6.,  0.,  0.],
                                                        [ 0., -3.,  0.],
                                                        [-2.,  0., -3.]]))

	def test_abs():
		x = da.Var(3.0, [1, 0, 0])
		y = da.Var(1.0, [0, 1, 0])
		z = da.Var(2.0, [0, 0, 1])
		f = da.Var([-x ** 3, -y ** 3, -x * z])		
		f1 = abs(f)
		np.testing.assert_array_equal(f1.val, np.array([27.,  1.,  6.]))
		np.testing.assert_array_equal(f1.val, np.array([27.,  1.,  6.]))

	def test_constant():
		x = da.Var(3.0)
		y = da.Var(1.0)
		f = da.Var([x ** 3, y ** 3])
		np.testing.assert_array_equal(f.val, np.array([27.,  1.]))
		np.testing.assert_array_equal(f.der,np.array([[27.],[ 3.]]))

	def test_add():
		x = da.Var(3.0, [1, 0, 0])
		y = da.Var(1.0, [0, 1, 0])
		z = da.Var(2.0, [0, 0, 1])
		f1 = da.Var([x ** 3, y ** 3, x * z])	
		f2 = da.Var([x ** 2, y * z, -x * z])
		f = f1 + f2	
		np.testing.assert_array_equal(f.val, np.array([36.,  3.,  0.]))
		np.testing.assert_array_equal(f.der,np.array([[33.,  0.,  0.],
                                                      [ 0.,  5.,  1.],
                                                      [ 0.,  0.,  0.]]))
		ff = f1 + 2	
		np.testing.assert_array_equal(ff.val, np.array([29.,  3.,  8.]))
		np.testing.assert_array_equal(ff.der,np.array([[27.,  0.,  0.],
                                                       [ 0.,  3.,  0.],
                                                       [ 2.,  0.,  3.]]))

	def test_radd():
		x = da.Var(3.0, [1, 0, 0])
		y = da.Var(1.0, [0, 1, 0])
		z = da.Var(2.0, [0, 0, 1])
		f1 = da.Var([x ** 3, y ** 3, x * z])
		f = 2 + f1
		np.testing.assert_array_equal(f.val, np.array([29.,  3.,  8.]))
		np.testing.assert_array_equal(f.der,np.array([[27.,  0.,  0.],
                                                      [ 0.,  3.,  0.],
                                                      [ 2.,  0.,  3.]]))
		
		ff = 2 + f1 + 5 + x
		np.testing.assert_array_equal(ff.val, np.array([37., 11., 16.]))
		np.testing.assert_array_equal(ff.der,np.array([[28.,  0.,  0.],
                                                       [ 1.,  3.,  0.],
                                                       [ 3.,  0.,  3.]]))

	def test_sub():
		x = da.Var(3.0, [1, 0, 0])
		y = da.Var(1.0, [0, 1, 0])
		z = da.Var(2.0, [0, 0, 1])
		f1 = da.Var([x ** 3, y ** 3, x * z])	
		f2 = da.Var([x ** 2, y * z, -x * z])
		f = f1 - f2	
		np.testing.assert_array_equal(f.val, np.array([18., -1., 12.]))
		np.testing.assert_array_equal(f.der,np.array([[21.,  0.,  0.],
                                                      [ 0.,  1., -1.],
                                                      [ 4.,  0.,  6.]]))
		
		ff = f1 - 2	
		np.testing.assert_array_equal(ff.val, np.array([25., -1.,  4.]))
		np.testing.assert_array_equal(ff.der,np.array([[27.,  0.,  0.],
                                                       [ 0.,  3.,  0.],
                                                       [ 2.,  0.,  3.]]))

	def test_rsub():
		x = da.Var(3.0, [1, 0, 0])
		y = da.Var(1.0, [0, 1, 0])
		z = da.Var(2.0, [0, 0, 1])
		f1 = da.Var([x ** 3, y ** 3, x * z])
		f = 2 - f1
		np.testing.assert_array_equal(f.val, np.array([-25.,   1.,  -4.]))
		np.testing.assert_array_equal(f.der,np.array([[-27.,   0.,   0.],
                                                      [  0.,  -3.,   0.],
                                                      [ -2.,   0.,  -3.]]))
		
		ff = 2 - f1 - 5 - x
		np.testing.assert_array_equal(ff.val, np.array([-33.,  -7., -12.]))
		np.testing.assert_array_equal(ff.der,np.array([[-28.,   0.,   0.],
                                                       [ -1.,  -3.,   0.],
                                                       [ -3.,   0.,  -3.]]))	

	def test_mul():
		x = da.Var(3.0, [1, 0, 0])
		y = da.Var(1.0, [0, 1, 0])
		z = da.Var(2.0, [0, 0, 1])
		f1 = da.Var([x ** 3, y ** 3, x * z])	
		f2 = da.Var([x ** 2, y * z, -x * z])
		f = f1 * f2	
		np.testing.assert_array_equal(f.val, np.array([243.,   2., -36.]))
		np.testing.assert_array_equal(f.der,np.array([[405.,   0.,   0.],
                                                      [  0.,   8.,   1.],
                                                      [-24.,   0., -36.]]))
		
		ff = f1 * 2	
		np.testing.assert_array_equal(ff.val, np.array([54.,  2., 12.]))
		np.testing.assert_array_equal(ff.der,np.array([[54.,  0.,  0.],
                                                       [ 0.,  6.,  0.],
                                                       [ 4.,  0.,  6.]]))

	def test_rmul():
		x = da.Var(3.0, [1, 0, 0])
		y = da.Var(1.0, [0, 1, 0])
		z = da.Var(2.0, [0, 0, 1])
		f1 = da.Var([x ** 3, y ** 3, x * z])
		f = 2 * f1
		np.testing.assert_array_equal(f.val, np.array([54.,  2., 12.]))
		np.testing.assert_array_equal(f.der,np.array([[54.,  0.,  0.],
                                                      [ 0.,  6.,  0.],
                                                      [ 4.,  0.,  6.]]))
		
		ff = 2 * f1 * 5 * x
		np.testing.assert_array_equal(ff.val, np.array([810.,  30., 180.]))
		np.testing.assert_array_equal(ff.der,np.array([[1080.,    0.,    0.],
                                                       [ 270.,   90.,    0.],
                                                       [ 330.,    0.,   90.]]))

	def test_truediv():
		x = da.Var(2, [1, 0])
		y = da.Var(3, [0, 1])
		z = da.Var([x ** 2, y ** 2, x * y])
		w = da.Var([x, y, x])
		s = z / w

		np.testing.assert_array_equal(s.val, np.array([2, 3, 3]))
		np.testing.assert_array_equal(s.der, np.array([[1, 0], [0, 1], [0, 1]]))

		a = da.Var([1, x, y, 4])
		b = da.Var([1, 2, 3, 4], None)
		c = da.Var([3, 3, 3, 3], None)

		f = c / a
		np.testing.assert_array_equal(f.val, np.array([3, 3 / 2, 1, 3 / 4]))
		np.testing.assert_array_equal(f.der, np.array([[0, 0], [-3 / 4, 0], [0, -1 / 3], [0, 0]]))

		g = c / b
		np.testing.assert_array_equal(g.val, np.array([3, 3 / 2, 1, 3 / 4]))
		np.testing.assert_array_equal(g.der, np.array(None))

		has_zero = da.Var([0, 1, 2])
		with np.testing.assert_raises(ZeroDivisionError):
			f = z / has_zero

		# Expect value of 1.5, derivative of 0.5
		x = da.Var(3.0)
		f = x / 2
		assert np.round(f.val, 2) == [1.5]
		assert np.round(f.der, 2) == [0.5]

		# Constant of 1 has derivative of 0
		f2 = x / x
		assert f2 == da.Var(1.0, 0.0)

		f3 = (x / 2) * (1 / x)
		assert f3 == da.Var(0.5, 0.0)

	def test_rtruediv():
		x = da.Var(2, [1, 0])
		y = da.Var(3, [0, 1])
		z = da.Var([x ** 2, y ** 2, x * y])
		a = da.Var([1, x, y, 4])
		b = da.Var([1, 2, 3, 4], None)

		f = 3 / a
		np.testing.assert_array_equal(f.val, np.array([3, 3 / 2, 1, 3 / 4]))
		np.testing.assert_array_equal(f.der, np.array([[0, 0], [-3 / 4, 0], [0, -1 / 3], [0, 0]]))

		g = 3 / b
		np.testing.assert_array_equal(g.val, np.array([3, 3 / 2, 1, 3 / 4]))
		np.testing.assert_array_equal(g.der, np.array(None))	

		has_zero = da.Var([0, 1, 2])
		with np.testing.assert_raises(ZeroDivisionError):
			f = z / has_zero

	def test_sin():
		x = da.Var(np.pi / 2, [1, 0, 0])
		y = da.Var(0, [0, 1, 0])
		z = da.Var(np.pi / 3, [0, 0, 1])
		f = da.Var([np.sin(x), np.sin(y) + 1, np.sin(z) ** 2])
		np.testing.assert_array_equal(np.round(f.val,2), np.array([1., 1., 0.75]))
		np.testing.assert_array_equal(np.round(f.der,2), np.array([[0.  , 0.  , 0.  ],
                                                                   [0.  , 1.  , 0.  ],
                                                                   [0.  , 0.  , 0.87]]))		
	
	def test_cos():
		x = da.Var(np.pi / 2, [1, 0, 0])
		y = da.Var(0, [0, 1, 0])
		z = da.Var(np.pi / 6, [0, 0, 1])
		f = da.Var([np.cos(x) + 1, np.cos(y), np.cos(z) ** 2])
		np.testing.assert_array_equal(np.round(f.val,2), np.array([1., 1., 0.75]))
		np.testing.assert_array_equal(np.round(f.der,2), np.array([[-1.  , 0.  , 0.  ],
                                                                   [0.  , 0.  , 0.  ],
                                                                   [0.  , 0.  , -0.87]]))

	def test_tan():
		x = da.Var(np.pi / 3, [1, 0, 0])
		y = da.Var(0, [0, 1, 0])
		z = da.Var(np.pi / 4, [0, 0, 1])
		f = da.Var([np.tan(x) + 1, np.tan(y), np.tan(z) ** 2])
		np.testing.assert_array_equal(np.round(f.val,2), np.array([2.73, 0., 1. ]))
		np.testing.assert_array_equal(np.round(f.der,2), np.array([[4., 0., 0.],
                                                                   [0., 1., 0.],
                                                                   [0., 0., 4.]]))	

	def test_arcsin():
		x = da.Var(1, [1, 0, 0])
		y = da.Var(0, [0, 1, 0])
		z = da.Var(np.pi / 4, [0, 0, 1])
		f = da.Var([np.arcsin(x), np.arcsin(y) + 1, np.arcsin(z) ** 2])
		np.testing.assert_array_equal(np.round(f.val,2), np.array([1.57, 1., 0.82]))
		np.testing.assert_array_equal(np.round(f.der,2), np.array([[np.nan, np.nan  , np.nan],
                                                                   [0.  , 1.  , 0.  ],
                                                                   [0.  , 0.  , 2.92]]))	
		
		# test on the boundary -1 and 1
		z1 = da.Var(1, [1, 0])
		z2 = da.Var(-1, [0, 1])
		f_z = da.Var([np.arcsin(z1) + np.arcsin(z2), np.arcsin(z2) ** 2 + 1])
		np.testing.assert_array_equal(np.round(f_z.val,2), np.array([0.  , 3.47]))
		np.testing.assert_array_equal(np.round(f_z.der,2), np.array([[np.nan, np.nan],
                                                                     [np.nan, np.nan]]))

	def test_arccos():
		x = da.Var(1, [1, 0, 0])
		y = da.Var(0, [0, 1, 0])
		z = da.Var(np.pi / 4, [0, 0, 1]) 
		f = da.Var([np.arccos(x), np.arccos(y) + 1, np.arccos(z) ** 2])   
		np.testing.assert_array_equal(np.round(f.val,2), np.array([0.  , 2.57, 0.45]))
		np.testing.assert_array_equal(np.round(f.der,2), np.array([[np.nan, np.nan  , np.nan],
                                                                   [ 0.  , -1.  ,  0.  ],
                                                                   [ 0.  ,  0.  , -2.16]]))	

    	# test on the boundary -1 and 1
		z1 = da.Var(1, [1, 0])
		z2 = da.Var(-1, [0, 1])
		f_z = da.Var([np.arccos(z1) + np.arccos(z2), np.arccos(z2) ** 2 + 1])
		np.testing.assert_array_equal(np.round(f_z.val,2), np.array([3.14, 10.87]))
		np.testing.assert_array_equal(np.round(f_z.der,2), np.array([[np.nan, np.nan],
                                                                     [np.nan, np.nan]]))

	def test_arctan():
		x = da.Var(0.5, [1, 0])
		y = da.Var(np.pi/2, [0, 1])
		f = da.Var([3 * np.arctan(x), 2 * np.arctan(y)])
		np.testing.assert_array_equal(np.round(f.val,2), np.array([1.39, 2.01]))
		np.testing.assert_array_equal(np.round(f.der,2), np.array([[2.4 , 0.  ],
                                                                   [0.  , 0.58]]))

	def test_sinh():
		x = da.Var(-1, [1, 0])
		y = da.Var(0, [0, 1])
		f = da.Var([3 * np.sinh(x), 2 * np.sinh(y)])
		np.testing.assert_array_equal(np.round(f.val,2), np.array([-3.53,  0.]))
		np.testing.assert_array_equal(np.round(f.der,2), np.array([[4.63, 0.  ],
                                                                   [0.  , 2.  ]]))
	def test_cosh():
		x = da.Var(-1, [1, 0])
		y = da.Var(0, [0, 1])
		f = da.Var([3 * np.cosh(x), 2 * np.cosh(y)])
		np.testing.assert_array_equal(np.round(f.val,2), np.array([4.63, 2.]))
		np.testing.assert_array_equal(np.round(f.der,2), np.array([[-3.53,  0.  ],
                                                                   [ 0.  ,  0.  ]]))

	def test_tanh():
		x = da.Var(-1, [1, 0])
		y = da.Var(0, [0, 1])
		f = da.Var([3 * np.tanh(x), 2 * np.tanh(y)])
		np.testing.assert_array_equal(np.round(f.val,2), np.array([-2.28,  0.  ]))
		np.testing.assert_array_equal(np.round(f.der,2), np.array([[1.26, 0.  ],
                                                                   [0.  , 2.  ]]))

	def test_pow():
		def method_version():
			'''test .pow() method'''
			x = da.Var(3.0, [1, 0, 0])
			y = da.Var(1.0, [0, 1, 0])
			z = da.Var(2.0, [0, 0, 1])
			f = da.Var([2 * x, y - 1, z ** 2])
			f1 = f.pow(2)
			np.testing.assert_array_equal(f1.val, np.array([36.,  0., 16.]))
			np.testing.assert_array_equal(np.round(f1.der,2), np.array([[24.,  0.,  0.]
                                                                        [ 0.,  0.,  0.]
                                                                        [ 0.,  0., 32.]]))

			# Divides by zero when computing derivative (0^{-1/2} term)
			with np.testing.assert_raises(ZeroDivisionError):
				f_zero = f.pow(1 / 2)	

			with np.testing.assert_raises(ValueError):	
				f3 = da.Var([2 * x, y + 1, -z ** 2])
				f_negative = f3.pow(1 / 2)

				f4 = abs(f3)
				f5 = f4.pow(1/2)
				np.testing.assert_array_equal(np.round(f5.val, 2), np.array([2.45, 1.41, 2.  ]))
				np.testing.assert_array_equal(np.round(f5.der,2), np.array([[0.41, 0.  , 0.  ],
                                                                            [0.  , 0.35 , 0.  ],
                                                                            [0.  , 0.  , 1.  ]]))      
       
        # Compute same tests using ** notation
		def dunder_version():
			x = da.Var(3.0, [1, 0, 0])
			y = da.Var(1.0, [0, 1, 0])
			z = da.Var(2.0, [0, 0, 1])
			f = da.Var([2 * x, y - 1, z ** 2])
			f1 = f ** 2
			np.testing.assert_array_equal(f1.val, np.array([36.,  0., 16.]))
			np.testing.assert_array_equal(np.round(f1.der,2), np.array([[24.,  0.,  0.]
                                                                        [ 0.,  0.,  0.]
                                                                        [ 0.,  0., 32.]]))

			with np.testing.assert_raises(ZeroDivisionError):
				f_zero = f ** (1 / 2)	

			with np.testing.assert_raises(ValueError):	
				f3 = da.Var([2 * x, y + 1, -z ** 2])
				f_negative = f3 ** (1 / 2)

				f4 = abs(f3)
				f5 = f4 ** (1/2)
				np.testing.assert_array_equal(np.round(f5.val, 2), np.array([2.45, 1.41, 2.  ]))
				np.testing.assert_array_equal(np.round(f5.der,2), np.array([[0.41, 0.  , 0.  ],
                                                                            [0.  , 0.35 , 0.  ],
                                                                            [0.  , 0.  , 1.  ]]))      

			method_version()
			dunder_version()

	def test_rpow():
		x = da.Var(3.0, [1, 0, 0])
		y = da.Var(1.0, [0, 1, 0])
		z = da.Var(2.0, [0, 0, 1])
		f = da.Var([2 * x, y - 1, z ** 2])
		f1 = 2 ** f
		np.testing.assert_array_equal(np.round(f1.val,2), np.array([64., 1., 16.]))
		np.testing.assert_array_equal(np.round(f1.der,2), np.array([[88.72, 0, 0], 
																	[0, 0.69, 0], 
																	[0, 0, 44.36]]))

	def test_sqrt():
		x = da.Var(3.0, [1, 0, 0])
		y = da.Var(1.0, [0, 1, 0])
		z = da.Var(2.0, [0, 0, 1])
		f = da.Var([2 * x, y + 1, z ** 2])
		f1 = np.sqrt(f)
		np.testing.assert_array_equal(np.round(f1.val,2), np.array([2.45, 1.41, 2.  ]))
		np.testing.assert_array_equal(np.round(f1.der,2), np.array([[0.41, 0.  , 0.  ],
                                                                    [0.  , 0.35, 0.  ],
                                                                    [0.  , 0.  , 1.  ]]))
		f2 = da.Var([2 * x, y - 1, z ** 2])

		# Raise 0 to powers of 6, 0, 4
		f2_1 = 0 ** f2
		np.testing.assert_array_equal(f2_1.val, np.array([0, 1, 0]))
		np.testing.assert_array_equal(f2_1.der, np.zeros((3, 3)))

		# Raise error when raising negative number to power, since derivative is undefined
		# because it includes log(n) for n < 0
		with np.testing.assert_raises(ValueError):
			f2_2 = (-2) ** f2		

		# Raise 0 to negative power with the -(z ** 2) = -4 term
		f3 = da.Var([2 * x, y - 1, -(z ** 2)])
		with np.testing.assert_raises(ZeroDivisionError):
			f3_1 = 0 ** f3

	def test_log():
		x = da.Var(3.0, [1, 0, 0])
		y = da.Var(1.0, [0, 1, 0])
		z = da.Var(2.0, [0, 0, 1])
		f = da.Var([2 * x, y + 1, z ** 2])
		f1 = f.log(10)
		np.testing.assert_array_equal(np.round(f1.val,2), np.array([0.78, 0.3 , 0.6 ]))
		np.testing.assert_array_equal(np.round(f1.der,2), np.array([[0.14, 0.  , 0.  ],
                                                                    [0.  , 0.22, 0.  ],
                                                                    [0.  , 0.  , 0.43]]))

		with np.testing.assert_raises(ValueError):
			f2 = da.Var([2 * x, y - 1, z ** 2])
			f3 = f2.log(2)

	def test_exp():
		x = da.Var(3.0, [1, 0, 0])
		y = da.Var(1.0, [0, 1, 0])
		z = da.Var(2.0, [0, 0, 1])
		f = da.Var([2 * x, y + 1, z ** 2])
		f1 = f.exp()
		np.testing.assert_array_equal(np.round(f1.val,2), np.array([403.43,   7.39,  54.6 ]))
		np.testing.assert_array_equal(np.round(f1.der,2), np.array([[806.86,   0.  ,   0.  ],
                                                                    [  0.  ,   7.39,   0.  ],
                                                                    [  0.  ,   0.  , 218.39]]))



	# Run tests within test_DeriveAlive_vector_functions_m_to_n()
	test_neg()
	test_abs()
	test_constant()
	test_add()
	test_radd()
	test_sub()
	test_rsub()
	test_mul()
	test_rmul()
	test_truediv()
	test_rtruediv()
	test_sin()
	test_cos()
	test_tan()
	test_arcsin()
	test_arccos()
	test_arctan()
	test_sinh()
	test_cosh()
	test_tanh()
	test_sqrt()
	test_pow()
	test_rpow()
	test_log()
	test_exp()


# Without pytest, user can run these tests manually
test_DeriveAlive_Var()
test_DeriveAlive_Vec()
test_DeriveAlive_scalar_functions()
test_DeriveAlive_comparisons()
test_DeriveAlive_vector_functions_1_to_n()
test_DeriveAlive_vector_functions_m_to_1()
test_DeriveAlive_vector_functions_m_to_n()
print ("All tests passed!")