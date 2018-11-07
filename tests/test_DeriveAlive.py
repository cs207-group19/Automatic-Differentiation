# Test suite for DeriveAlive module

# import sys
# sys.path.append('../')
# print (sys.path) 

import pytest
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
		x = da.Var([1, 2, 3])
		np.testing.assert_array_equal(x.val, [1, 2, 3])
		assert type(x.val) == np.ndarray
		np.testing.assert_array_equal(x.der, [1, 1, 1])
		assert type(x.der) == np.ndarray

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

	test_scalar_without_bracket()
	test_scalar_with_bracket()
	test_vector_input()
	test_with_preset_der()
	test_repr()


def test_DeriveAlive_scalar_functions():
	'''Test scalar functions split up by operation type.

	Note that most tests make use of elementary operations such as addition,
	multiplications, etc., so there are not tests explicitly for these methods.
	'''

	def test_neg():
		x = da.Var(3.0)
		f = -x
		assert f.val == [-3.0]
		assert f.der == [-1.0]

		# Negate operator applied after the power
		f2 = -x.pow(2)
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

	def test_div():
		# Expect value of 1.5, derivative of 0.5
		x3 = da.Var(3.0)
		f3 = x3 / 2
		assert np.round(f3.val, 2) == [1.5]
		assert np.round(f3.der, 2) == [0.5]

	def test_rdiv():
		# Expect value of 0.5, derivative of -0.25
		x2 = da.Var(2.0)
		f2 = 1 / x2
		assert np.round(f2.val, 2) == [0.5]
		assert np.round(f2.der, 2) == [-0.25]

	def test_sin():
		# Expect value of 18.42, derivative of 6.0
		x1 = da.Var(np.pi / 2)
		f1 = 3 * 2 * x1.sin() + 2 * x1 + 4 * x1 + 3
		assert np.round(f1.val, 2) == [18.42]
		assert np.round(f1.der, 2) == [6.0]

		x11 = da.Var(np.pi)
		f11 = 3 * x11.sin() +  3
		assert np.round(f11.val, 2) == [3.0]
		assert f11.der == [-3.0]

	def test_cos():
		# Expect value of -10pi, derivative of 0.0 (because of -sin(pi))
		x = da.Var(np.pi)
		f = 5 * x.cos()+ x.cos() * 5
		assert f.val == [-10]
		assert abs(f.der) <= 1e-14

	def test_tan():
		# Expect value of 9.0, derivative of 12.0
		x4 = da.Var(np.pi / 4)
		f4 = 3 * 2 * x4.tan() + 3
		assert np.round(f4.val, 2) == [9.0]
		assert np.round(f4.der, 2) == [12.0]

		x12 = da.Var(np.pi / 4)
		f12 = 3 * 2 * x12.tan() + 3
		assert f12.val == 9.0
		assert np.round(f12.der, 2) == [12.0]

		# Tangent is undefined for multiples of pi/2 >= pi/2
		with np.testing.assert_raises(ValueError):
			x = da.Var(3 * np.pi / 2)
			f = x.tan()

	def test_arcsin():
		x = da.Var(0)
		f = x.arcsin()
		assert f.val == [0.0]
		assert f.der == [1.0]

		# Domain of arcsin(x) is -1 <= x <= 1
		with np.testing.assert_raises(ValueError):
			x = da.Var(-1.01)
			x.arcsin()

	def test_arccos():
		x = da.Var(0)
		f = x.arccos()
		assert f.val == [np.pi / 2]
		assert f.der == [-1.0]

		# Domain of arccos(x) is -1 <= x <= 1
		with np.testing.assert_raises(ValueError):
			x = da.Var(1.01)
			x.arccos()

	def test_arctan():
		x = da.Var(0.5)
		f = x.arctan()
		assert np.round(f.val, 2) == [0.46]
		assert f.der == [0.8]

	def test_sinh():
		x = da.Var(-1.0)
		f = x.sinh()
		assert np.round(f.val, 2) == [-1.18]
		assert np.round(f.der, 2) == [1.54]

	def test_cosh():
		x = da.Var(0.5)
		f = x.cosh()
		assert np.round(f.val, 2) == [1.13]
		assert np.round(f.der, 2) == [0.52]

	def test_tanh():
		x = da.Var(0.75)
		f = x.tanh()
		assert np.round(f.val, 2) == [0.64]
		assert np.round(f.der, 2) == [0.60]

	def test_pow():
		# Expect value of 64.0, derivative of 48.0
		x5 = da.Var(4.0)
		f5 = x5.pow(3)
		assert np.round(f5.val, 2) == [64.0]
		assert np.round(f5.der, 2) == [48.0]
	
		# Divides by zero when computing derivative (0^{-1/2} term)
		with np.testing.assert_raises(ZeroDivisionError):
			zero = da.Var(0)
			f_zero = zero.pow(1 / 2)

		x6 = da.Var(4.0)
		f6 = x6.pow(1 / 2)
		assert np.round(f6.val, 2) == [2.0]
		assert np.round(f6.der, 2) == [0.25]

		with np.testing.assert_raises(ValueError):
			x7 = da.Var(-2)
			f7 = x7.pow(1 / 2)

		x13 = da.Var(4.0)
		f13 = x13.pow(3)
		assert f13.val == [64.0]
		assert f13.der == [48.0]

	def test_rpow():
		pass

	def test_log():
		# Expect value of 1.0, derivative of 0.23025850929940458
		x8 = da.Var(10)
		f8 = x8.log(10)
		assert np.round(f8.val, 2) == [1.0]
		assert np.round(f8.der, 2) == [0.23]

		with np.testing.assert_raises(ValueError):
			x9 = da.Var(0)
			f9 = x9.log(2)

	def test_exp():
		# Expect value of 2.718281828459045, derivative of 2.718281828459045
		x10 = da.Var(1)
		f10 = x10.exp()
		assert np.round(f10.val, 2) == [2.72]
		assert np.round(f10.der, 2) == [2.72]


	# Run tests
	test_neg()
	test_abs()
	test_div()
	test_rdiv()
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
	test_log()
	test_exp()

test_DeriveAlive_Var()
test_DeriveAlive_scalar_functions()
