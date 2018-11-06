import sys
sys.path.append('../DeriveAlive/')

import pytest
import DeriveAlive as da
import numpy as np
import math

def test_DeriveAlive_results():
	# Expect value of 18.42, derivative of 6.0
	x1 = da.Var(np.pi / 2)
	f1 = 3 * 2 * x1.sin() + 2 * x1 + 4 * x1 + 3
	assert np.round(f1.val, 2) == [18.42]
	assert np.round(f1.der, 2) == [6.0]

	# Expect value of 0.5, derivative of -0.25
	x2 = da.Var(2.0)
	f2 = 1 / x2
	assert np.round(f2.val, 2) == [0.5]
	assert np.round(f2.der, 2) == [-0.25]

	# Expect value of 1.5, derivative of 0.5
	x3 = da.Var(3.0)
	f3 = x3 / 2
	assert np.round(f3.val, 2) == [1.5]
	assert np.round(f3.der, 2) == [0.5]

	# Expect value of 9.0, derivative of 12.0
	x4 = da.Var(np.pi / 4)
	f4 = 3 * 2 * x4.tan() + 3
	assert np.round(f4.val, 2) == [9.0]
	assert np.round(f4.der, 2) == [12.0]

	# Expect value of 64.0, derivative of 48.0
	x5 = da.Var(4.0)
	f5 = x5.pow(3)
	assert np.round(f5.val, 2) == [64.0]
	assert np.round(f5.der, 2) == [48.0]
	
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
	#assert np.round(f7.val, 2) == [float('nan')]
	#assert np.round(f7.der, 2) == [float('nan')]

	# Expect value of 1.0, derivative of 0.23025850929940458
	x8 = da.Var(10)
	f8 = x8.log(10)
	assert np.round(f8.val, 2) == [1.0]
	assert np.round(f8.der, 2) == [0.23]

	with np.testing.assert_raises(ValueError):
		x9 = da.Var(0)
		f9 = x9.log(2)

	# Expect value of 2.718281828459045, derivative of 2.718281828459045
	x10 = da.Var(1)
	f10 = x10.exp()
	assert np.round(f10.val, 2) == [2.72]
	assert np.round(f10.der, 2) == [2.72]

	x11 = da.Var(np.pi)
	f11 = 3 * x11.sin() +  3
	assert np.round(f11.val, 2) == [3.0]
	assert f11.der == [-3.0]

	x12 = da.Var(np.pi / 4)
	f12 = 3 * 2 * x12.tan() + 3
	assert f12.val == 9.0
	assert np.round(f12.der, 2) == [12.0]

	x13 = da.Var(4.0)
	f13 = x13.pow(3)
	assert f13.val == [64.0]
	assert f13.der == [48.0]

test_DeriveAlive_results()
