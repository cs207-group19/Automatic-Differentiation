# Module for autodifferentiation
# Harvard University, CS 207, Fall 2018
# Authors: Chen Shi, Stephen Slater, Yue Sun

import numpy as np

# Change built-in warnings to exceptions when using numpy
np.seterr(all='raise')

class Var():
	'''Builds a Var object supporting custom operations implemented below.'''
	def __init__(self, values, der=None):
		"""
		Inputs:
			values: int, float, list, or np.array -> transformed into into np.array
			der: int, float, list, or np.array -> transformed into np.array
		"""
		if isinstance(values, float) or isinstance(values, int):
			values = [values]
		if der is None:
			der = np.ones_like(values)
		elif isinstance(der, float) or isinstance(der, int):
			der = [der]
		self.val = np.array(values, dtype=float)
		self.der = np.array(der, dtype=float)

	def __repr__(self):
		return 'Var({}, {})'.format(self.val, self.der)

	def __add__(self, other):
		try:
			val = self.val + other.val
			der = self.der + other.der
		except AttributeError:
			val = self.val + other
			der = self.der
		return Var(val, der)

	def __radd__(self, other):
		# Maintain state of self and create new trace variable new_var
		new_var = Var(self.val, self.der)
		return new_var.__add__(other)

	def __sub__(self, other):
		if isinstance(other, int) or isinstance(other, float):
			# Maintain state of self and create new trace variable new_var
			new_var = Var(self.val, self.der)
			return new_var.__add__(-other)
		return (-other).__add__(self)

	def __rsub__(self, other):
		return (-self).__add__(float(other))

	def __mul__(self, other):
		try:
			val = self.val * other.val
			der = self.der * other.val + self.val * other.der
		except AttributeError:
			val = self.val * other
			der = self.der * other
		return Var(val, der)

	def __rmul__(self, other):
		# Maintain state of self and create new trace variable new_var
		new_var = Var(self.val, self.der)
		return new_var.__mul__(other)

	def __truediv__(self, other):
		try:
			val = np.divide(self.val, other.val)
			der = (np.multiply(other.val, self.der) - np.multiply(self.val, other.der)) / (other.val ** 2)
		except AttributeError:
			val = np.divide(self.val, other)
			der = np.divide(self.der, other)
		return Var(val, der)

	def __rtruediv__(self, other):
		'''Note: self contains denominator; other contains numerator'''
		try:
			val = np.divide(other.val, self.val)
			der = (np.multiply(self.val, other.der) - np.multiply(other.val, self.der)) / (np.linalg.norm(self.val) ** 2)
		except AttributeError:
			val = np.divide(other, self.val)
			der = (-np.multiply(other, self.der)) / (np.linalg.norm(self.val) ** 2)
		return Var(val, der)

	def __neg__(self):
		val = -self.val
		der = -self.der
		return Var(val, der)

	# TODO: double check derivative of abs function
	def __abs__(self):
		val = abs(self.val)
		if 0 in self.val:
			raise ValueError("Absolute value is not differentiable at 0.")
		der = np.array([-1 if x < 0 else 1 for x in self.val])
		return Var(val, der)

	def __eq__(self, other):
		try:
			return (np.array_equal(self.val, other.val) and 
					np.array_equal(self.der, other.der))
		except:
			return False

	def __pow__(self, n):
		values = map(lambda x: x >= 0, self.val)
		if n % 1 != 0 and not all(values):
			raise ValueError("Non-positive number raised to a fraction encountered in pow.")
		elif n < 1 and 0 in self.val:
			raise ZeroDivisionError("Cannot compute derivative of 0^y for y < 1.")

		val = np.power(self.val, n)
		der = n * np.multiply((self.val ** (n - 1)), self.der)
		return Var(val, der)

	def __rpow__(self, n):
		if n == 0:
			if self.val == 0:
				val = 1
				der = 0
			if self.val > 0:
				val = 0
				der = 0
			if self.val < 0:
				raise ZeroDivisionError("0.0 cannot be raised to a negative power.")
		elif n < 0:
			raise ValueError("Real numbers only, math domain error.")
		else:
			val = n ** self.val
			der = n ** self.val * np.log(n)
		return Var(val, der)

	def sin(self):
		val = np.sin(self.val)
		der = np.cos(self.val) * self.der
		return Var(val, der)

	def cos(self):
		val = np.cos(self.val)
		der = -np.sin(self.val) * self.der
		return Var(val, der)

	def tan(self):
		# Ensure that no values in self.val are of the form (pi/2 + k*pi)        
		values = map(lambda x: ((x / np.pi) - 0.5) % 1 == 0.0, self.val)
		if any(values):
			raise ValueError("Tangent not valid at pi/2, -pi/2.")
		val = np.tan(self.val)
		der = np.multiply(np.power(1 / np.cos(self.val), 2), self.der)
		return Var(val, der)

	def arcsin(self):
		values = map(lambda x: -1 <= x <= 1, self.val)
		if not all(values):
			raise ValueError("Domain of arcsin is [-1, 1].")		
		val = np.arcsin(self.val)
		der = 1 / np.sqrt(1 - (self.val ** 2))
		return Var(val, der)

	def arccos(self):
		values = map(lambda x: -1 <= x <= 1, self.val)
		if not all(values):
			raise ValueError("Domain of arccos is [-1, 1].")	
		val = np.arccos(self.val)
		der = -1 / np.sqrt(1 - (self.val ** 2))
		return Var(val, der)

	def arctan(self):
		val = np.arctan(self.val)
		der = 1 / (1 + (self.val) ** 2)
		return Var(val, der)

	def sinh(self):
		val = np.sinh(self.val)
		der = np.cosh(self.val)
		return Var(val, der)

	def cosh(self):
		val = np.cosh(self.val)
		der = np.sinh(self.val)
		return Var(val, der)

	def tanh(self):
		val = np.tanh(self.val)
		der = 1 / np.power(np.cosh(self.val), 2)
		return Var(val, der)

	def pow(self, n):
		# Maintain state of self and create new trace variable new_var
		new_var = Var(self.val, self.der)
		return new_var.__pow__(n)
    
	def log(self, base):
		values = map(lambda x: x > 0, self.val)
		if not all(values):
			raise ValueError("Non-positive number encountered in log.")
		else:
			val = np.math.log(self.val, base)
			der = np.multiply(np.divide(np.log(base), self.val), self.der)
		return Var(val, der)

	def exp(self):
		val = np.exp(self.val)
		der = np.multiply(np.exp(self.val), self.der)
		return Var(val, der)
