# Module for autodifferentiation
# Harvard University, CS 207, Fall 2018
# Authors: Chen Shi, Stephen Slater, Yue Sun

import numpy as np

# Change built-in warnings to exceptions when using numpy
np.seterr(all='raise')

class Var(object):
	"""
	Creates a Var class supporting custom operations for Automatic Differentiation (AD).

	Attributes
	==========
	val : numpy.ndarray
		  The value of user defined function(s) f evaluated at x.

	der : numpy.ndarray
		  The corresponding derivative, gradient, or Jacobian of user defined
		  functions(s). Element (i, j) in the Jacobian contains the derivative of 
		  element i of the Var with respect to input variable j.
	 """
	def __init__(self, values, der=[1]):
		"""
		INPUTS
		=======
		val : int, float, list, or np.array
			  The value of user defined function(s) f evaluated at x.

		der : int, float, list, np.array, or str, optional (default=[1])
			  The corresponding derivative, gradient, or Jacobian of user defined
			  functions(s). Element (i, j) in the Jacobian contains the derivative of 
			  element i of the Var with respect to input variable j.
			  If der is a string, it must be of the form 'x,y', where x is the number
			  of input variables, and y is the position (0-indexed) of the current variable.
			  For example, instead of defining z = Var(2, [0, 1, 0 ,0]), one could define
			  z = Var(2, '4,1'). This makes scaling easier when there are many input variables.
		
		NOTES
		=====
		PRE: 
			 - val: int, float, list, or np.array
			 - der: int, float, list, or np.array
		POST:
			 - val: np.array
			 - der: np.array

		EXAMPLES
		=========
		# Input a constant
		>>> Var(3.0, None) 
		Var([3.], None)

		# Input a scalar variable
		>>> Var(3.0) 
		Var([3.], [1])

		# Input a vector with two elements
		>>> x = Var(3.0, [1, 0])
		>>> y = Var(3.0, [0, 1])
		>>> z = Var([x, y])
		>>> z
		Values:
		[3. 3.],
		Jacobian:
		[[1 0]
		 [0 1]]
		"""
		# Convert string input to unit vector
		if isinstance(der, str):
			assert ',' in der
			length, pos = map(int, der.split(','))
			assert pos < length
			der = np.zeros((length))
			der[pos] = 1

		if isinstance(values, float) or isinstance(values, int):
			values = [values]
		if len(values) == 1:
			if isinstance(values[0], Var):
				self.val = values[0].val
				self.der = values[0].der
			else:
				if isinstance(der, float) or isinstance(der, int):
					der = [der]

				self.val = np.array(values)
				self.der = np.array(der)
		else:
			all_non_Vars = len(list(filter(lambda x: isinstance(x, Var), values))) == 0
			if all_non_Vars:
				self.val = np.array(values)
				self.der = np.array(der)
			else:
				# self.values is a vector that contains a Var object at initialization
				max_num_vars = max([
					len(x.der[0]) if isinstance(x, Var) and len(x.der.shape) > 1 else
					(len(x.der) if isinstance(x, Var) and len(x.der.shape) == 1 else 0) for x in values])
				new_values = []
				new_derivatives = []
				for x in values:
					try:
						new_values.append(x.val)
						new_derivatives.append(x.der)
					except:
						new_values.append(x)
						new_derivatives.append(np.zeros(max_num_vars))

				self.val = np.hstack((new_values))
				self.der = np.vstack((new_derivatives))

		# Convert values of -0.0 to 0.0 in self.val
		if len(self.val.shape):
			shape = self.val.shape
			val_vals = self.val.flatten()
			for i, val_val in enumerate(val_vals):
				if val_val == -0.0:
					val_vals[i] = 0.0
			self.val = np.reshape(val_vals, shape)

		# Convert values of -0.0 to 0.0 in self.der
		if len(self.der.shape):
			shape = self.der.shape
			der_vals = self.der.flatten()
			for i, der_val in enumerate(der_vals):
				if der_val == -0.0:
					der_vals[i] = 0.0
			self.der = np.reshape(der_vals, shape)

	def __repr__(self):
		"""
		Prints self in the form of Var([val], [der]) when self is a scalar or constant.
		Prints self in the form of Values([val]) Jacobian([der]) when self is a vector.
		
		Returns
		=======
		x: Var object with val if x is a constant
		   Var object with val and der if x is a scalar
		   Var object with val and der if x is a vector
		
		Examples
		========
		# Input a constant
		>>> x = Var(3.0, None) 
		>>> print(x)
		Var([3.], None)

		# Input a scalar
		>>> x = Var(3.0) 
		>>> print(x)
		Var([3.], [1])

		# Input a vector with two elements
		>>> x = Var(3.0, [1, 0])
		>>> y = Var(3.0, [0, 1])
		>>> z = Var([x, y])
		>>> print(z)
		Values:
		[3. 3.],
		Jacobian:
		[[1 0]
		 [0 1]]
		"""

		val = val_copy = np.copy(self.val)
		der = der_copy = np.copy(self.der)

		# Convert values of -0.0 or <1e-12 to 0.0 in self.val
		if len(val_copy.shape):
			shape = val_copy.shape
			val = val_copy.flatten()
			for i, val_i in enumerate(val):
				if val_i == -0.0 or abs(val_i) < 1e-12:
					val[i] = 0.0
			val = np.reshape(val, shape)

		# Convert values of -0.0 or <1e-12 to 0.0 in self.der
		if len(der_copy.shape):
			shape = der_copy.shape
			der = der_copy.flatten()
			for i, der_i in enumerate(der):
				if der_i == -0.0 or abs(der_i) < 1e-12:
					der[i] = 0.0
			der = np.reshape(der, shape)
		
		if len(val) == 1:
			return 'Var({}, {})'.format(val, der)
		return 'Values:\n{},\nJacobian:\n{}'.format(val, der)

	def __add__(self, other):
		""" 
		Returns the addition of self and other.
		
		Parameters
		==========
		self: Var object
		other: Var object, float, or int
		
		Returns
		======== 
		z: Var object that is the sum of self and other
		
		Examples
		======== 
		>>> z = Var(3, None) + 2
		>>> print(z)
		Var([5], None)

		>>> z = Var(3) + Var(4)
		>>> print(z)
		Var([7], [2])

		>>> z = Var(3) + 2
		>>> print(z)
		Var([5], [1])
		
		>>> z = Var(3, [1, 0]) + Var(4, [0, 1])
		>>> print(z)
		Var([7], [1 1])
		"""
		try:
			val = self.val + other.val

			# Handle case when self.der or other.der contains None 
			# i.e. self or other is a vector of scalars, not of Vars
			len_self_der_shape = len(self.der.shape)
			len_other_der_shape = len(other.der.shape)

			if not len_self_der_shape and len_other_der_shape:
				der = other.der
			elif len_self_der_shape and not len_other_der_shape:
				der = self.der
			elif not len_self_der_shape and not len_other_der_shape:
				der = None
			else:
				der = self.der + other.der
		except AttributeError:
			val = self.val + other
			der = self.der
		return Var(val, der)

	def __radd__(self, other):
		"""
		Returns the addition of other to self.

		Parameters
		==========
		self: Var object
		other: Var object, float, or int
		
		Returns
		======== 
		z: Var object that is the sum of other and self
		
		Examples
		======== 
		>>> z = 2 + Var(3, None)
		>>> print(z)
		Var([5], None)

		>>> z = 2 + Var(3)
		>>> print(z)
		Var([5], [1])
		"""

		# Maintain state of self and create new trace variable new_var
		new_var = Var(self.val, self.der)
		return new_var.__add__(other)

	def __sub__(self, other):
		"""
		Returns the substraction of other from self.

		Parameters
		==========
		self: Var object
		other: Var object, float, or int
		
		Returns
		========
		z: Var object that is the difference between self and other
		
		Examples
		======== 
		>>> z = Var(3, None) - 2
		>>> print(z)
		Var([1], None)

		>>> z = Var(3) - 2
		>>> print(z)
		Var([1], [1])
		
		>>> z = Var(3) - Var(4)
		>>> print(z)
		Var([-1], [0])
		
		>>> z = Var(3, [1, 0]) - Var(4, [0, 1])
		>>> print(z)
		Var([-1], [ 1 -1])
		"""
		if isinstance(other, int) or isinstance(other, float):
			# Maintain state of self and create new trace variable new_var
			new_var = Var(self.val, self.der)
			return new_var.__add__(-other)
		return (-other).__add__(self)

	def __rsub__(self, other):
		"""
		Returns the subtraction of self from other.

		Parameters
		==========
		self: Var object
		other: Var object, float, or int
		
		Returns
		========
		z: Var object that is the difference between other and self
		
		Examples
		======== 
		>>> z = 2.1 - Var(3, None)
		>>> print(z)
		Var([-0.9], None)

		>>> z = 2 - Var(3)
		>>> print(z)
		Var([-1.], [-1])
		"""
		return (-self).__add__(float(other))

	def __mul__(self, other):
		""" 
		Returns the product of self and other.
		
		Parameters
		==========
		self: Var object
		other: Var object, float, or int

		Returns
		========== 
		z: Var object that is the product of self and other
		
		Examples
		========= 
		>>> x = Var(3.0, None)
		>>> y = Var(2.0, None)
		>>> z = x * y
		>>> print(z)
		Var([6.], None)

		>>> x = Var(3.0)
		>>> y = Var(2.0)
		>>> z = x * y
		>>> print(z)
		Var([6.], [5.])

		>>> x = Var(3.0, [1, 0])
		>>> y = Var(2, [0, 1])
		>>> z = x * y
		>>> print(z)
		Var([6.], [2. 3.])
		"""

		# Check if self.der is an array containing None
		len_self_der_shape = len(self.der.shape)

		try:
			len_other_der_shape = len(other.der.shape)
			self_val = np.expand_dims(self.val, 1) if len(other.der.shape) > 1 else self.val
			other_val = np.expand_dims(other.val, 1) if len(self.der.shape) > 1 else other.val
			val = np.multiply(self.val, other.val)
			
			# Handle case when self.der or other.der contains None 
			# i.e. self or other is a vector of scalars, not of Vars
			if not len_self_der_shape and len_other_der_shape:
				der = np.multiply(self_val, other.der)
			elif len_self_der_shape and not len_other_der_shape:
				der = np.multiply(other_val, self.der)
			elif not len_self_der_shape and not len_other_der_shape:
				der = None
			else:
				p1 = np.multiply(other_val, self.der) 
				p2 = np.multiply(self_val, other.der)

				len_p1_shape = len(p1.shape)
				len_p2_shape = len(p2.shape)
				if len_p1_shape > len_p2_shape:
					if len_p1_shape == 2 and p1.shape[1] > 1:
						p2 = np.tile(p2, (p1.shape[0], 1))
					else:
						p2 = np.expand_dims(p2, 1)
				if len_p2_shape > len_p1_shape:
					if len_p2_shape == 2 and p2.shape[1] > 1:
						p1 = np.tile(p1, (p2.shape[1], 1))
					else:
						p1 = np.expand_dims(p1, 1)
				der = p1 + p2
		except AttributeError:
			val = self.val * other
			if isinstance(other, float) or isinstance(other, int) or np.array_equal(self.der.shape, other.shape):
				der = self.der * other if len_self_der_shape else None

			# other is a numpy array or Var of scalars, not of Vars
			else:
				other_val = np.expand_dims(other, 1) if len_self_der_shape > len(other.shape) else other
				der = self.der * other_val if len_self_der_shape else None
		return Var(val, der)

	def __rmul__(self, other):
		""" 
		Returns the product of other and self.
		
		Parameters
		==========
		self: Var object
		other: Var object, float, or int

		Returns
		======= 
		z: Var object that is the product of other and self
		
		Examples
		========= 
		>>> x = Var(5.0)
		>>> z = 2 * x
		>>> print(z)
		Var([10.], [2])

		>>> x = Var(3.0, [1, 0, 0])
		>>> y = Var(1.0, [0, 1, 0])
		>>> w = Var(2.0, [0, 0, 1])
		>>> z = x + y ** 2 + x * w
		>>> print(z)
		Var([10.], [3. 2. 3.])
		"""
		# Maintain state of self and create new trace variable new_var
		new_var = Var(self.val, self.der)
		return new_var.__mul__(other)

	def __truediv__(self, other):
		""" 
		Returns the division of self by other.
		
		Parameters
		========== 
		self: Var object
		other: Var object, float, or int
		
		Returns
		=======  
		z: Var object that is the division of self and other
		
		Examples
		========  
		>>> x = Var(3.0)
		>>> z = x / 2
		>>> print(z)
		Var([1.5], [0.5])

		>>> x = Var(3.0, [1, 0, 0])
		>>> y = Var(1.0, [0, 1, 0])
		>>> w = Var(2.0, [0, 0, 1])
		>>> z = (x + y ** 2 + x * w)/2
		>>> print(z)
		Var([5.], [1.5 1.  1.5])
		"""
		# Handle case when self.der or other.der is None
		other_is_scalar = isinstance(other, float) or isinstance(other, int)
		other_is_numpy = isinstance(other, np.ndarray)
		len_self_der_shape = len(self.der.shape)

		if (other_is_scalar and other == 0) or (other_is_numpy and 0 in other) or (
			not (other_is_scalar or other_is_numpy) and 0 in other.val):
			raise ZeroDivisionError

		try:
			is_vec = len(self.val) > 1
			len_other_der_shape = len(other.der.shape)
			val = np.divide(self.val, other.val)
			self_val = np.expand_dims(self.val, 1) if len_other_der_shape > 1 else self.val
			other_val = np.expand_dims(other.val, 1) if len_self_der_shape > 1 else other.val

			# Handle case when self.der or other.der contains None 
			# i.e. self or other is a vector of scalars, not of Vars
			if not len_self_der_shape and len_other_der_shape:
				num = -np.multiply(self_val, other.der)
			elif len_self_der_shape and not len_other_der_shape:
				num = np.multiply(other_val, self.der)
			elif not len_self_der_shape and not len_other_der_shape:
				num = None
			else:
				num = np.multiply(other_val, self.der) - np.multiply(self_val, other.der)

			if num is not None:
				denom = other_val ** 2 if len(num.shape) > 1 else other.val ** 2
				if len(num.shape) > len(denom.shape):
					denom = np.expand_dims(denom, 1)

				der = np.divide(num, denom)
			else:
				der = None

		except AttributeError:
			val = np.divide(self.val, other)
			if isinstance(other, float) or isinstance(other, int) or np.array_equal(self.der.shape, other.shape):
				der = np.divide(self.der, other) if len_self_der_shape else None

			# other is a numpy array or Var of scalars, not of Vars
			else:
				other_val = np.expand_dims(other, 1) if len_self_der_shape > len(other.shape) else other
				der = np.divide(self.der, other_val) if len_self_der_shape else None

		return Var(val, der)

	def __rtruediv__(self, other):
		""" 
		Returns the division of other and self.

		Note: self contains denominator (Var); other contains numerator.
		
		Parameters
		==========
		self: Var object
		other: Var object, float, or int
		
		Returns
		======= 
		z: Var object that is the division of self and other
		
		Examples
		========   
		>>> x = Var(2.0)
		>>> z = 1 / x
		>>> print(z)
		Var([0.5], [-0.25])

		>>> x = Var(3.0)
		>>> a = Var([ 1., x, x, 4.])
		>>> z = 3 / a
		>>> print(z)
		Values:
		[3.   1.   1.   0.75],
		Jacobian:
		[[ 0.        ]
		 [-0.33333333]
		 [-0.33333333]
		 [ 0.        ]]
		"""

		# Check for ZeroDivisionError at start rather than nesting exception block
		if isinstance(other, np.ndarray):
			other = Var(other)

		if (self == 0 or 
			not (isinstance(self, float) or isinstance(self, int)) and not all(self.val)):
			raise ZeroDivisionError
			
		val = np.divide(other, self.val)

		if isinstance(other, float) or isinstance(other, int):
			if len(self.der.shape):
				num = (-np.multiply(other, self.der))
				self_val = np.expand_dims(self.val, 1) if len(num.shape) > len(self.val.shape) else self.val
				der = num / (self_val ** 2)
			else:
				der = None

		# other is a numpy array or Var of scalars, not of Vars
		else:
			other_val = np.expand_dims(other, 1) if len(self.der.shape) > len(other.shape) else other
			num = (-np.multiply(other_val, self.der))
			self_val = np.expand_dims(self.val, 1) if len(num.shape) > len(self.val.shape) else self.val
			der = num / (self_val ** 2) 

		return Var(val, der)

	def __neg__(self):
		""" 
		Returns negation of self.

		Parameters
		==========
		self: Var object
		
		Returns
		======= 
		z: Var object with val = -self.val, der = -self.der

		Examples
		======== 
		>>> x = Var(3.0)
		>>> z = -x
		>>> print(z)
		Var([-3.], [-1])
		"""
		val = -self.val
		der = -self.der if len(self.der.shape) else None
		return Var(val, der)

	def __abs__(self):
		""" 
		Returns the absolute value of self and the updated derivative.

		Parameters
		==========
		self: Var object
		
		Returns
		======= 
		z: Var object with val = abs(self.val), der = -self.der

		Examples
		======== 
		>>> x = Var(-4.0)
		>>> z = abs(x)
		>>> print(z)
		Var([4.], [-1])
		"""
		val = abs(self.val)
		if 0 in self.val:
			raise ValueError("Absolute value is not differentiable at 0.")

		der_copy = np.copy(self.der)
		if len(der_copy.shape):
			for i, val_i in enumerate(self.val):
				if val_i < 0:
					der_copy[i] = -1 * der_copy[i]
		return Var(val, der_copy)

	def __eq__(self, other):
		""" 
		Check if self and other are equal, meaning same values and derivative.
		
		Parameters
		==========
		self: Var object
		other: Var object
		
		Returns
		======= 
		Boolean: True if self == other, otherwise False

		Examples
		======== 
		>>> x = Var(3)
		>>> y = Var(3)
		>>> x == y
		True
		"""
		try:
			return (np.array_equal(self.val, other.val) and 
					np.array_equal(self.der, other.der))
		except:
			# Compare scalar Vars with derivative 1 to scalars
			if len(self.val) == 1 and np.array_equal(self.der, [1.]):
				return self.val == other
			return False

	def __ne__(self, other):
		""" 
		Check self and other is not equal

		Parameters
		==========
		self: Var object
		other: Var object
		
		Returns
		======= 
		Boolean: True if self != other, otherwise False

		Examples
		======== 
		>>> x = Var(3)
		>>> y = Var(5)
		>>> x != y
		True
		"""
		return not self.__eq__(other)

	def __lt__(self, other):
		""" 
		Check whether the value self is less than other 

		Parameters
		==========
		self: Var object
		other: Var object, int, float
		
		Returns
		======= 
		Boolean: True if self.val < other.val, otherwise False

		Examples
		======== 
		>>> x = Var(3)
		>>> x < 4
		array([ True])

		>>> x = Var(3)
		>>> y = Var(5)
		>>> x < y
		array([ True])
		"""
		# Numpy internally checks if the dimensions of self and other match
		try:
			return self.val < other.val
		except:
			return self.val < other

	def __le__(self, other):
		""" 
		Check whether the value self is less than or equal to other 

		Parameters
		==========
		self: Var object
		other: Var object, int, float
		
		Returns
		======= 
		Boolean: True if self.val <= other.val, otherwise False

		Examples
		======== 
		>>> x = Var(3)
		>>> x <= 3
		array([ True])
		"""
		return self.__lt__(other) or self.__eq__(other)

	def __gt__(self, other):
		""" 
		Check whether the value self is greater than other 

		Parameters
		==========
		self: Var object
		other: Var object, int, float
		
		Returns
		======= 
		Boolean: True if self.val > other.val, otherwise False

		Examples
		======== 
		>>> x = Var(3)
		>>> x > 2
		array([ True])

		>>> x = Var(3)
		>>> y = Var(1)
		>>> x > y
		array([ True])
		"""
		try:
			return self.val > other.val
		except:
			return self.val > other

	def __ge__(self, other):
		""" 
		Check whether the value self is larger than or equal to other 

		Parameters
		==========
		self: Var object
		other: Var object, int, float
		
		Returns
		======= 
		Boolean: True if self.val >= other.val, otherwise False

		Examples
		======== 
		>>> x = Var(3)
		>>> x >= 3
		array([ True])
		"""
		return self.__gt__(other) or self.__eq__(other)

	def __pow__(self, n):
		""" 
		Return power calculation of Var object in the form of Var object ** 2

		Parameters
		==========
		self: Var object
		n: real number
		
		Returns
		=======
		z: Var object that is self raised to the power n
		
		Examples
		========= 
		>>> x = Var(4)
		>>> z = x ** 2
		>>> print(z)
		Var([16], [8])

		>>> x = Var(3.0, [1, 0, 0])
		>>> y = Var(1.0, [0, 1, 0])
		>>> z = Var(2.0, [0, 0, 1])
		>>> f = Var([2 * x, y - 1, z ** 2])
		>>> z = f.pow(2)
		>>> print(z)
		Values:
		[36.  0. 16.],
		Jacobian:
		[[24.  0.  0.]
		 [ 0.  0.  0.]
		 [ 0.  0. 32.]]
		"""
		values = map(lambda x: x >= 0, self.val)
		if isinstance(n, float) or isinstance(n, int):
			if n % 1 != 0 and not all(values):
				raise ValueError("Non-positive number raised to a fraction encountered in pow.")
			elif n < 1 and 0 in self.val:
				raise ZeroDivisionError("Cannot compute derivative of 0^y for y < 1.")
		
			val = np.power(self.val, n)
			if len(self.der.shape):
				self_val = np.expand_dims(self.val, 1) if len(self.der.shape) > len(self.val.shape) else self.val
				der = n * np.multiply((self_val ** (n - 1)), self.der)
			else:
				der = None

			return Var(val, der)

		# n is a Var
		else:
			for n_i in n.val:
				if n_i % 1 != 0 and not all(values):
					raise ValueError("Non-positive number raised to a fraction encountered in pow.")
				elif n_i < 1 and 0 in self.val:					
					raise ZeroDivisionError("Cannot compute derivative of 0^y for y < 1.")

			val = np.power(self.val, n.val)
			# Check for constant Vars of scalars or vectors
			if len(n.der.shape) == 0 and len(self.der.shape):
				der = None
			elif len(self.der.shape) == 0 and len(n.der.shape):
				der = None
			elif len(self.der.shape) == 0 and len(n.der.shape) == 0:
				der = None
			else:
				# Both self and n are Vars of valid variable(s), not constants
				if len(self.val) > 1 and len(n.val) > 1 and len(self.val) != len(n.val):
					raise ValueError("x and y cannot be of different dimensions > 1 in x^y.")
				else:
					der = val * ((n.val / self.val) * self.der + np.log(self.val) * n.der)
			return Var(val, der)

	def __rpow__(self, n):
		""" 
		Return power calculation of n to the power of the Var object value(s) in the form of n ** Var

		Parameters
		==========
		self: Var object
		n: real number
		
		Returns
		=======
		z: Var object that is n raised to the power self Var object
		
		Examples
		========= 
		>>> x = Var(4)
		>>> z = 2 ** x
		>>> print(z)
		Var([16], [11.09035489])
		"""
		if n == 0:
			if len(self.val) == 1:
				if self.val == 0:
					val = 1
					der = 0
				elif self.val > 0:
					val = 0
					der = 0
				elif self.val < 0:
					raise ZeroDivisionError("0.0 cannot be raised to a negative power.")
			else:
				val = np.zeros((self.val.shape))
				der = np.zeros((self.der.shape))
				for i, val_i in enumerate(self.val):
					if val_i == 0:
						val[i] = 1
					elif val_i > 0:
						val[i] = 0
					elif val_i < 0:
						raise ZeroDivisionError("0.0 cannot be raised to a negative power.")
		elif n < 0:
			raise ValueError("Real numbers only, math domain error.")
		else:
			val = n ** self.val
			if len(self.der.shape):
				der = np.expand_dims(n ** self.val * np.log(n), 1) * self.der if len(self.der.shape) > 1 else n ** self.val * np.log(n) * self.der
			else:
				der = None

		return Var(val, der)

	def sin(self):
		""" 
		Returns the sine of Var object.
		
		Parameters
		==========
		self: Var object
		
		Returns
		========= 
		z: sine of self
		
		Examples
		========= 
		>>> import numpy as np
		>>> x = Var(np.pi / 2, [1, 0])
		>>> y = Var(0, [0, 1])
		>>> z = 3 * np.sin(x) + 2 * np.sin(y)
		>>> print(z)
		Var([3.], [0. 2.])
		"""
		val = np.sin(self.val)
		if len(self.der.shape):
			to_multiply = np.cos(self.val)
			to_multiply = np.expand_dims(to_multiply, 1) if len(self.der.shape) > len(to_multiply.shape) else to_multiply
			der = to_multiply * self.der
		else:
			der = None
		return Var(val, der)

	def cos(self):
		""" 
		Returns the cosine of Var object.
		
		Parameters
		==========
		self: Var object
		
		Returns
		========= 
		z: cosine of self
		
		Examples
		========= 
		>>> import numpy as np
		>>> x = Var(np.pi / 2, [1, 0])
		>>> y = Var(0, [0, 1])
		>>> z = 3 * np.cos(x) + 2 * np.cos(y)
		>>> print(z)
		Var([2.], [-3.  0.])
		"""
		val = np.cos(self.val)
		if len(self.der.shape):
			to_multiply = -np.sin(self.val)
			to_multiply = np.expand_dims(to_multiply, 1) if len(self.der.shape) > len(to_multiply.shape) else to_multiply
			der = to_multiply * self.der
		else:
			der = None
		return Var(val, der)

	def tan(self):
		""" 
		Returns the tangent of Var object.
		
		Parameters
		==========
		self: Var object
		
		Returns
		========= 
		z: tangent of self
		
		Examples
		========= 
		>>> import numpy as np
		>>> x = Var(np.pi / 4, [1, 0])
		>>> y = Var(np.pi / 3, [0, 1])
		>>> z = 3 * np.tan(x) + 2 * np.tan(y) 
		>>> print(z)
		Var([6.46410162], [6. 8.])
		"""
		# Ensure that no values in self.val are of the form (pi/2 + k*pi)        
		values = map(lambda x: ((x / np.pi) - 0.5) % 1 == 0.0, self.val)
		if any(values):
			raise ValueError("Tangent not valid at pi/2, -pi/2.")
		val = np.tan(self.val)
		if len(self.der.shape):
			to_multiply = np.power(1 / np.cos(self.val), 2)
			to_multiply = np.expand_dims(to_multiply, 1) if len(self.der.shape) > len(to_multiply.shape) else to_multiply
			der = np.multiply(to_multiply, self.der)
		else:
			der = None
		return Var(val, der)

	def arcsin(self):
		""" 
		Returns the arcsine of Var object.
		
		Parameters
		==========
		self: Var object
		
		Returns
		========= 
		z: arcsine of self
		
		Examples
		========= 
		>>> import numpy as np
		>>> x = Var(0, [1, 0])
		>>> y = Var(0, [0, 1])
		>>> z = 3 * np.arcsin(x) + 2 * np.arcsin(y)
		>>> print(z)
		Var([0.], [3. 2.])
		"""
		values = map(lambda x: -1 <= x <= 1, self.val)
		if not all(values):
			raise ValueError("Domain of arcsin is [-1, 1].")		
		val = np.arcsin(self.val)
		if len(self.der.shape):
			if self.val == 1:
				to_multiply = np.nan
			elif self.val == -1:
				to_multiply = np.nan
			else:
				to_multiply = 1 / np.sqrt(1 - (self.val ** 2))
				to_multiply = np.expand_dims(to_multiply, 1) if len(self.der.shape) > len(to_multiply.shape) else to_multiply
			der = to_multiply * self.der
		else:
			der = None
		return Var(val, der)	

	def arccos(self):
		""" 
		Returns the arccosine of Var object.
		
		Parameters
		==========
		self: Var object
		
		Returns
		========= 
		z: arccosine of self
		
		Examples
		========= 
		>>> import numpy as np
		>>> x = Var(0, [1, 0])
		>>> y = Var(0.5, [0, 1])
		>>> z = 3 * np.arccos(x) + 2 * np.arccos(y)
		>>> print(z)
		Var([6.80678408], [-3.         -2.30940108])
		"""
		values = map(lambda x: -1 <= x <= 1, self.val)
		if not all(values):
			raise ValueError("Domain of arccos is [-1, 1].")	
		val = np.arccos(self.val)
		if len(self.der.shape):
			if self.val == 1:
				to_multiply = np.nan
			elif self.val == -1:
				to_multiply = np.nan
			else:
				to_multiply = -1 / np.sqrt(1 - (self.val ** 2))
				to_multiply = np.expand_dims(to_multiply, 1) if len(self.der.shape) > len(to_multiply.shape) else to_multiply
			der = to_multiply * self.der
		else:
			der = None
		return Var(val, der)

	def arctan(self):
		""" 
		Returns the arctan of Var object.
		
		Parameters
		==========
		self: Var object
		
		Returns
		========= 
		z: arctan of self
		
		Examples
		========= 
		>>> import numpy as np
		>>> x = Var(0.5, [1, 0])
		>>> y = Var(np.pi/2, [0, 1])
		>>> z = 3 * np.arctan(x) + 2 * np.arctan(y)
		>>> print(z)
		Var([3.39871247], [2.4        0.57680088])
		"""		
		val = np.arctan(self.val)
		if len(self.der.shape):
			to_multiply = 1 / (1 + (self.val) ** 2)
			to_multiply = np.expand_dims(to_multiply, 1) if len(self.der.shape) > len(to_multiply.shape) else to_multiply
			der = to_multiply * self.der
		else:
			der = None
		return Var(val, der)		

	def sinh(self):
		""" 
		Returns the sinh of Var object.
		
		Parameters
		==========
		self: Var object
		
		Returns
		========= 
		z: sinh of self
		
		Examples
		========= 
		>>> x = Var(-1, [1, 0])
		>>> y = Var(0, [0, 1])
		>>> z = 3 * np.sinh(x) + 2 * np.sinh(y)
		>>> print(z)
		Var([-3.52560358], [4.6292419 2.       ])
		"""		
		val = np.sinh(self.val)
		if len(self.der.shape):
			to_multiply = np.cosh(self.val)
			to_multiply = np.expand_dims(to_multiply, 1) if len(self.der.shape) > len(to_multiply.shape) else to_multiply
			der = to_multiply * self.der
		else:
			der = None
		return Var(val, der)	

	def cosh(self):
		""" 
		Returns the cosh of Var object.
		
		Parameters
		==========
		self: Var object
		
		Returns
		========= 
		z: cosh of self
		
		Examples
		========= 
		>>> x = Var(-1, [1, 0])
		>>> y = Var(0, [0, 1])
		>>> z = 3 * np.cosh(x) + 2 * np.cosh(y)
		>>> print(z)
		Var([6.6292419], [-3.52560358  0.        ])
		"""		
		val = np.cosh(self.val)
		if len(self.der.shape):
			to_multiply = np.sinh(self.val)
			to_multiply = np.expand_dims(to_multiply, 1) if len(self.der.shape) > len(to_multiply.shape) else to_multiply
			der = to_multiply * self.der
		else:
			der = None
		return Var(val, der)	

	def tanh(self):
		""" 
		Returns the tanh of Var object.
		
		Parameters
		==========
		self: Var object
		
		Returns
		========= 
		z: tanh of self
		
		Examples
		========= 
		>>> x = Var(-1, [1, 0])
		>>> y = Var(0, [0, 1])
		>>> z = 3 * np.tanh(x) + 2 * np.tanh(y)
		>>> print(z)
		Var([-2.28478247], [1.25992302 2.        ])
		"""				
		val = np.tanh(self.val)
		if len(self.der.shape):
			to_multiply = 1 / np.power(np.cosh(self.val), 2)
			to_multiply = np.expand_dims(to_multiply, 1) if len(self.der.shape) > len(to_multiply.shape) else to_multiply
			der = to_multiply * self.der
		else:
			der = None
		return Var(val, der)	

	def pow(self, n):
		""" 
		Return power calculation of self to the power n in the form of self.pow(n)

		Parameters
		==========
		self: Var object
		n: real number
		
		Returns
		=======
		z: Var object that is self raised to the power n
		
		Examples
		========= 
		>>> x = Var(4)
		>>> z = x.pow(2)
		>>> print(z)
		Var([16], [8])
		"""
		# Maintain state of self and create new trace variable new_var
		new_var = Var(self.val, self.der)
		return new_var.__pow__(n)
 
	def sqrt(self):
		""" 
		Return square root of self.

		Parameters
		==========
		self: Var object
		
		Returns
		=======
		z: Var object that is self raised to the power n
		
		Examples
		========= 
		>>> x = Var(4)
		>>> z = np.sqrt(x)
		>>> print(z)
		Var([2.], [0.25])
		"""

		# Maintain state of self and create new trace variable new_var
		new_var = Var(self.val, self.der)
		return new_var.__pow__(0.5)

	def log(self, base):
		""" 
		Return the log (with user-specified base) of self.

		Parameters
		==========
		self: Var object
		base: int
		
		Returns
		=======
		z: Var object that is log (with user-specified base) of self 

		NOTES
		======
		Use as Var.log(base) with customer defined base 

		Examples
		========= 
		>>> x = Var(10)
		>>> z =  x.log(10)
		>>> print(z)
		Var([1.], [0.04342945])
		"""

		values = map(lambda x: x > 0, self.val)
		if not all(values):
			raise ValueError("Non-positive number encountered in log.")
		else:
			val = np.array([np.math.log(v, base) for v in self.val])
			if len(self.der.shape):
				to_multiply = 1 / np.multiply(np.log(base), self.val)
				to_multiply = np.expand_dims(to_multiply, 1) if len(self.der.shape) > len(to_multiply.shape) else to_multiply
				der = np.multiply(to_multiply, self.der)
			else:
				der = None
		return Var(val, der)

	def exp(self):
		""" 
		Return the exponential of self.

		Parameters
		==========
		self: Var object
		
		Returns
		=======
		z: Var object that is the exponetial of self 

		Examples
		========= 
		>>> x = Var(1)
		>>> z = np.exp(x)
		>>> print(z)
		Var([2.71828183], [2.71828183])
		"""
		val = np.exp(self.val)
		if len(self.der.shape):
			to_multiply = np.exp(self.val)
			to_multiply = np.expand_dims(to_multiply, 1) if len(self.der.shape) > len(to_multiply.shape) else to_multiply
			der = np.multiply(to_multiply, self.der)
		else:
			der = None
		return Var(val, der)

	def logistic(self):
		""" 
		Return the logistic function evaluation (sigmoid): f(x) = 1 / (1 + e^{-x})

		Parameters
		==========
		self: Var object
		
		Returns
		=======
		z: Var object that is the logistic evaluation of self

		Examples
		========= 
		# Scalar variable
		>>> x = Var(1)
		>>> z = x.logistic()
		>>> print(z)
		Var([0.73105858], [0.19661193])

		# Vector of constants
		>>> y = Var([1, 2, 3, 4], None)
		>>> w = y.logistic()
		>>> print(w)
		Values:
		[0.73105858 0.88079708 0.95257413 0.98201379],
		Jacobian:
		None
		"""

		# Case for constant scalar or vector with no derivative
		if len(self.der.shape) == 0:
			val = 1 / (1 + np.exp(-self.val))
			der = None
			return Var(val, der)
		
		val = 1 / (1 + np.exp(-self.val))
		der = np.exp(self.val) / ((1 + np.exp(self.val)) ** 2)

		if len(self.der.shape) > 1:
			new_der = np.reshape(der, [-1, 1])
		else:
			new_der = der
		
		final_der = new_der * self.der
		return Var(val, final_der)
