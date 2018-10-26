import numpy as np   
from math import log
import warnings
warnings.filterwarnings('error')


class Var(object):
	def __init__(self, a, der=1.0):
		self.val = a
		self.der = der
	
	def __add__(self, other):
		val = self.val
		der = self.der
		try:
			val += other.val
			der += other.der
		except AttributeError:
			val += other
		return Var(val, der)
	
	def __radd__(self, other):
		return self.__add__(other)
		
	def __mul__(self, other):
		val = self.val
		der = self.der
		try:
			val *= other.val
			der *= other.val
		except AttributeError:
			val *= other
			der *= other
		return Var(val, der)

	def __rmul__(self, other):
		return self.__mul__(other)

	def __truediv__(self, other):
		val = self.val
		der = self.der
		try:
			val /= other.val
			der = (other.val * self.der - val * other.der) / (other.val ** 2)
		except AttributeError:
			val /= other
			der /= other
		return Var(val, der)

	def __rtruediv__(self, other):
		'''Note: self contains denominator; other contains numerator'''
		try:
			val = other.val / self.val
			der = (self.val * other.der - other.val * self.der) / (self.val ** 2)
		except AttributeError:
			val = other / self.val
			der = (self.val * 0 - other * self.der) / (self.val ** 2)
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
		val = np.tan(self.val)
		der = 1/np.cos(self.val) ** 2 * self.der
		return Var(val, der)


	def pow(self, n):
		try:
			val = np.power(self.val, n)
			der = n * (self.val ** (n - 1)) * self.der
		except ZeroDivisionError:
			raise
			val = 0
			der = float('nan')
		except Warning:
			val = float('nan')
			der = float('nan')
		return Var(val, der)


	def log(self, base):
		if self.val == 0:
			return float('inf')
		elif self.val < 0 :
			return float('nan')
		else:
			val = log(self.val, base) 
			der = log(base)/self.val * self.der
		return Var(val, der)

	def exp(self):
		val = np.exp(self.val) 
		der = np.exp(self.val) * self.der
		return Var(val, der)			


# Expect value of 18.42, derivative of 6.0
x1 = Var(np.pi / 2)
f1 = 3 * 2 * x1.sin() + 2 * x1 + 4 * x1 +  3
print (f1.val, f1.der)

# Expect value of 0.5, derivative of -0.25
x2 = Var(2.0)
f2 = 1 / x2
print (f2.val, f2.der)

# Expect value of 1.5, derivative of 0.5
x3 = Var(3.0)
f3 = x3 / 2
print (f3.val, f3.der)


# Expect value of 9.0, derivative of 12.0
x4 = Var(np.pi/4)
f4 = 3 * 2 * x4.tan() +  3
print (f4.val, f4.der)


# Expect value of 64.0, derivative of 48.0
x5= Var(4.0)
f5 = x5.pow(3)
print (f5.val, f5.der)

x5= Var(4.0)
f5 = x5.pow(1/2)
print (f5.val, f5.der) # sqrt 2.0 0.25

x5= Var(-2)
f5 = x5.pow(1/2)
print (f5.val, f5.der) # nan nan


# Expect value of 1.0, derivative of 0.23025850929940458
x6 = Var(10)
f6 = x6.log(10)
print (f6.val, f6.der)

x6 = Var(0)
f6 = x6.log(2)
print (f6) # inf


# Expect value of 2.718281828459045, derivative of 2.718281828459045
x7 = Var(1)
f7 = x7.exp()
print (f7.val, f7.der)



# TODO: handle case where there are multiple inputs
# (each Var should have self.grads = {} instead of 1 self.der)
# x4 = Var(5.0)
# f4 = x3 / x4
# print (f4.val, f4.der)



