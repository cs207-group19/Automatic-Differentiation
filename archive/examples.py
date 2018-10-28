'''
# Expect value of 18.42, derivative of 6.0
x1 = Var(np.pi / 2)
f1 = 3 * 2 * x1.sin() + 2 * x1 + 4 * x1 + 3
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
f4 = 3 * 2 * x4.tan() + 3
print (f4.val, f4.der)

# Expect value of 64.0, derivative of 48.0
x5 = Var(4.0)
f5 = x5.pow(3)
print (f5.val, f5.der)

x6 = Var(4.0)
f6 = x6.pow(1/2)
print (f6.val, f6.der) # sqrt 2.0 0.25

x7 = Var(-2)
f7 = x7.pow(1/2)
print (f7.val, f7.der) # nan nan

# Expect value of 1.0, derivative of 0.23025850929940458
x8 = Var(10)
f8 = x8.log(10)
print (f8.val, f8.der)

with np.testing.assert_raises(ValueError):
	x9 = Var(0)
	f9 = x9.log(2)
	print (f9.val, f9.der)

# Expect value of 2.718281828459045, derivative of 2.718281828459045
x10 = Var(1)
f10 = x10.exp()
print (f10.val, f10.der)

# TODO: handle vector to scalar case
# (each Var should have self.grads = {} instead of 1 self.der)
x11a = Var(2.0)
x11b = Var(3.0)
f11 = x11a * x11b
print (f11.val, f11.der)
'''