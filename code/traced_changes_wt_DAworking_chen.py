###### 1.NEW FEATURE ADDED 
### Added __repr__, now can do
v1 = da.Var([1,2,3]) 
#Var(array([1, 2, 3]), array([1, 1, 1]))

v1.sin()
#Var(array([0.84147098, 0.90929743, 0.14112001]), array([ 0.54030231, -0.41614684, -0.9899925 ]))



###### 2 Test 
v2 = da.Var([0.1, 100, 50, 0.9])
#Var(array([  0.1, 100. ,  50. ,   0.9]), array([1., 1., 1., 1.]))

v3 = da.Var(0.999)
v = (v3.sin()*v1.cos()).tan()
#Var(array([ 0.48843956, -0.36497236, -1.0989693 ]), array([-0.56399243, -0.55760522, -0.16859617]))


### 2.1 Test each funcation:
# v = 3+ v, v = v+ 4, v = v* 4, v = 4 * v, v/4, 4/v (0/4, 4/0), sin, cos, tan,sinh(), cosh() all work :) 

# !!!!!!!! Problems for some functions !!!!!!!!

### 2.1.1 Pow 
###### 1 
>>> v.pow(0)
#Var(array([1., 1., 1.]), array([-0.,  0.,  0.]))

FIXED
# change the first der '-0' to 0, Now
#Var(array([1., 1., 1.]), array([0., 0., 0.]))

###### 2
# v = Var(array([ 0.48843956, -0.36497236, -1.0989693 ]), array([-0.56399243, -0.55760522, -0.16859617]))
# we shouldn't get nan when situation slike power = 1/3 (should get nan when situations like power = 1/2)
>>> v.pow(1/3)
#Var(array([nan]), array([nan]))
>>> v.pow(1/2)
#Var(array([nan]), array([nan]))

#FIXED (power = 1/3)
 v4 = da.Var([1,-1,4,-4])
 v4.pow(1/3) 
 # Var(array([ 1.        , -1.        ,  1.58740105, -1.58740105]), array([ 0.33333333, -0.33333333,  0.13228342, -0.13228342]))
 
 ## !!!! NOTE !!!! !!!! NOTE !!!! !!!! NOTE !!!! !!!! NOTE !!!! 
 v4.pow(1/2) # give us the output ( which can be useful if we can add j ?????)
 #Var(array([ 1., -1.,  2., -2.]), array([ 0.5 , -0.5 ,  0.25, -0.25]))


### 2.1.2 v4.arcsin(), v4.arccos(), v4.arctan()(FIXED for arctan()) behave weirdly !
v4.arcsin()
File "<stdin>", line 1, in <module>
  File "/Users/chenshi/Dropbox/Harvard/courses/2018Fall/CS207/cs207-FinalProject/code/DeriveAlive_working_chen.py", line 87, in arcsin
    val = np.arcsin(self.val)
FloatingPointError: invalid value encountered in arcsin
# since the input value have to be within range (-2*pi , 2pi), I try to catch the FloatingPointError, then we wont have trouble when testing
# HOWEVER, for e.g., our array v4 =da.Var([1,-1,4,-4]), so for 1 and -1 we should be able to get output, since they are within the range
# do we want to return the arcsin() for the values that have arcsin()  and Nan for those that dont have arcsin() 
# or do we just wanna return nan as long as one value in the array doesn't have arcsin() ?

v4.arccos()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/Users/chenshi/Dropbox/Harvard/courses/2018Fall/CS207/cs207-FinalProject/code/DeriveAlive_working_chen.py", line 92, in arccos
    val = np.arccos(self.val)
FloatingPointError: invalid value encountered in arccos
# same thing for arccos(), 
# and also same issue for POW when power is like 1/2, when we have v4 =da.Var([1,-1,4,-4]
# since 1 and 4 will have 1/2 power, while the other two don't

v4.arctan()
# FIXED 
# Var(array([ 0.78539816, -0.78539816,  1.32581766, -1.32581766]), array([0.02857143]))
# I changed the line #der = 1/(1+np.linalg.norm(self.val)**2) to  der = 1/(1+(self.val)**2)
# Now the output
# Var(array([ 0.78539816, -0.78539816,  1.32581766, -1.32581766]), array([0.5       , 0.5       , 0.05882353, 0.05882353]))


### 2.2 
 v = (v3.sin()*v1.cos()).tan()+ v2
 Error : 
 File "<stdin>", line 1, in <module>
  File "/Users/chenshi/Dropbox/Harvard/courses/2018Fall/CS207/cs207-FinalProject/code/DeriveAlive_working_chen.py", line 28, in __add__
    val += other.val
ValueError: operands could not be broadcast together with shapes (3,) (4,) (3,) 
# Because v and v2 differ in length (Maybe Should consider this situation????)
# Do we need to fix ??????!!!!!!!! since if they are the same length, it works gracefully :) 
v = (v3.sin()*v1.cos()).tan()+ v3
#Var(array([ 1.48743956,  0.63402764, -0.0999693 ]), array([0.43600757, 0.44239478, 0.83140383]))








