import sys
sys.path.append('../DeriveAlive/')

import spline as sp
import DeriveAlive as da
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

logo = []
original = []

'''
first letter
'''

def f1(var):
	return (-1/0.5**2) * var**2 + 1

xMin1 = -0.5
xMax1 = 0
nIntervals = 1
nSplinePoints = 5

y1, A1, coeffs1, ks1 = sp.quad_spline_coeff(f1, xMin1, xMax1, nIntervals)
spline_points1 = sp.spline_points(f1, coeffs1, ks1, nSplinePoints)
logo.append(spline_points1)
original.append(ks1)

def f2(var):
	return -1 + (1/0.5**2) * var**2

xMin2 = -0.5
xMax2 = 0
nIntervals = 1
nSplinePoints = 5

y2, A2, coeffs2, ks2 = sp.quad_spline_coeff(f2, xMin2, xMax2, nIntervals)
spline_points2 = sp.spline_points(f2, coeffs2, ks2, nSplinePoints)
logo.append(spline_points2)
original.append(ks2)

def f3(var):
	return (-0.5/0.5**2) * var**2 + 1

xMin3 = 0
xMax3 = 0.5
nIntervals = 1
nSplinePoints = 5

y3, A3, coeffs3, ks3 = sp.quad_spline_coeff(f3, xMin3, xMax3, nIntervals)
ks3New = deepcopy(ks3)
ks3New[0].val = ks3New[0].val + 0.5
ks3New[1].val = ks3New[1].val + 0.5
spline_points3 = sp.spline_points(f3, coeffs3, ks3New, nSplinePoints)
spline_points3[0][0] = spline_points3[0][0] - 0.5
spline_points3[0][1] = -1*spline_points3[0][1] + 1.5
logo.append(spline_points3)
original.append(ks3)

def f4(var):
	return -1 + (0.5/0.5**2) * var**2

xMin4 = 0
xMax4 = 0.5
nIntervals = 1
nSplinePoints = 5

y4, A4, coeffs4, ks4 = sp.quad_spline_coeff(f4, xMin4, xMax4, nIntervals)
ks4New = deepcopy(ks4)
ks4New[0].val = ks4New[0].val - 0.5
ks4New[1].val = ks4New[1].val - 0.5
spline_points4 = sp.spline_points(f4, coeffs4, ks4, nSplinePoints)
spline_points4[0][0] = -1*spline_points4[0][0] + 0.5
spline_points4[0][1] = -1*spline_points4[0][1] - 1.5
logo.append(spline_points4)
original.append(ks4)

'''
second letter
'''

def f5(var):
	return 1 + (-0.5/0.5**2) * (-1.5+var)**2

xMin5 = 1.5
xMax5 = 2
nIntervals = 1
nSplinePoints = 5

y5, A5, coeffs5, ks5 = sp.quad_spline_coeff(f5, xMin5, xMax5, nIntervals)
ks5New = deepcopy(ks5)
ks5New[0].val = ks5New[0].val + 0.5
ks5New[1].val = ks5New[1].val + 0.5
spline_points5 = sp.spline_points(f5, coeffs5, ks5New, nSplinePoints)
spline_points5[0][0] = spline_points5[0][0] - 0.5
spline_points5[0][1] = -1*spline_points5[0][1] + 1.5
logo.append(spline_points5)
original.append(ks5)

def f6(var):
	return -1 + (0.5/0.5**2) * (-1.5+var)**2

xMin6 = 1.5
xMax6 = 2
nIntervals = 1
nSplinePoints = 5

y6, A6, coeffs6, ks6 = sp.quad_spline_coeff(f6, xMin6, xMax6, nIntervals)
ks6New = deepcopy(ks6)
ks6New[0].val = ks6New[0].val - 0.5
ks6New[1].val = ks6New[1].val - 0.5
spline_points6 = sp.spline_points(f6, coeffs6, ks6, nSplinePoints)
spline_points6[0][0] = -1*spline_points6[0][0] + 3.5
spline_points6[0][1] = -1*spline_points6[0][1] - 1.5
logo.append(spline_points6)
original.append(ks6)

def f7(var):
	return (-0.5/0.5**2) * (-1.5+var)**2 + 1

xMin7 = 1
xMax7 = 1.5
nIntervals = 1
nSplinePoints = 5

y7, A7, coeffs7, ks7 = sp.quad_spline_coeff(f7, xMin7, xMax7, nIntervals)
spline_points7 = sp.spline_points(f7, coeffs7, ks7, nSplinePoints)
logo.append(spline_points7)
original.append(ks7)

def f8(var):
	return -1 + (0.5/0.5**2) * (-1.5+var)**2

xMin8 = 1
xMax8 = 1.5
nIntervals = 1
nSplinePoints = 5

y8, A8, coeffs8, ks8 = sp.quad_spline_coeff(f8, xMin8, xMax8, nIntervals)
spline_points8 = sp.spline_points(f8, coeffs8, ks8, nSplinePoints)
logo.append(spline_points8)
original.append(ks8)

def f9(var):
	return (0.5/0.5**2) * (-1.5+var)**2

xMin9 = 1
xMax9 = 1.5
nIntervals = 1
nSplinePoints = 5

y9, A9, coeffs9, ks9 = sp.quad_spline_coeff(f9, xMin9, xMax9, nIntervals)
spline_points9 = sp.spline_points(f9, coeffs9, ks9, nSplinePoints)
logo.append(spline_points9)
original.append(ks9)

def f10(var):
	return (-0.5/0.5**2) * (-1.5+var)**2

xMin10 = 1.5
xMax10 = 2
nIntervals = 1
nSplinePoints = 5

y10, A10, coeffs10, ks10 = sp.quad_spline_coeff(f10, xMin10, xMax10, nIntervals)
ks10New = deepcopy(ks10)
ks10New[0].val = ks10New[0].val + 0.5
ks10New[1].val = ks10New[1].val + 0.5
spline_points10 = sp.spline_points(f10, coeffs10, ks10New, nSplinePoints)
spline_points10[0][0] = spline_points10[0][0] - 0.5
spline_points10[0][1] = -1*spline_points10[0][1] - 0.5
logo.append(spline_points10)
original.append(ks10)

'''
third letter
'''

def f11(var):
	return (-0.5/0.5**2) * (-3+var)**2 + 1

xMin11 = 2.5
xMax11 = 3
nIntervals = 1
nSplinePoints = 5

y11, A11, coeffs11, ks11 = sp.quad_spline_coeff(f11, xMin11, xMax11, nIntervals)
spline_points11 = sp.spline_points(f11, coeffs11, ks11, nSplinePoints)
logo.append(spline_points11)
original.append(ks11)

def f12(var):
	return (-0.5/0.5**2) * (-3+var)**2 + 1

xMin12 = 3
xMax12 = 3.5
nIntervals = 1
nSplinePoints = 5

y12, A12, coeffs12, ks12 = sp.quad_spline_coeff(f12, xMin12, xMax12, nIntervals)
ks12New = deepcopy(ks12)
ks12New[0].val = ks12New[0].val + 0.5
ks12New[1].val = ks12New[1].val + 0.5
spline_points12 = sp.spline_points(f12, coeffs12, ks12New, nSplinePoints)
spline_points12[0][0] = spline_points12[0][0] - 0.5
spline_points12[0][1] = -1*spline_points12[0][1] + 1.5
logo.append(spline_points12)
original.append(ks12)

def f13(var):
    return -4.75 + 1.5*var

xMin13 = 2.5333
xMax13 = 3.5
nIntervals = 2
nSplinePoints = 5

y13, A13, coeffs13, ks13 = sp.quad_spline_coeff(f13, xMin13, xMax13, nIntervals)
spline_points13 = sp.spline_points(f13, coeffs13, ks13, nSplinePoints)
logo.append(spline_points13)
original.append(ks13)

def f14(var):
    return da.Var([-0.95], None)

xMin14 = 2.5
xMax14 = 3.5
nIntervals = 2
nSplinePoints = 5

y14, A14, coeffs14, ks14 = sp.quad_spline_coeff(f14, xMin14, xMax14, nIntervals)
spline_points14 = sp.spline_points(f14, coeffs14, ks14, nSplinePoints)
logo.append(spline_points14)
original.append(ks14)

'''
fourth letter
'''

def f15(var):
    return (-1/0.5**2) * (-4.5+var)**2 + 1

xMin15 = 4
xMax15 = 4.5
nIntervals = 1
nSplinePoints = 5

y15, A15, coeffs15, ks15 = sp.quad_spline_coeff(f15, xMin15, xMax15, nIntervals)
spline_points15 = sp.spline_points(f15, coeffs15, ks15, nSplinePoints)
logo.append(spline_points15)
original.append(ks15)

def f16(var):
    return -1 + (1/0.5**2) * (-4.5+var)**2

xMin16 = 4
xMax16 = 4.5
nIntervals = 1
nSplinePoints = 5

y16, A16, coeffs16, ks16 = sp.quad_spline_coeff(f16, xMin16, xMax16, nIntervals)
spline_points16 = sp.spline_points(f16, coeffs16, ks16, nSplinePoints)
logo.append(spline_points16)
original.append(ks16)

def f17(var):
    return (-1/0.5**2) * (-4.5+var)**2 + 1

xMin17 = 4.5
xMax17 = 5
nIntervals = 1
nSplinePoints = 5

y17, A17, coeffs17, ks17 = sp.quad_spline_coeff(f17, xMin17, xMax17, nIntervals)
ks17New = deepcopy(ks17)
ks17New[0].val = ks17New[0].val + 0.5
ks17New[1].val = ks17New[1].val + 0.5
spline_points17 = sp.spline_points(f17, coeffs17, ks17New, nSplinePoints)
spline_points17[0][0] = spline_points17[0][0] - 0.5
spline_points17[0][1] = -1*spline_points17[0][1] + 1
logo.append(spline_points17)
original.append(ks17)

def f18(var):
    return -1 + (1/0.5**2) * (-4.5+var)**2

xMin18 = 4.5
xMax18 = 5
nIntervals = 1
nSplinePoints = 5

y18, A18, coeffs18, ks18 = sp.quad_spline_coeff(f18, xMin18, xMax18, nIntervals)
ks18New = deepcopy(ks18)
ks18New[0].val = ks18New[0].val + 0.5
ks18New[1].val = ks18New[1].val + 0.5
spline_points18 = sp.spline_points(f18, coeffs18, ks18New, nSplinePoints)
spline_points18[0][0] = spline_points18[0][0] - 0.5
spline_points18[0][1] = -1*spline_points18[0][1] - 1
logo.append(spline_points18)
original.append(ks18)

'''
fifth letter
'''

def f19(var):
    return da.Var([1], None)

xMin19 = 5.5
xMax19 = 6.5
nIntervals = 2
nSplinePoints = 5

y19, A19, coeffs19, ks19 = sp.quad_spline_coeff(f19, xMin19, xMax19, nIntervals)
spline_points19 = sp.spline_points(f19, coeffs19, ks19, nSplinePoints)
logo.append(spline_points19)
original.append(ks19)

def f20(var):
    return (-2/(-0.75)**2) * (-6.5+var)**2 + 1

xMin20 = 5.75
xMax20 = 6.5
nIntervals = 1
nSplinePoints = 5

y20, A20, coeffs20, ks20 = sp.quad_spline_coeff(f20, xMin20, xMax20, nIntervals)
spline_points20 = sp.spline_points(f20, coeffs20, ks20, nSplinePoints)
logo.append(spline_points20)
original.append(ks20)


def drawPoints():
    
    plt.figure()
    
    fx = []
    fy = []
    for k in ks1:
        fx.append(k.val)
        fy.append(f1(k).val)
        plt.plot(fx, fy, 'o-', linewidth=2, color='k')


    fx = []
    fy = []
    for k in ks2:
        fx.append(k.val)
        fy.append(f2(k).val)
        plt.plot(fx, fy, 'o-', linewidth=2, color='k')

    fx = []
    fy = []
    for k in ks3:
        fx.append(k.val)
        fy.append(f3(k).val)
        plt.plot(fx, fy, 'o-', linewidth=2, color='k')    

    fx = []
    fy = []
    for k in ks4:
        fx.append(k.val)
        fy.append(f4(k).val)
        plt.plot(fx, fy, 'o-', linewidth=2, color='k')  

    fx = []
    fy = []
    for k in ks5:
        fx.append(k.val)
        fy.append(f5(k).val)
        plt.plot(fx, fy, 'o-', linewidth=2, color='k')    

    fx = []
    fy = []
    for k in ks6:
        fx.append(k.val)
        fy.append(f6(k).val)
        plt.plot(fx, fy, 'o-', linewidth=2, color='k')  

    fx = []
    fy = []
    for k in ks7:
        fx.append(k.val)
        fy.append(f7(k).val)
        plt.plot(fx, fy, 'o-', linewidth=2, color='k')  

    fx = []
    fy = []
    for k in ks8:
        fx.append(k.val)
        fy.append(f8(k).val)
        plt.plot(fx, fy, 'o-', linewidth=2, color='k')  

    fx = []
    fy = []
    for k in ks9:
        fx.append(k.val)
        fy.append(f9(k).val)
        plt.plot(fx, fy, 'o-', linewidth=2, color='k') 

    fx = []
    fy = []
    for k in ks10:
        fx.append(k.val)
        fy.append(f10(k).val)
        plt.plot(fx, fy, 'o-', linewidth=2, color='k') 

    fx = []
    fy = []
    for k in ks11:
        fx.append(k.val)
        fy.append(f11(k).val)
        plt.plot(fx, fy, 'o-', linewidth=2, color='k') 

    fx = []
    fy = []
    for k in ks12:
        fx.append(k.val)
        fy.append(f12(k).val)
        plt.plot(fx, fy, 'o-', linewidth=2, color='k') 

    fx = []
    fy = []
    for k in ks13:
        fx.append(k.val)
        fy.append(f13(k).val)
        plt.plot(fx, fy, 'o-', linewidth=2, color='k') 

    fx = []
    fy = []
    for k in ks14:
        fx.append(k.val)
        fy.append(f14(k).val)
        plt.plot(fx, fy, 'o-', linewidth=2, color='k') 

    fx = []
    fy = []
    for k in ks15:
        fx.append(k.val)
        fy.append(f15(k).val)
        plt.plot(fx, fy, 'o-', linewidth=2, color='k') 

    fx = []
    fy = []
    for k in ks16:
        fx.append(k.val)
        fy.append(f16(k).val)
        plt.plot(fx, fy, 'o-', linewidth=2, color='k') 

    fx = []
    fy = []
    for k in ks17:
        fx.append(k.val)
        fy.append(f17(k).val)
        plt.plot(fx, fy, 'o-', linewidth=2, color='k') 

    fx = []
    fy = []
    for k in ks18:
        fx.append(k.val)
        fy.append(f18(k).val)
        plt.plot(fx, fy, 'o-', linewidth=2, color='k') 

    fx = []
    fy = []
    for k in ks19:
        fx.append(k.val)
        fy.append(f19(k).val)
        plt.plot(fx, fy, 'o-', linewidth=2, color='k') 

    fx = []
    fy = []
    for k in ks20:
        fx.append(k.val)
        fy.append(f20(k).val)
        plt.plot(fx, fy, 'o-', linewidth=2, color='k') 

    plt.show()

def drawSpline():
    plt.figure()
    for i in logo:
        for spline_point in i:
            plt.plot(spline_point[0], spline_point[1], linewidth=3, color='r')

    plt.show()

def drawTogether():

    plt.figure()

    fx = []
    fy = []
    for k in ks1:
        fx.append(k.val)
        fy.append(f1(k).val)
        plt.plot(fx, fy, 'o-', linewidth=2, color='k')


    fx = []
    fy = []
    for k in ks2:
        fx.append(k.val)
        fy.append(f2(k).val)
        plt.plot(fx, fy, 'o-', linewidth=2, color='k')

    fx = []
    fy = []
    for k in ks3:
        fx.append(k.val)
        fy.append(f3(k).val)
        plt.plot(fx, fy, 'o-', linewidth=2, color='k')    

    fx = []
    fy = []
    for k in ks4:
        fx.append(k.val)
        fy.append(f4(k).val)
        plt.plot(fx, fy, 'o-', linewidth=2, color='k')  

    fx = []
    fy = []
    for k in ks5:
        fx.append(k.val)
        fy.append(f5(k).val)
        plt.plot(fx, fy, 'o-', linewidth=2, color='k')    

    fx = []
    fy = []
    for k in ks6:
        fx.append(k.val)
        fy.append(f6(k).val)
        plt.plot(fx, fy, 'o-', linewidth=2, color='k')  

    fx = []
    fy = []
    for k in ks7:
        fx.append(k.val)
        fy.append(f7(k).val)
        plt.plot(fx, fy, 'o-', linewidth=2, color='k')  

    fx = []
    fy = []
    for k in ks8:
        fx.append(k.val)
        fy.append(f8(k).val)
        plt.plot(fx, fy, 'o-', linewidth=2, color='k')  

    fx = []
    fy = []
    for k in ks9:
        fx.append(k.val)
        fy.append(f9(k).val)
        plt.plot(fx, fy, 'o-', linewidth=2, color='k') 

    fx = []
    fy = []
    for k in ks10:
        fx.append(k.val)
        fy.append(f10(k).val)
        plt.plot(fx, fy, 'o-', linewidth=2, color='k') 

    fx = []
    fy = []
    for k in ks11:
        fx.append(k.val)
        fy.append(f11(k).val)
        plt.plot(fx, fy, 'o-', linewidth=2, color='k') 

    fx = []
    fy = []
    for k in ks12:
        fx.append(k.val)
        fy.append(f12(k).val)
        plt.plot(fx, fy, 'o-', linewidth=2, color='k') 

    fx = []
    fy = []
    for k in ks13:
        fx.append(k.val)
        fy.append(f13(k).val)
        plt.plot(fx, fy, 'o-', linewidth=2, color='k') 

    fx = []
    fy = []
    for k in ks14:
        fx.append(k.val)
        fy.append(f14(k).val)
        plt.plot(fx, fy, 'o-', linewidth=2, color='k') 

    fx = []
    fy = []
    for k in ks15:
        fx.append(k.val)
        fy.append(f15(k).val)
        plt.plot(fx, fy, 'o-', linewidth=2, color='k') 

    fx = []
    fy = []
    for k in ks16:
        fx.append(k.val)
        fy.append(f16(k).val)
        plt.plot(fx, fy, 'o-', linewidth=2, color='k') 

    fx = []
    fy = []
    for k in ks17:
        fx.append(k.val)
        fy.append(f17(k).val)
        plt.plot(fx, fy, 'o-', linewidth=2, color='k') 

    fx = []
    fy = []
    for k in ks18:
        fx.append(k.val)
        fy.append(f18(k).val)
        plt.plot(fx, fy, 'o-', linewidth=2, color='k') 

    fx = []
    fy = []
    for k in ks19:
        fx.append(k.val)
        fy.append(f19(k).val)
        plt.plot(fx, fy, 'o-', linewidth=2, color='k') 

    fx = []
    fy = []
    for k in ks20:
        fx.append(k.val)
        fy.append(f20(k).val)
        plt.plot(fx, fy, 'o-', linewidth=2, color='k') 

    for i in logo:
        for spline_point in i:
            plt.plot(spline_point[0], spline_point[1], linewidth=3, color='r')

    plt.show()








