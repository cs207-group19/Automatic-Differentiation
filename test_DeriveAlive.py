import pytest
import DeriveAlive as ad
import numpy as np
import math

def test_DeriveAive_results():
    x1 = ad.Var(np.pi)
    f1 = 3 * x1.sin() +  3
    assert np.round(f1.val,2) == 3.0
    assert f1.der == -3.0

    x4 = ad.Var(np.pi/4)
    f4 = 3 * 2 * x4.tan() +  3
    assert f4.val == 9.0
    assert np.round(f4.der,2) == 12.0

    x5= ad.Var(4.0)
    f5 = x5.pow(3)
    assert f5.val == 64.0
    assert f5.der == 48.0

def test_DeriveAlive_ZeroDivisionError():
    with pytest.raises(ZeroDivisionError):
        x6 = ad.Var(0)
        f6 = x6.pow(1/2)
        assert f6.val == float('nan')
        assert f6.der == float('nan')
