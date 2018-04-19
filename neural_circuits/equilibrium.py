import sympy as sp
from equations import *

eqs = sp.nonlinsolve(
    [vr_p, vs_p, vv_p, vc_p, wr_p, ws_p, wv_p, wc_p],
    [vr, wr, vs, ws, vv, wv, vc, wc])

print(eqs)
