import sympy as sp

a = 0.7
b = 0.8

vr, wr = sp.symbols('vr wr') # voltage, relaxation for DRI
vs, ws = sp.symbols('vs ws') # voltage, relaxation for DSI
vv, wv = sp.symbols('vv wv') # voltage, relaxation for VSI
vc, wc = sp.symbols('vc wc') # voltage, relaxation for C2

# Couplings format: from_to
vv_vr, vc_vr = sp.symbols('vv_vr vc_vr')
vv_vs, vc_vs, vr_vs = sp.symbols('vv_vs vc_vs vr_vs')
vc_vv, vs_vv = sp.symbols('vc_vv vs_vv')
vs_vc, vv_vc = sp.symbols('vs_vc vv_vc')

# System of equations for voltage
vr_p = vr - vr**3/3 - wr + vv*vv_vr + vc*vc_vr
vs_p = vs - vs**3/3 - ws + vv*vv_vs + vc*vc_vs + vr*vr_vs
vv_p = vv - vv**3/3 - wv + vc*vc_vv + vs*vs_vv
vc_p = vc - vc**3/3 - wc + vs*vs_vc + vv*vv_vc

# System of equations for relaxation
wr_p = vr + a - b*wr
ws_p = vs + a - b*ws
wv_p = vv + a - b*wv
wc_p = vc + a - b*wc
