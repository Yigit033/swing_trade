import numpy as np
import matplotlib.pyplot as plt

g = 9.81 #m/s^2
rho = 1025 #kg/m^3
visc = 0.001002 #Pa.s
visc_kin = 0.000001188

L_bp = 48.0 #m
L_wl = 49.0 #m
L_fore = 0 #m
L_aft = 0 #m
B = 9.5 #m, Beam
T = 3.632 #m, Draft
T_f = 2.981 #m Forward Draft
T_a = 2.986 #m Aft Draft
D = 6.0 # m Depth
Disp = 610.0 #m^3, Displacement

C_b = 0.489 # Block Coefficient
C_p = 0.669
C_wp = 0.831
C_m = 0.733 # Midship Coefficient

A_bt = 0 #m^2
h_b = 0 #m

S_app = [0.00001]
k2i = [1]

A_t = 6.213 #m^2

c_stern = 10

c23 = (0.453 + 0.4425*C_b - 0.2862*C_m - 0.003467*(B/T) + 0.3696*C_wp)
S = 475.633 #m^2

A_v = S*0.5
C_da = 0.8

LCB = 21.751

v = 18 #Knot

Fr_d = (v*0.5144)/(g*L_wl)**0.5
LCB = -(0.44*Fr_d - 0.094)

ship_parameters = [L_bp, L_wl, L_fore, L_aft, B, T, T_f, T_a, D, Disp,
                   C_b, C_p, C_m, S, A_bt, h_b, S_app, k2i, A_v, C_da, A_t, LCB, c_stern, v]

def holtrop_mennen(*ship_parameters):

    L_r = L_wl * (1-C_p + 0.06*C_p*LCB)/(4*C_p-1)
    a = -((L_wl/B)**0.80856) * (1-C_wp)**0.30484 * (1 - C_p - 0.0225*LCB)**0.6367 * (L_r/B)**0.34574 * ((100*Disp/L_wl**3)**0.16302)
    i_e = 1 + 89*np.exp(a)
    Fr = (v*0.5144)/(g*L_wl)**0.5
    Re = (v*0.5144) * L_wl/visc_kin

    c14 = 1 + 0.011*c_stern
    k = -0.07 + (0.487118*c14 * ((B/L_wl)**1.06806) * ((T/L_wl)**0.46106) * ((L_wl/L_r)**0.121563) * ((L_wl**3/Disp)**0.36486) * ((1-C_p)**-0.604247))

    C_f = 0.075/(np.log10(Re)-2)**2
    R_f = 0.5 * rho * (v*0.5144)**2 * S * C_f

    count = 0
    sum_k = 0

    for i in k2i:
        sum_k = sum_k + ((1+i) * S_app[count])
        count = count + 1

    k2 = sum_k/np.sum(S_app) - 1

    d_th = 0.5
    cd_th = 0.01
    R_th = rho*(v*0.5144) * np.pi * d_th**2 * cd_th

    R_app = 0.5 * rho * (v*0.5144) * (k2+1) * C_f * np.sum(S_app) + R_th

    if B/L_wl <= 0.11:
        c7 = 0.229577 * (B/L_wl)**(1/3)
    elif B/L_wl > 0.11 and B/L_wl <= 0.25:
        c7 = B/L_wl
    elif B/L_wl > 0.25:
        c7 = 0.5 - 0.0625 * (L_wl/B)

    c1 = 2223105 * c7**3.78613 * (T/B)**1.07961 * (90 - i_e)**-1.37565
    c3 = 0.56 * (A_bt**1.5) / (B*T * (0.31*A_bt**0.5 + T_f - h_b))
    c2 = np.exp(-1.89 * c3**0.5)

    c5 = 1 - 0.8 * (A_t/(B*T*C_m))

    d = -0.9

    if L_wl/B <= 12:
        lamb = 1.446*C_p - 0.03 * (L_wl/B)
    elif L_wl/B >12:
        lamb = 1.446*C_p - 0.36

    if C_p <= 0.8:
        c16 = 8.07981*C_p - 13.8673*C_p**2 + 6.984388*C_p**3
    elif C_p > 0.8:
        c16 = 1.73014 - 0.7067*C_p

    m1 = 0.0140407 * (L_wl/T) - 1.75254 * (Disp**(1/3) / L_wl) - 4.79323*(B/L_wl) - c16

    if L_wl**3 / Disp <= 512:
        c15 = -1.69385
    elif L_wl**3 / Disp > 512 and L_wl**3 / Disp <=1726.91:
        c15 = -1.69385 + (((L_wl / Disp**(1/3)) - 8)/2.36)
    elif L_wl**3 / Disp > 1726.91:
        c15 = 0

    c17 = 6919.3*C_m**-1.3346 * (Disp/L_wl**3)**2.00977 * (L_wl/B - 2)**1.40692
    m3 = -7.2035 * ((B/L_wl)**0.326869) * (T/B)**0.605375

    def R_w_a(Fn):
        m4_a = 0.4*c15*np.exp(-0.034 * Fn**-3.29)
        return c1*c2*c5*rho*g*Disp*np.exp(m1*Fn**d + m4_a * np.cos(lamb*Fn**-2))

    def R_w_b(Fn):
        m4_b = 0.4*c15 * np.exp(0.034 * Fn**-3.29)
        return c17*c2*c5*rho*g*Disp*np.exp(m3*Fn**d + m4_b * np.cos(lamb*Fn**-2))

    if Fr <= 0.4:
        R_w = R_w_a(Fr)
    elif Fr > 0.55:
        R_w = R_w_b(Fr)
    else:
        R_w = R_w_a(0.4) + ((20*Fr - 8)/3)*(R_w_b(0.55) - R_w_a(0.4))

    if A_bt > 0:
        h_f = C_p * C_m * (B*T/L_wl) * (136 - 316.3*Fr) * Fr**3
        h_w = i_e * (v*0.5144)**2 / (400*g)

        Fr_i = (v*0.5144) / (g * (T_f - h_b - (0.25 * A_bt**0.5) + h_f + h_w))**0.5

        P_b = 0.56 * (A_bt**0.5) / (T_f - 1.5*h_b + h_f)
        R_b = 0.11 * rho * g * A_bt**(3/2) * (Fr_i**3 / (1+Fr_i**2)) * np.exp(-3*(P_b**-2))
    else:
        R_b = 0

    if A_t > 0:
        Fr_t = (v*0.5144) / ((2*g*A_t) / (B + B*C_wp))**0.5
    else:
        Fr_t = 0

    if Fr_t < 5:
        c6 = 0.2*(1-0.2*Fr_t)
    elif Fr_t >= 5:
        c6 = 0

    R_tr = 0.5 * rho * (v*0.5144)**2 * A_t * c6

    if T_f / L_wl <= 0.04:
        c4 = T_f / L_wl
    elif T_f / L_wl > 0.04:
        c4 = 0.04

    C_a = 0.00546 * (L_wl + 100)**-0.16 - 0.002 + 0.003 * (L_wl/7.5)**0.5 * C_b**4 * c2 * (0.04-c4)

    ks = 150

    if ks <= 150:
        delta_C_a = 0
    elif ks > 150:
        delta_C_a = (0.105 * ((ks*1e-6)**(1/3) - 0.005579))/(L_wl**(1/3))

    R_a = 0.5 * rho * (v*0.5144)**2 * (C_a + delta_C_a) * (S + np.sum(S_app))

    R_aa = 0.5 * 1.225 * (v*0.5144)**2 * C_da * A_v

    R_t = R_f*(1+k) + R_app + R_a + R_w + R_b + R_tr + R_aa
    P_t = R_t*(v*0.5144)

    print("Total Resistance:", R_t*10**-3, "kN")
    print("Towing Power:", P_t*10**-3, "kW")

    return {"R_t": R_t}

resistance_results = holtrop_mennen(*ship_parameters)