import numpy as np

#def calc_lab_temp(tp, g, depth):
#    lab_temp = tp + g * abs(depth)*1e3
#    return lab_temp

def calc_lab_temp(tp, g, depth, topo):
    lab_temp = tp + g * abs(depth-topo)*1e3
    return lab_temp

def calc_q_zero(k, tp, kappa, age):
    q_zero = (k * tp)/np.sqrt(np.pi * kappa * age)
    return q_zero

def calc_s(depth, topo, kappa, v, dip, b):
    s = 1. + (b * np.sqrt(((abs(depth-topo)*1.e3)
                          * v * abs(np.sin(dip)))/kappa))
    return s

def calc_slab_lab_int_sigma(sli_depth, sli_topo, sli_temp, sli_s, sli_k,
                              sli_q_zero, v):
    sli_sigma = ((sli_temp * sli_s * sli_k)
                 / (v * abs(sli_depth-sli_topo)*1.e3)) - (sli_q_zero/v)
    return sli_sigma

#### modificado 13-Sept para calcular slab_sigma como un descencso lineal desde el valor en sli hasta la fosa. Igual que en linea 71-72 de thermal.py
def calc_slab_sigma(depth, topo, sli_depth, sli_topo, sli_sigma):
    #mu = sli_sigma / (1. - np.exp(d))
    # print(abs(depth-topo)*1.e3 * d/ abs())
    #slab_sigma = mu * (1. - np.exp(abs(depth-topo)*1.e3 * d
    #                               / (abs(sli_depth-sli_topo)*1.e3)))

    slab_sigma = sli_sigma * ((abs(depth-topo)*1.e3)
                                       / (abs(sli_depth-sli_topo)*1.e3))
    return slab_sigma

def calc_slab_temp(depth, topo, q_zero, slab_sigma, v, k, s):
    slab_temp = ((q_zero + slab_sigma * v) * abs(depth-topo)*1.e3
                 / (k * s))
    return slab_temp

def calc_slab_temp_from_sli_temp(sli_temp, sli_depth, sli_topo, depth, topo, age, k, kappa, dip, v, tp, b):
    age = age*1.e6*365.*24.*60.*60.
    v = v/(1.e6*365.*24.*60.*60.)
    sli_s = calc_s(
        sli_depth,
        sli_topo,
        kappa,
        v,
        dip,
        b
    )
    q_zero = calc_q_zero(
        k,
        tp,
        kappa,
        age
    )
    sli_sigma = calc_slab_lab_int_sigma(
        sli_depth,
        sli_topo,
        sli_temp,
        sli_s,
        k,
        q_zero,
        v
    )
    slab_sigma = calc_slab_sigma(
        depth,
        topo,
        sli_depth,
        sli_topo,
        sli_sigma
    )
    slab_s = calc_s(
        depth,
        topo,
        kappa,
        v,
        dip,
        b
    )
    slab_temp = calc_slab_temp(
        depth,
        topo,
        q_zero,
        slab_sigma,
        v,
        k,
        slab_s
    )
    return slab_temp

def calc_H0_M1(k, Z0, Zi, Zm, Zb, Tb):
    Zi = -(Zi-Z0)*1.e3
    Zm = -(Zm-Z0)*1.e3
    Zb = -(Zb-Z0)*1.e3
    def Ho(shf):
        shf = shf/1.e3
        return (shf*Zb - Tb*k)*np.exp(Zm/Zi)/(Zi*(Zb*np.exp(Zm/Zi) - Zb + Zm - Zi*np.exp(Zm/Zi) + Zi))
    return Ho

def calc_k_M1(Ho, Z0, Zi, Zm, Zb, Tb):
    Zi = -(Zi-Z0)*1.e3
    Zm = -(Zm-Z0)*1.e3
    Zb = -(Zb-Z0)*1.e3
    def k(shf):
        shf = shf/1.e3
        return (Ho*Zi*(Zb - Zm - Zi) + (-Ho*Zb*Zi + Ho*Zi**2 + shf*Zb)*np.exp(Zm/Zi))*np.exp(-Zm/Zi)/Tb
    return k

def calc_Tp_M1(k, Ho, Z0, Zi, Zm, Zb, G):
    Zi = -(Zi-Z0)*1.e3
    Zm = -(Zm-Z0)*1.e3
    Zb = -(Zb-Z0)*1.e3
    def Tp(shf):
        shf = shf/1.e3
        Tb = (-Ho*Zb*Zi/k + Ho*Zb*Zi*np.exp(-Zm/Zi)/k - Ho*Zm*Zi*np.exp(-Zm/Zi)/k + Ho*Zi**2/k - Ho*Zi**2*np.exp(-Zm/Zi)/k + shf*Zb/k)
        return Tb - G * Zb
    return Tp

def calc_Huc_M2(Hlc, k, Z0, Zi, Zm, Zb, Tb):
    Zi = -(Zi-Z0)*1.e3
    Zm = -(Zm-Z0)*1.e3
    Zb = -(Zb-Z0)*1.e3
    def Huc(shf):
        shf = shf/1.e3
        return (2*Hlc*Zb*Zi - 2*Hlc*Zb*Zm - Hlc*Zi**2 + Hlc*Zm**2 + 2*shf*Zb - 2*Tb*k)/(Zi*(2*Zb - Zi))
    return Huc

def calc_Hlc_M2(Huc, k, Z0, Zi, Zm, Zb, Tb):
    Zi = -(Zi-Z0)*1.e3
    Zm = -(Zm-Z0)*1.e3
    Zb = -(Zb-Z0)*1.e3
    def Hlc(shf):
        shf = shf/1.e3
        return (2*Huc*Zb*Zi - Huc*Zi**2 - 2*shf*Zb + 2*Tb*k)/(2*Zb*Zi - 2*Zb*Zm - Zi**2 + Zm**2)
    return Hlc

def calc_k_M2(Huc, Hlc, Z0, Zi, Zm, Zb, Tb):
    Zi = -(Zi-Z0)*1.e3
    Zm = -(Zm-Z0)*1.e3
    Zb = -(Zb-Z0)*1.e3
    def k(shf):
        shf = shf/1.e3
        return (Hlc*Zb*Zi - Hlc*Zb*Zm - Hlc*Zi**2/2 + Hlc*Zm**2/2 - Huc*Zb*Zi + Huc*Zi**2/2 + shf*Zb)/Tb
    return k

def calc_Tp_M2(k, Huc, Hlc, Z0, Zi, Zm, Zb, G):
    Zi = -(Zi-Z0)*1.e3
    Zm = -(Zm-Z0)*1.e3
    Zb = -(Zb-Z0)*1.e3
    def Tp(shf):
        shf = shf/1.e3
        Tb = (Hlc*Zb*Zi/k - Hlc*Zb*Zm/k - Hlc*Zi**2/(2*k) + Hlc*Zm**2/(2*k) - Huc*Zb*Zi/k + Huc*Zi**2/(2*k) + shf*Zb/k)
        return Tb - G * Zb
    return Tp

def calc_Huc_M3(Hlc, k_uc, k_lcm, Z0, Zi, Zm, Zb, Tb):
    Zi = -(Zi-Z0)*1.e3
    Zm = -(Zm-Z0)*1.e3
    Zb = -(Zb-Z0)*1.e3
    def Huc(shf):
        shf = shf/1.e3
        return (2*Hlc*Zb*Zi*k_uc - 2*Hlc*Zb*Zm*k_uc - Hlc*Zi**2*k_uc + Hlc*Zm**2*k_uc + 2*shf*Zb*k_uc + 2*shf*Zi*k_lcm - 2*shf*Zi*k_uc - 2*Tb*k_lcm*k_uc)/(Zi*(2*Zb*k_uc + Zi*k_lcm - 2*Zi*k_uc))
    return Huc

def calc_Hlc_M3(Huc, k_uc, k_lcm, Z0, Zi, Zm, Zb, Tb):
    Zi = -(Zi-Z0)*1.e3
    Zm = -(Zm-Z0)*1.e3
    Zb = -(Zb-Z0)*1.e3
    def Hlc(shf):
        shf = shf/1.e3
        return (2*Huc*Zb*Zi*k_uc + Huc*Zi**2*k_lcm - 2*Huc*Zi**2*k_uc - 2*shf*Zb*k_uc - 2*shf*Zi*k_lcm + 2*shf*Zi*k_uc + 2*Tb*k_lcm*k_uc)/(k_uc*(2*Zb*Zi - 2*Zb*Zm - Zi**2 + Zm**2))
    return Hlc

def calc_kuc_M3(Huc, Hlc, k_lcm, Z0, Zi, Zm, Zb, Tb):
    Zi = -(Zi-Z0)*1.e3
    Zm = -(Zm-Z0)*1.e3
    Zb = -(Zb-Z0)*1.e3
    def kuc(shf):
        shf = shf/1.e3
        return Zi*k_lcm*(-Huc*Zi + 2*shf)/(-2*Hlc*Zb*Zi + 2*Hlc*Zb*Zm + Hlc*Zi**2 - Hlc*Zm**2 + 2*Huc*Zb*Zi - 2*Huc*Zi**2 - 2*shf*Zb + 2*shf*Zi + 2*Tb*k_lcm)
    return kuc

def calc_klcm_M3(Huc, Hlc, k_uc, Z0, Zi, Zm, Zb, Tb):
    Zi = -(Zi-Z0)*1.e3
    Zm = -(Zm-Z0)*1.e3
    Zb = -(Zb-Z0)*1.e3
    def klcm(shf):
        shf = shf/1.e3
        return k_uc*(2*Hlc*Zb*Zi - 2*Hlc*Zb*Zm - Hlc*Zi**2 + Hlc*Zm**2 - 2*Huc*Zb*Zi + 2*Huc*Zi**2 + 2*shf*Zb - 2*shf*Zi)/(Huc*Zi**2 - 2*shf*Zi + 2*Tb*k_uc)
    return klcm

def calc_Tp_M3(k_uc, k_lcm, Huc, Hlc, Z0, Zi, Zm, Zb, G):
    Zi = -(Zi-Z0)*1.e3
    Zm = -(Zm-Z0)*1.e3
    Zb = -(Zb-Z0)*1.e3
    def Tp(shf):
        shf = shf/1.e3
        Tb = (Hlc*Zb*Zi/k_lcm - Hlc*Zb*Zm/k_lcm - Hlc*Zi**2/(2*k_lcm) + Hlc*Zm**2/(2*k_lcm) - Huc*Zb*Zi/k_lcm - Huc*Zi**2/(2*k_uc) + Huc*Zi**2/k_lcm + shf*Zb/k_lcm + shf*Zi/k_uc - shf*Zi/k_lcm)
        return Tb - G * Zb
    return Tp
