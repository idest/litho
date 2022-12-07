import numpy as np

#global_thermal_constants = {
#    'Tp': 1375.0,    #[ºC]
#    'G': 0.0004,     #[K/m]
#    'kappa': 1e-06,  #[m2/s]
#    'alpha': 20.0,   #[º]
#    'V': 6.6e4,      #[m/Ma]
#    'b': 1.0,
#    'D': 0.0015,
#    'k': 2.0,        #[W/mK]
#    'kuc': 3.0,
#    'klcm': 1.0,
#    'H0': 3e-06,     #[W/m3]
#    'Huc': 1.65e-06,
#    'Hlc': 4e-07,
#}

class ThermalEquations(object):
    def __init__(self,
        constants=None,
        Tp=1375.0, G=0.0004, kappa=1e-06,
        alpha=20.0, V=6.6e4, b=1.0, D=0.0015, **kwargs
    ):
        # print('...ThermalEquations init')
        if constants is not None:
            self.Tp = constants['Tp']
            self.G = constants['G']
            self.kappa = constants['kappa']
            self.dip = constants['alpha']
            self.v = constants['V']/(1.e6*365.*24.*60.*60.)
            self.b = constants['b']
            #self.d = constants['D']
        else:
            self.Tp = Tp
            self.G = G
            self.kappa = kappa
            self.dip = alpha
            self.v = V/(1.e6*365.*24.*60.*60.)
            self.b = b
            #self.d = D
        self.constants = {
            'Tp': self.Tp,
            'G': self.G,
            'kappa': self.kappa,
            'dip': self.dip,
            'v': self.v,
            'b': self.b
        }

    def calc_lab_temp(self, z, z0):
        Tp, G = self.Tp, self.G
        return Tp + G * abs(z-z0)*1.e3

    def calc_q_zero(self, k, age):
        Tp, kappa = self.Tp, self.kappa
        return (k * Tp)/np.sqrt(np.pi * kappa * age)

    def calc_s(self, z, z0):
        b, v, dip, kappa = self.b, self.v, self.dip, self.kappa
        s = 1. + (b * np.sqrt(((abs(z-z0)*1.e3)
                              * v * abs(np.sin(dip)))/kappa))
        return s

    def calc_slab_lab_int_sigma(self, sli_zb, sli_z0, k, age):
        sli_k = k
        v = self.v
        sli_q_zero = self.calc_q_zero(k, age)
        sli_temp = self.calc_lab_temp(sli_zb, sli_z0)
        sli_s = self.calc_s(sli_zb, sli_z0)
        sli_sigma = ((sli_temp * sli_s * sli_k)
                     / (v * abs(sli_zb-sli_z0)*1.e3)) - (sli_q_zero/v)
        return sli_sigma

    def calc_slab_sigma(self, z, z0, sli_zb, sli_z0, k, age):
        #d = self.d
        sli_sigma = self.calc_slab_lab_int_sigma(sli_zb, sli_z0, k, age)
        #mu = sli_sigma / (1. - np.exp(d))
        #slab_sigma = mu * (1. - np.exp(abs(z-z0)*1.e3 * d
        #                               / (abs(sli_zb-sli_z0)*1.e3)))
        # Sin considerar parametro D
        slab_sigma = sli_sigma * ((abs(z-z0)*1.e3)
                                       / (abs(sli_zb-sli_z0)*1.e3))
        return slab_sigma

    def calc_slab_temp_internal(self, z, z0, sli_z, sli_z0, age, k):
        v = self.v
        q_zero = self.calc_q_zero(k, age)
        slab_sigma = self.calc_slab_sigma(z, z0, sli_z, sli_z0, k, age)
        s = self.calc_s(z, z0)
        slab_temp = ((q_zero + slab_sigma * v) * abs(z-z0)*1.e3
                     / (k * s))
        return slab_temp

class TM0(ThermalEquations):
    def __init__(self, H0=3.e-06, delta=None, k=2.0, **kwargs):
        super().__init__(**kwargs)
        self.name='TM0'
        self.H0 = H0
        if delta is None:
            self.delta = None
        else:
            self.delta = delta*1.e3
        self.k = k
        self.input = {
            'H0': self.H0,
            'delta': self.delta,
            'k': self.k,
            'constants': self.constants
        }

    def calc_slab_temp(self, Zb, Z0, Zb_i, Z0_i, age):
        age = age*1.e6*365.*24.*60.*60.
        slab_temp = self.calc_slab_temp_internal(Zb, Z0, Zb_i, Z0_i, age, self.k)
        return slab_temp

    def calc_geotherm(self, z, Z0, Zi, Zm, Zb, Tb):
        Zi = -(Zi-Z0)*1.e3
        Zm = -(Zm-Z0)*1.e3
        Zb = -(Zb-Z0)*1.e3
        z = -(z-Z0)*1.e3
        if self.delta is None:
            self.delta = Zi
        H0, k, delta = self.H0, self.k, self.delta
        geotherm = (
            ((H0*delta**2)/k)*(1-np.exp(-z/delta))
            +(z/Zb)*(Tb-((H0*delta**2)/k)*(1-np.exp(-Zb/delta)))
        )
        geotherm = geotherm.where((0 <= z) & (z <= Zb))
        #base_temp = temp_sl-((h*delta**2)/k)*(np.exp(z_topo/delta)
        #                                      - np.exp(z_sl/delta))
        #rad_temp = ((h*delta**2)/k)*(np.exp(z_topo/delta)-np.exp(z/delta))
        #geotherm = rad_temp + (abs(z-z_topo)/abs(z_sl-z_topo))*base_temp
        #geotherm = (
        #    (((H0*delta**2)/k)*(np.exp((Z0*1.e3)/delta)-np.exp((z*1.e3)/delta)))
        #    +(abs((z*1.e3)-(Z0*1.e3))/abs((Zb*1.e3)-(Z0*1.e3)))
        #    *(Tb-((H0*delta**2)/k)*(np.exp((Z0*1.e3)/delta)
        #    - np.exp((Zb*1.e3)/delta)))
        #)
        return geotherm

    def calc_heat_flow(self, z, Z0, Zi, Zm, Zb, Tb):
        Zi = -(Zi-Z0)*1.e3
        Zm = -(Zm-Z0)*1.e3
        Zb = -(Zb-Z0)*1.e3
        z = -(z-Z0)*1.e3
        if self.delta is None:
            self.delta = Zi
        H0, k, delta = self.H0, self.k, self.delta
        heat_flow = (
            +(H0*delta*np.exp(-z/delta))
            +k*(Tb/Zb)
            -(((H0*delta**2)/(Zb))*(1-np.exp(-Zb/delta)))
        )
        ##Z0 = Z0*1.e3
        ##Zb = Zb*1.e3
        ##z = z*1.e3
        ##H0, k, delta = self.H0, self.k, self.delta
        ##base_temp = Tb-((H0*delta**2)/k)*(np.exp(Z0/delta)
        ##                                      - np.exp(Zb/delta))
        ##heat_flow = (-H0*delta*(np.exp(z/delta))-k/(abs(Zb-Z0))*base_temp)

        #heat_flow2 = -(k*temp_sl)/abs(z_sl-z_topo) - (h*delta) + ((h*delta**2)/abs(z_sl-z_topo))*(np.exp(z_topo/delta)-np.exp(z_sl/delta))
        #if np.allclose(heat_flow, heat_flow2, equal_nan=True):
        #    print('Arrays are equal')
        return heat_flow

class TM1(ThermalEquations):
    def __init__(self, H0=3e-06, delta=None, k=2.0, **kwargs):
        super().__init__(**kwargs)
        self.name='TM1'
        self.H0 = H0
        if delta is None:
            self.delta = None
        else:
            self.delta = delta*1.e3
        self.k = k
        self.input = {
            'H0': self.H0,
            'delta': self.delta,
            'k': self.k,
            'constants': self.constants
        }
        # print(self.input)

    def calc_slab_temp(self, Zb, Z0, Zb_i, Z0_i, age):
        age = age*1.e6*365.*24.*60.*60.
        slab_temp = self.calc_slab_temp_internal(Zb, Z0, Zb_i, Z0_i, age, self.k)
        return slab_temp

    def calc_geotherm(self, z, Z0, Zi, Zm, Zb, Tb):
        Zi = -(Zi-Z0)*1.e3
        Zm = -(Zm-Z0)*1.e3
        Zb = -(Zb-Z0)*1.e3
        z = -(z-Z0)*1.e3
        if self.delta is None:
            self.delta = Zi
        gt_c = self.calc_geotherm_crust(z.where((0 <= z) & (z < Zm)), Zm, Zb, Tb)
        gt_m = self.calc_geotherm_mantle(z.where((Zm <= z) & (z <= Zb)), Zm, Zb, Tb)
        geotherm = gt_c.fillna(0) + gt_m.fillna(0)
        geotherm = geotherm.where((0 <= z) & (z <= Zb))
        return geotherm

    def calc_heat_flow(self, z, Z0, Zi, Zm, Zb, Tb):
        Zi = -(Zi-Z0)*1.e3
        Zm = -(Zm-Z0)*1.e3
        Zb = -(Zb-Z0)*1.e3
        z = -(z-Z0)*1.e3
        if self.delta is None:
            self.delta = Zi
        hf_c = self.calc_heat_flow_crust(z.where((0 <= z) & (z < Zm)), Zm, Zb, Tb)
        hf_m = self.calc_heat_flow_mantle(Zm, Zb, Tb).where((Zm <= z) & (z <= Zb))
        heat_flow = hf_c.fillna(0) + hf_m.fillna(0)
        heat_flow = heat_flow.where((0 <= z) & (z <= Zb))
        return heat_flow

    def calc_geotherm_crust(self, z, Zm, Zb, Tb):
        H0, k, delta = self.H0, self.k, self.delta
        geotherm = H0*delta**2/k - H0*delta**2*np.exp(-z/delta)/k - H0*delta*z*np.exp(-Zm/delta)/k + z*(H0*delta*(Zm + delta) + (-H0*delta**2 + Tb*k)*np.exp(Zm/delta))*np.exp(-Zm/delta)/(Zb*k)
        return geotherm

    def calc_geotherm_mantle(self, z, Zm, Zb, Tb):
        H0, k, delta = self.H0, self.k, self.delta
        geotherm = (Tb*k - (H0*delta*(Zm + delta) + (-H0*delta**2 + Tb*k)*np.exp(Zm/delta))*np.exp(-Zm/delta) + z*(H0*delta*(Zm + delta) + (-H0*delta**2 + Tb*k)*np.exp(Zm/delta))*np.exp(-Zm/delta)/Zb)/k
        return geotherm

    def calc_heat_flow_crust(self, z, Zm, Zb, Tb):
        H0, k, delta = self.H0, self.k, self.delta
        heat_flow = k*(H0*delta*np.exp(-z/delta)/k - H0*delta*np.exp(-Zm/delta)/k + (H0*delta*(Zm + delta) + (-H0*delta**2 + Tb*k)*np.exp(Zm/delta))*np.exp(-Zm/delta)/(Zb*k))
        return heat_flow

    def calc_heat_flow_mantle(self, Zm, Zb, Tb):
        H0, k, delta = self.H0, self.k, self.delta
        heat_flow = (H0*delta*(Zm + delta) + (-H0*delta**2 + Tb*k)*np.exp(Zm/delta))*np.exp(-Zm/delta)/Zb
        return heat_flow

class TM2(ThermalEquations):
    def __init__(self, Huc=1.65e-06, Hlc=4e-07, k=2.0, **kwargs):
        super().__init__(**kwargs)
        self.name='TM2'
        self.Huc = Huc
        self.Hlc = Hlc
        self.k = k
        self.input = {
            'Huc': self.Huc,
            'Hlc': self.Hlc,
            'k': self.k,
            'constants': self.constants
        }

    def calc_slab_temp(self, Zb, Z0, Zb_i, Z0_i, age):
        age = age*1.e6*365.*24.*60.*60.
        slab_temp = self.calc_slab_temp_internal(Zb, Z0, Zb_i, Z0_i, age, self.k)
        return slab_temp

    def calc_geotherm(self, z, Z0, Zi, Zm, Zb, Tb):
        Zi = -(Zi-Z0)*1.e3
        Zm = -(Zm-Z0)*1.e3
        Zb = -(Zb-Z0)*1.e3
        z = -(z-Z0)*1.e3
        gt_uc = self.calc_geotherm_upper_crust(z.where((0 <= z) & (z < Zi)), Zi, Zm, Zb, Tb)
        gt_lc = self.calc_geotherm_lower_crust(z.where((Zi <= z) & (z < Zm)), Zi, Zm, Zb, Tb)
        gt_m = self.calc_geotherm_mantle(z.where((Zm <= z) & (z <= Zb)), Zi, Zm, Zb, Tb)
        geotherm = gt_uc.fillna(0) + gt_lc.fillna(0) + gt_m.fillna(0)
        geotherm = geotherm.where((0 <= z) & (z <= Zb))
        return geotherm

    def calc_heat_flow(self, z, Z0, Zi, Zm, Zb, Tb):
        Zi = -(Zi-Z0)*1.e3
        Zm = -(Zm-Z0)*1.e3
        Zb = -(Zb-Z0)*1.e3
        z = -(z-Z0)*1.e3
        hf_uc = self.calc_heat_flow_upper_crust(z.where((0 <= z) & (z < Zi)), Zi, Zm, Zb, Tb)
        hf_lc = self.calc_heat_flow_lower_crust(z.where((Zi <= z) & (z < Zm)), Zi, Zm, Zb, Tb)
        hf_m = self.calc_heat_flow_mantle(Zi, Zm, Zb, Tb).where((Zm <= z) & (z <= Zb))
        heat_flow = hf_uc.fillna(0) + hf_lc.fillna(0) + hf_m.fillna(0)
        heat_flow = heat_flow.where((0 <= z) & (z <= Zb))
        return heat_flow

    def calc_geotherm_upper_crust(self, z, Zi, Zm, Zb, Tb):
        Huc, Hlc, k = self.Huc, self.Hlc, self.k
        geotherm = z*(-2*Hlc*Zi + 2*Hlc*Zm + 2*Huc*Zi - Huc*z + (Hlc*Zi**2 - Hlc*Zm**2 - Huc*Zi**2 + 2*Tb*k)/Zb)/(2*k)
        return geotherm

    def calc_geotherm_lower_crust(self, z, Zi, Zm, Zb, Tb):
        Huc, Hlc, k = self.Huc, self.Hlc, self.k
        geotherm = (Hlc*Zm*z - Hlc*z**2/2 + Zi**2*(-Hlc + Huc)/2 + z*(Hlc*Zi**2 - Hlc*Zm**2 - Huc*Zi**2 + 2*Tb*k)/(2*Zb))/k
        return geotherm

    def calc_geotherm_mantle(self, z, Zi, Zm, Zb, Tb):
        Huc, Hlc, k = self.Huc, self.Hlc, self.k
        geotherm = (-Hlc*Zi**2/2 + Hlc*Zm**2/2 + Huc*Zi**2/2 + z*(Hlc*Zi**2 - Hlc*Zm**2 - Huc*Zi**2 + 2*Tb*k)/(2*Zb))/k
        return geotherm

    def calc_heat_flow_upper_crust(self, z, Zi, Zm, Zb, Tb):
        Huc, Hlc, k = self.Huc, self.Hlc, self.k
        heat_flow = -Hlc*Zi + Hlc*Zm + Huc*Zi - Huc*z + (Hlc*Zi**2 - Hlc*Zm**2 - Huc*Zi**2 + 2*Tb*k)/(2*Zb)
        return heat_flow

    def calc_heat_flow_lower_crust(self, z, Zi, Zm, Zb, Tb):
        Huc, Hlc, k = self.Huc, self.Hlc, self.k
        heat_flow = Hlc*Zm - Hlc*z + (Hlc*Zi**2 - Hlc*Zm**2 - Huc*Zi**2 + 2*Tb*k)/(2*Zb)
        return heat_flow

    def calc_heat_flow_mantle(self, Zi, Zm, Zb, Tb):
        Huc, Hlc, k = self.Huc, self.Hlc, self.k
        heat_flow = (Hlc*Zi**2 - Hlc*Zm**2 - Huc*Zi**2 + 2*Tb*k)/(2*Zb)
        return heat_flow

class TM3(ThermalEquations):
    def __init__(self, Huc=1.65e-06, Hlc=4e-07, kuc=3.0, klcm=1.0, **kwargs):
        super().__init__(**kwargs)
        self.name='TM3'
        self.Huc = Huc
        self.Hlc = Hlc
        self.kuc = kuc
        self.klcm = klcm
        self.input = {
            'Huc': self.Huc,
            'Hlc': self.Hlc,
            'kuc': self.kuc,
            'klcm': self.klcm,
            'constants': self.constants
        }

    def calc_slab_temp(self, Zb, Z0, Zb_i, Z0_i, age):
        age = age*1.e6*365.*24.*60.*60.
        slab_temp = self.calc_slab_temp_internal(Zb, Z0, Zb_i, Z0_i, age, self.klcm)
        return slab_temp

    def calc_geotherm(self, z, Z0, Zi, Zm, Zb, Tb):
        Zi = -(Zi-Z0)*1.e3
        Zm = -(Zm-Z0)*1.e3
        Zb = -(Zb-Z0)*1.e3
        z = -(z-Z0)*1.e3
        gt_uc = self.calc_geotherm_upper_crust(z.where((0 <= z) & (z < Zi)), Zi, Zm, Zb, Tb)
        gt_lc = self.calc_geotherm_lower_crust(z.where((Zi <= z) & (z < Zm)), Zi, Zm, Zb, Tb)
        gt_m = self.calc_geotherm_mantle(z.where((Zm <= z) & (z <= Zb)), Zi, Zm, Zb, Tb)
        geotherm = gt_uc.fillna(0) + gt_lc.fillna(0) + gt_m.fillna(0)
        geotherm = geotherm.where((0 <= z) & (z <= Zb))
        return geotherm

    def calc_heat_flow(self, z, Z0, Zi, Zm, Zb, Tb):
        Zi = -(Zi-Z0)*1.e3
        Zm = -(Zm-Z0)*1.e3
        Zb = -(Zb-Z0)*1.e3
        z = -(z-Z0)*1.e3
        hf_uc = self.calc_heat_flow_upper_crust(z.where((0 <= z) & (z < Zi)), Zi, Zm, Zb, Tb)
        hf_lc = self.calc_heat_flow_lower_crust(z.where((Zi <= z) & (z < Zm)), Zi, Zm, Zb, Tb)
        hf_m = self.calc_heat_flow_mantle(Zi, Zm, Zb, Tb).where((Zm <= z) & (z <= Zb))
        heat_flow = hf_uc.fillna(0) + hf_lc.fillna(0) + hf_m.fillna(0)
        heat_flow = heat_flow.where((0 <= z) & (z <= Zb))
        return heat_flow

    def calc_geotherm_upper_crust(self, z, Zi, Zm, Zb, Tb):
        Huc, Hlc, kuc, klcm = self.Huc, self.Hlc, self.kuc, self.klcm
        geotherm = z*(-2*Hlc*Zi + 2*Hlc*Zm + 2*Huc*Zi - Huc*z + 2*(Hlc*Zi**2*klcm - Hlc*Zi**2*kuc/2 - Hlc*Zi*Zm*klcm + Hlc*Zi*Zm*kuc - Hlc*Zm**2*kuc/2 - Huc*Zi**2*klcm/2 + Tb*klcm*kuc)/(Zb*kuc + Zi*klcm - Zi*kuc))/(2*kuc)
        return geotherm

    def calc_geotherm_lower_crust(self, z, Zi, Zm, Zb, Tb):
        Huc, Hlc, kuc, klcm = self.Huc, self.Hlc, self.kuc, self.klcm
        geotherm = (Hlc*Zm*z - Hlc*z**2/2 + Zi*(-2*Hlc*Zi*klcm + 2*Hlc*Zm*klcm + Huc*Zi*klcm + 2*klcm*(Hlc*Zi**2*klcm - Hlc*Zi**2*kuc/2 - Hlc*Zi*Zm*klcm + Hlc*Zi*Zm*kuc - Hlc*Zm**2*kuc/2 - Huc*Zi**2*klcm/2 + Tb*klcm*kuc)/(Zb*kuc + Zi*klcm - Zi*kuc) + kuc*(Hlc*Zi - 2*Hlc*Zm - 2*(Hlc*Zi**2*klcm - Hlc*Zi**2*kuc/2 - Hlc*Zi*Zm*klcm + Hlc*Zi*Zm*kuc - Hlc*Zm**2*kuc/2 - Huc*Zi**2*klcm/2 + Tb*klcm*kuc)/(Zb*kuc + Zi*klcm - Zi*kuc)))/(2*kuc) + z*(Hlc*Zi**2*klcm - Hlc*Zi**2*kuc/2 - Hlc*Zi*Zm*klcm + Hlc*Zi*Zm*kuc - Hlc*Zm**2*kuc/2 - Huc*Zi**2*klcm/2 + Tb*klcm*kuc)/(Zb*kuc + Zi*klcm - Zi*kuc))/klcm
        return geotherm

    def calc_geotherm_mantle(self, z, Zi, Zm, Zb, Tb):
        Huc, Hlc, kuc, klcm = self.Huc, self.Hlc, self.kuc, self.klcm
        geotherm = (Tb*klcm - Zb*(Hlc*Zi**2*klcm - Hlc*Zi**2*kuc/2 - Hlc*Zi*Zm*klcm + Hlc*Zi*Zm*kuc - Hlc*Zm**2*kuc/2 - Huc*Zi**2*klcm/2 + Tb*klcm*kuc)/(Zb*kuc + Zi*klcm - Zi*kuc) + z*(Hlc*Zi**2*klcm - Hlc*Zi**2*kuc/2 - Hlc*Zi*Zm*klcm + Hlc*Zi*Zm*kuc - Hlc*Zm**2*kuc/2 - Huc*Zi**2*klcm/2 + Tb*klcm*kuc)/(Zb*kuc + Zi*klcm - Zi*kuc))/klcm
        return geotherm

    def calc_heat_flow_upper_crust(self, z, Zi, Zm, Zb, Tb):
        Huc, Hlc, kuc, klcm = self.Huc, self.Hlc, self.kuc, self.klcm
        heat_flow = -Hlc*Zi + Hlc*Zm + Huc*Zi - Huc*z + (Hlc*Zi**2*klcm - Hlc*Zi**2*kuc/2 - Hlc*Zi*Zm*klcm + Hlc*Zi*Zm*kuc - Hlc*Zm**2*kuc/2 - Huc*Zi**2*klcm/2 + Tb*klcm*kuc)/(Zb*kuc + Zi*klcm - Zi*kuc)
        return heat_flow

    def calc_heat_flow_lower_crust(self, z, Zi, Zm, Zb, Tb):
        Huc, Hlc, kuc, klcm = self.Huc, self.Hlc, self.kuc, self.klcm
        heat_flow = Hlc*Zm - Hlc*z + (Hlc*Zi**2*klcm - Hlc*Zi**2*kuc/2 - Hlc*Zi*Zm*klcm + Hlc*Zi*Zm*kuc - Hlc*Zm**2*kuc/2 - Huc*Zi**2*klcm/2 + Tb*klcm*kuc)/(Zb*kuc + Zi*klcm - Zi*kuc)
        return heat_flow

    def calc_heat_flow_mantle(self, Zi, Zm, Zb, Tb):
        Huc, Hlc, kuc, klcm = self.Huc, self.Hlc, self.kuc, self.klcm
        heat_flow = (Hlc*Zi**2*klcm - Hlc*Zi**2*kuc/2 - Hlc*Zi*Zm*klcm + Hlc*Zi*Zm*kuc - Hlc*Zm**2*kuc/2 - Huc*Zi**2*klcm/2 + Tb*klcm*kuc)/(Zb*kuc + Zi*klcm - Zi*kuc)
        return heat_flow
