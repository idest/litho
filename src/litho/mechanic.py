import numpy as np
import xarray as xr
from litho.data import rhe_data_setup

rhe_data = rhe_data_setup()


class MechanicEquations(object):
    def __init__(
        self,
        constants=None,
        Bs_t=20.0e3,
        Bs_c=-55.0e3,
        e=1.0e-15,
        R=8.31,
        s_max=200,
        **kwargs
    ):
        # print('...MechanicEquations init')
        if constants is not None:
            self.Bs_t = constants["Bs_t"]
            self.Bs_c = constants["Bs_c"]
            self.e = constants["e"]
            self.R = constants["R"]
            self.s_max = constants["s_max"]
        else:
            self.Bs_t = Bs_t
            self.Bs_c = Bs_c
            self.e = e
            self.R = R
            self.s_max = s_max

    def calc_brittle_yield_strength(self, bs, depth):
        bys = bs * (depth) * 1.0e-6
        return bys

    def calc_ductile_yield_strength(self, es, n, a, h, r, temp):
        dys = (es / a) ** (1 / n) * np.exp(h / (n * r * (temp + 273.15))) * 1.0e-6
        return dys

    def calc_brittle_yield_depth(self, bs, bys):
        # depth from brittle yield strength
        depth = bys / (bs * 1.0e3 * 1.0e-6)
        return depth

    def calc_ductile_yield_temperature(self, es, n, a, h, r, dys):
        # temperature from ductile yield strength
        temp = h / (n * r * np.log(dys / (1.0e-6 * (es / a) ** (1 / n)))) - 273.15
        return temp

    def calc_polyphase_rheologic_vars(self, f1, rheo_1, rheo_2):
        # Based on Tullis et al., 1991
        f2 = 1 - f1
        n1 = rheo_1["n"]
        a1 = rheo_1["A"]
        h1 = rheo_1["H"]
        n2 = rheo_2["n"]
        a2 = rheo_2["A"]
        h2 = rheo_2["H"]
        n = 10 ** (f1 * np.log10(n1) + f2 * np.log10(n2))
        a = 10 ** ((np.log10(a2) * (n - n1) - np.log10(a1) * (n - n2)) / (n2 - n1))
        h = (h2 * (n - n1) - h1 * (n - n2)) / (n2 - n1)
        rheo = {
            "n": n,
            "A": a,
            "H": h,
        }
        return rheo

    def calc_eet_detached(self, ths):
        # detached_ths[np.isnan(ths)] = 0
        ths = ths.fillna(0)
        eet = (ths[:, :, 0] ** 3 + ths[:, :, 1] ** 3 + ths[:, :, 2] ** 3) ** (1 / 3)
        return eet


class MM(MechanicEquations):
    def __init__(self, uc=1, lc=14, lm=30, serp=23, serp_pct=0.65, **kwargs):
        super().__init__(**kwargs)
        self.uc = uc
        self.lc = lc
        self.lm = lm
        self.serp = serp
        self.serp_pct = serp_pct

    def get_rheo_info(self):
        uc_rheo, lc_rheo, lm_rheo, serp_rheo = [
            {key: rhe_data[str(rheo_id)][key] for key in ["n", "A", "H"]}
            if rheo_id is not None
            else None
            for rheo_id in [self.uc, self.lc, self.lm, self.serp]
        ]
        return {"uc": uc_rheo, "lc": lc_rheo, "lm": lm_rheo, "serp": serp_rheo}

    def z_rheo(self, z, Z0, Zi, Zm, Zb, serp_domain):
        # r = z.copy(data=np.ones(z.shape)*np.nan).expand_dims(
        #    dim={'rheo': ['n', 'A', 'H']}, axis=3
        # )
        r = (
            xr.full_like(z, np.nan)
            .expand_dims(dim={"var": ["n", "A", "H"]}, axis=3)
            .copy()
        )
        # uc_rheo, lc_rheo, lm_rheo, serp_rheo = [
        #    {key: rhe_data[str(rheo_id)][key] for key in ['n', 'A', 'H']}
        #    if rheo_id is not None else None
        #    for rheo_id in [self.uc, self.lc, self.lm, self.serp]
        # ]
        rheos = self.get_rheo_info()
        r = xr.where((0 <= z) & (z < Zi), list(rheos["uc"].values()), r)
        r = xr.where((Zi <= z) & (z < Zm), list(rheos["lc"].values()), r)
        r = xr.where((Zm <= z) & (z <= Zb), list(rheos["lm"].values()), r)
        if self.serp is not None:
            flm_rheo = self.calc_polyphase_rheologic_vars(
                self.serp_pct, rheos["serp"], rheos["lm"]
            )
            r = xr.where(
                (Zm <= z) & (z <= Zb) & (serp_domain == 1), list(flm_rheo.values()), r
            )
        return r

    def calc_brittle_yield_strength_envelope(self, z, Z0):
        z = -(z - Z0) * 1.0e3
        byse_t = self.calc_brittle_yield_strength(self.Bs_t, z)
        byse_c = self.calc_brittle_yield_strength(self.Bs_c, z)
        return (byse_t, byse_c)

    def calc_ductile_yield_strength_envelope(
        self, z, Z0, Zi, Zm, Zb, serp_domain, temp
    ):
        Zi = -(Zi - Z0) * 1.0e3
        Zm = -(Zm - Z0) * 1.0e3
        Zb = -(Zb - Z0) * 1.0e3
        z = -(z - Z0) * 1.0e3
        z_rheo = self.z_rheo(z, Z0, Zi, Zm, Zb, serp_domain)
        dyse = self.calc_ductile_yield_strength(
            self.e,
            z_rheo.loc[{"var": "n"}],
            z_rheo.loc[{"var": "A"}],
            z_rheo.loc[{"var": "H"}],
            self.R,
            temp,
        )
        return dyse

    def calc_yield_strength_envelope(self, byse, dyse, cond="tension"):
        byse_t, byse_c = byse
        yse_t = xr.where(byse_t < dyse, byse_t, dyse)
        yse_c = xr.where(byse_c > -dyse, byse_c, -dyse)
        if cond == "tension":
            yse = yse_t
        elif cond == "compression":
            yse = -yse_c
        else:
            yse = (yse_t, yse_c)
        return yse

    def calc_yield_depth_00(self, dyse):
        depth = xr.where(dyse >= self.s_max, dyse, np.nan).idxmin(
            dim="depth", skipna=True
        )
        return depth

    def calc_yield_depths(self, z, Z0, geotherm, isosurface_func):
        z = -(z - Z0) * 1.0e3
        ## brittle
        brittle_yield_depths = xr.DataArray(
            coords=[
                ("lon", z["lon"].data),
                ("lat", z["lat"].data),
                ("state", ["tension", "compression"]),
            ]
        )
        for key, Bs in {"tension": self.Bs_t, "compression": self.Bs_c}.items():
            brittle_yield_depths.loc[:, :, key] = Z0 - self.calc_brittle_yield_depth(
                Bs, self.s_max
            )
        ## ductile
        rheos = self.get_rheo_info()
        if self.serp is not None:
            rheos["flm"] = self.calc_polyphase_rheologic_vars(
                self.serp_pct, rheos["serp"], rheos["lm"]
            )
        del rheos["serp"]  # rheos.pop('serp', None)
        temp_rheos = {}
        for rheo_key, rheo in rheos.items():
            temp_rheos[rheo_key] = self.calc_ductile_yield_temperature(
                self.e, rheo["n"], rheo["A"], rheo["H"], self.R, self.s_max
            )
        ductile_yield_temps = xr.DataArray(
            np.array(list(temp_rheos.values())),
            coords=[("rheo", list(temp_rheos.keys()))],
        )
        ductile_yield_depths = xr.DataArray(
            coords=[
                ("lon", z["lon"].data),
                ("lat", z["lat"].data),
                ("rheo", ductile_yield_temps["rheo"].data),
            ]
        )
        for rheo_key in rheos.keys():
            ductile_yield_depths.loc[:, :, rheo_key] = isosurface_func(
                geotherm,
                ductile_yield_temps.loc[rheo_key],
                lower=np.inf,
                higher=-np.inf,
            )
        yield_depths = {
            "brittle": brittle_yield_depths,
            "ductile": ductile_yield_depths,
        }
        return yield_depths

    def calc_elastic_tuples(self, Z0, Zi, Zm, Zb, yield_depths, state):
        tuples = {}
        for rheo_key in yield_depths["ductile"]["rheo"].values:
            if rheo_key == "uc":
                top_boundary = Z0
                bottom_boundary = Zi
            elif rheo_key == "lc":
                top_boundary = Zi
                bottom_boundary = Zm
            elif rheo_key == "lm" or rheo_key == "flm":
                top_boundary = Zm
                bottom_boundary = Zb
            top_elastic = xr.ufuncs.minimum(
                top_boundary, yield_depths["brittle"].loc[:, :, state]
            )
            bottom_elastic = xr.ufuncs.maximum(
                bottom_boundary, yield_depths["ductile"].loc[:, :, rheo_key]
            )
            thickness = top_elastic - bottom_elastic
            top_elastic = top_elastic.where(thickness > 0)
            bottom_elastic = bottom_elastic.where(thickness > 0)
            thickness = thickness.where(thickness > 0)
            elastic_tuple = xr.DataArray(
                np.stack((top_elastic, bottom_elastic, thickness), axis=2),
                coords=[
                    ("lon", Z0["lon"].data),
                    ("lat", Z0["lat"].data),
                    ("label", ["top", "bottom", "thickness"]),
                ],
            )
            tuples[rheo_key] = elastic_tuple
        return tuples

    def calc_eet(self, elastic_tuples):
        es = elastic_tuples
        pass
        # Get the coupled and decoupled zones for moho and icd
        share_moho = es["lc"].loc[:, :, "bottom"] == es["lm"].loc[:, :, "top"]
        share_icd = es["uc"].loc[:, :, "bottom"] == es["lc"].loc[:, :, "top"]
        layers_thickness_i = xr.DataArray(
            np.stack(
                (
                    es["uc"].loc[:, :, "thickness"],
                    es["lc"].loc[:, :, "thickness"],
                    es["lm"].loc[:, :, "thickness"],
                ),
                axis=2,
            ),
            coords=[
                ("lon", es["uc"]["lon"].data),
                ("lat", es["uc"]["lat"].data),
                ("rheo", ["uc", "lc", "lm"]),
            ],
        )
        # If icd is coupled sum thickness of upper crust with lower crust
        c_icd_ths = layers_thickness_i.where(share_icd)
        d_icd_ths = layers_thickness_i.where(~share_icd)
        c_icd_ths.loc[:, :, "lc"] = (
            c_icd_ths.loc[:, :, "uc"] + c_icd_ths.loc[:, :, "lc"]
        )
        c_icd_ths.loc[:, :, "uc"] = 0
        layers_thickness = xr.where(share_icd, c_icd_ths, d_icd_ths)
        # If moho is coupled sum thickness of lower crust with litospheric mantle
        c_moho_ths = layers_thickness_i.where(share_moho)
        d_moho_ths = layers_thickness_i.where(~share_moho)
        c_moho_ths.loc[:, :, "lm"] = (
            c_moho_ths.loc[:, :, "lc"] + c_moho_ths.loc[:, :, "lm"]
        )
        c_moho_ths.loc[:, :, "lc"] = 0
        layers_thickness = xr.where(share_moho, c_moho_ths, d_moho_ths)
        # EET
        eet = self.calc_eet_detached(layers_thickness)
        return eet

    def calc_integrated_strength(self, yse):
        return yse.sum(dim="depth", skipna=True, min_count=1)
