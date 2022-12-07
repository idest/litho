import numpy as np
import xarray as xr
import pandas as pd
from litho.data import geom_data_setup, age_data_setup, shf_data_setup
from litho.stats import evaluate_model
from litho.thermal import TM1, TM2, TM3
from litho.mechanic import MM
from litho.plots import plot


class Lithosphere(object):
    def __init__(self, thermal_model=None, mechanic_model=None, space=None, age=None):
        self.data = {
            "geometry": geom_data_setup(),
            "age": age_data_setup(),
            "shf": shf_data_setup(),
        }
        self.geometry = self.init_geometry()
        self.thermal_model = thermal_model
        self.mechanic_model = mechanic_model
        self.space = space
        self.state = {
            "age": self.set_age(age=age),
            "Tb": self.base_temperature(thermal_model=thermal_model),
        }
        # print(self.state["age"])
        self.plot = plot

    def init_geometry(self):
        geometry = xr.Dataset(
            {
                "Z0": (
                    ("lon", "lat"),
                    self.data["geometry"].topo,
                    {"name": "Topography level", "units": "kilometers"},
                ),
                "Zi": (
                    ("lon", "lat"),
                    self.data["geometry"].icd,
                    {"name": "ICD", "units": "kilometers"},
                ),
                "Zm": (
                    ("lon", "lat"),
                    self.data["geometry"].moho,
                    {"name": "Moho", "units": "kilometers"},
                ),
                "Zb": (
                    ("lon", "lat"),
                    self.data["geometry"].slab_lab,
                    {"name": "Lithosphere base", "units": "kilometers"},
                ),
                "lab_domain": (
                    ("lat", "lon"),
                    self.data["geometry"].lab_domain,
                    {"name": "LAB domain"},
                ),
            },
            coords={
                "lon": self.data["geometry"].lon_axis,
                "lat": self.data["geometry"].lat_axis,
            },
        )
        geometry["lab_start"] = geometry["lab_domain"].idxmax(dim="lon")
        return geometry

    def set_age(self, age=None):
        if age is None:
            age = self.data["age"][:, 1]
        elif isinstance(age, list):
            age = age
        else:
            age = np.repeat(age, len(self.state["lat"]))
        age = xr.DataArray(
            age,
            coords={"lat": self.data["geometry"].lat_axis},
            attrs={"name": "Trench age", "units": "million years"},
        )
        return age

    def topo(self):
        return self.geometry["Z0"]

    def icd(self):
        return self.geometry["Zi"]

    def moho(self):
        return self.geometry["Zm"]

    def slab_lab(self):
        return self.geometry["Zb"]

    def boundaries(self):
        return {
            "Z0": self.geometry["Z0"],
            "Zi": self.geometry["Zi"],
            "Zm": self.geometry["Zm"],
            "Zb": self.geometry["Zb"],
        }

    def lab_domain(self):
        return self.geometry["lab_domain"]

    def lab_start(self):
        return self.geometry["lab_start"]

    def depths_array(self, input_bounds=None):
        coords = {}
        coords["lon"] = self.geometry["lon"]
        coords["lat"] = self.geometry["lat"]
        coords["depth"] = xr.DataArray(
            self.data["geometry"].depth_axis,
            coords=[("depth", self.data["geometry"].depth_axis)],
        )
        bounds = {
            "lat": (np.inf, -np.inf),
            "lon": (-np.inf, np.inf),
            "depth": (np.inf, -np.inf),
        }
        if input_bounds is not None:
            for key, val in input_bounds.items():
                bounds[key] = val
        for key, val in bounds.items():
            if isinstance(val, xr.DataArray):
                coords[key] = val
            elif isinstance(val, tuple):
                coords[key] = coords[key].loc[val[0] : val[1]]
            else:
                coords[key] = coords[key].loc[val:val]
        XY = xr.DataArray(
            coords=[("lon", coords["lon"].data), ("lat", coords["lat"].data)]
        )
        Z = coords["depth"].reindex_like(XY).broadcast_like(XY)
        return xr.Dataset({"Z": Z})

    def set_depths_array(self, input_bounds=None):
        self.space = self.depths_array(input_bounds)

    def isosurface(self, array, value, lower=None, higher=None):
        # Variable                                     # Example a, Example b
        #########################################################################
        # array                                     #a) [ 50, 100, 150, 200, 250]
        #                                           #b) [250, 200, 150, 100,  50]
        # value                                     #   135
        Nm = len(array["depth"]) - 1

        lslice = {"depth": slice(None, -1)}  #
        uslice = {"depth": slice(1, None)}

        prop = array - value  # a) [-85, -35,  15,  65, 115]
        #                                           #b) [115,  65,  15, -35, -85]

        propl = prop.isel(**lslice)  # a) [-85, -35,  15,  65]
        #                                           #b) [115,  65,  15, -35]
        propl.coords["depth"] = np.arange(Nm)
        propu = prop.isel(**uslice)  # a) [-35,  15,  65, 115]
        #                                           #b) [ 65,  15, -35, -85]
        propu.coords["depth"] = np.arange(Nm)

        # propu*propl                               #a) [ +X,  -X,  +X,  +X]
        #                                           #b) [ +X,  +X,  -X,  +X]

        zc = xr.where((propu * propl) < 0.0, 1.0, 0.0)  # a) [0.0, 1.0, 0.0, 0.0]
        #                                           #b) [0.0, 0.0, 1.0, 0.0]

        Z = self.depths_array(array.coords)["Z"]  #   [ 10,  20,  30,  40,  50]
        #                                           #

        varl = Z.isel(**lslice)  #   [ 10,  20,  30,  40]
        varl.coords["depth"] = np.arange(Nm)
        varu = Z.isel(**uslice)  #   [ 20,  30,  40,  50]
        varu.coords["depth"] = np.arange(Nm)

        propl = (propl * zc).sum("depth")  # a) sum([  0, -35,   0,   0]) = -35
        #                                       #b) sum([  0    0,  15,   0]) =  15

        propu = (propu * zc).sum("depth")  # a) sum([  0,  15,   0,   0]) =  15
        #                                       #b) sum([  0,   0, -35,   0]) = -35

        varl = (varl * zc).sum("depth")  #   sum([  0,  20,   0,   0]) =  20

        varu = (varu * zc).sum("depth")  #   sum([  0,  30,   0,   0]) =  30

        iso = varl + (-propl) * (varu - varl) / (
            propu - propl
        )  # 20+(35)*((30-20)/(15+35))
        #                                             # 20+(-15)*((30-20)/(-35-15))
        if lower is not None:
            lower_mask = prop.min(dim="depth") > 0
            iso = iso.where(~lower_mask, other=lower)
        if higher is not None:
            higher_mask = prop.max(dim="depth") < 0
            iso = iso.where(~higher_mask, other=higher)
        return iso

    def set_thermal_state(self, thermal_model=None):
        # print("...init_thermal_state")
        print(".", end="", flush=True)
        if thermal_model is None:
            thermal_model = TM1
        self.thermal_model = thermal_model
        self.state["Tb"] = self.base_temperature()

    def set_mechanic_state(self, mechanic_model=None):
        if mechanic_model is None:
            mechanic_model = MM(uc=1, lc=14, lm=30, serp=23, serp_pct=0.65)
        self.mechanic_model = mechanic_model

    def base_temperature(self, thermal_model=None):
        # TODO: think about this thermal_model is None logic
        # for cases in which thermal_model = None, thermal_model = self.thermal_model,
        # and thermal_model = TM1(...)
        if thermal_model is None:
            thermal_model = self.thermal_model
        if thermal_model is not None:
            slab_temp = thermal_model.calc_slab_temp(
                self.geometry["Zb"].where(self.geometry["lab_domain"] == 0),
                self.geometry["Z0"].where(self.geometry["lab_domain"] == 0),
                self.geometry["Zb"].sel(lon=self.geometry["lab_start"]),
                self.geometry["Z0"].sel(lon=self.geometry["lab_start"]),
                self.state["age"],
            )
            lab_temp = thermal_model.calc_lab_temp(
                self.geometry["Zb"].where(self.geometry["lab_domain"] == 1),
                self.geometry["Z0"].where(self.geometry["lab_domain"] == 1),
            )
            base_temperature = slab_temp.combine_first(lab_temp)
            base_temperature = base_temperature.assign_attrs(
                {"name": "Lithosphere base temperature", "units": "celsius"}
            )
        else:
            base_temperature = None
        return base_temperature

    def geotherm(self, input_bounds=None, thermal_model=None):
        if input_bounds is None:
            Z = self.space["Z"]
        else:
            Z = self.depths_array(input_bounds)["Z"]
        if thermal_model is None:
            thermal_model = self.thermal_model
            base_temp = self.state["Tb"]
        else:
            base_temp = self.base_temperature(thermal_model=thermal_model)
        geotherm = thermal_model.calc_geotherm(
            Z,
            self.geometry["Z0"],
            self.geometry["Zi"],
            self.geometry["Zm"],
            self.geometry["Zb"],
            base_temp,
        )
        geotherm = geotherm.assign_attrs({"name": "Geotherm", "units": "celsius"})
        return geotherm

    def heat_flow(self, input_bounds=None, thermal_model=None):
        if input_bounds is None:
            Z = self.space["Z"]
        else:
            Z = self.depths_array(input_bounds)["Z"]
        if thermal_model is None:
            thermal_model = self.thermal_model
            base_temp = self.state["Tb"]
        else:
            base_temp = self.base_temperature(thermal_model=thermal_model)
        heat_flow = (
            thermal_model.calc_heat_flow(
                Z,
                self.geometry["Z0"],
                self.geometry["Zi"],
                self.geometry["Zm"],
                self.geometry["Zb"],
                base_temp,
            )
            * 1.0e3
        )
        heat_flow = heat_flow.assign_attrs({"name": "Heat flow", "units": "milliwatts"})
        return heat_flow

    def surface_heat_flow(self, thermal_model=None):
        if thermal_model is None:
            thermal_model = self.thermal_model
            base_temp = self.state["Tb"]
        else:
            base_temp = self.base_temperature(thermal_model=thermal_model)
        heat_flow = (
            thermal_model.calc_heat_flow(
                self.geometry["Z0"],
                self.geometry["Z0"],
                self.geometry["Zi"],
                self.geometry["Zm"],
                self.geometry["Zb"],
                base_temp,
            )
            * 1.0e3
        )
        heat_flow = heat_flow.assign_attrs(
            {"name": "Surface heat flow", "units": "milliwatts"}
        )
        return heat_flow

    def serp_domain(self, thermal_model=None):
        if thermal_model is None:
            thermal_model = self.thermal_model
        # TODO: find a better way to get serp_domain
        serp_domain = xr.DataArray(
            np.nan,
            coords=[
                ("lon", self.geometry["lon"].data),
                ("lat", self.geometry["lat"].data),
            ],
        )
        if thermal_model.name == "TM1":
            max_heat_flow_moho_lon = self.heat_flow(
                {"depth": self.moho()}, thermal_model=thermal_model
            ).idxmax("lon")
            min_depth_moho_lon = self.moho().idxmin("lon")
            serp_domain1 = xr.where(serp_domain["lon"] < max_heat_flow_moho_lon, 1, 0)
            serp_domain2 = xr.where(serp_domain["lon"] < min_depth_moho_lon, 1, 0)
            serp_domain = xr.where(
                serp_domain2 > serp_domain1, serp_domain2, serp_domain1
            )
        else:
            max_heat_flow_icd_lon = self.heat_flow(
                {"depth": self.icd()}, thermal_model=thermal_model
            ).idxmax("lon")
            min_depth_moho_lon = self.moho().idxmin("lon")
            serp_domain1 = xr.where(serp_domain["lon"] < max_heat_flow_icd_lon, 1, 0)
            serp_domain2 = xr.where(serp_domain["lon"] < min_depth_moho_lon, 1, 0)
            serp_domain = xr.where(
                serp_domain2 > serp_domain1, serp_domain2, serp_domain1
            )
        return serp_domain

    def brittle_yield_strength_envelope(self, input_bounds=None, mechanic_model=None):
        if input_bounds is None:
            Z = self.space["Z"]
        else:
            Z = self.depths_array(input_bounds)["Z"]
        if mechanic_model is None:
            mechanic_model = self.mechanic_model
        byse = mechanic_model.calc_brittle_yield_strength_envelope(
            Z, self.geometry["Z0"]
        )
        return byse

    def ductile_yield_strength_envelope(
        self, input_bounds=None, thermal_model=None, mechanic_model=None
    ):
        if input_bounds is None:
            Z = self.space["Z"]
        else:
            Z = self.depths_array(input_bounds)["Z"]
        if mechanic_model is None:
            mechanic_model = self.mechanic_model
        serp_domain = self.serp_domain(thermal_model=thermal_model)
        geotherm = self.geotherm(input_bounds=input_bounds, thermal_model=thermal_model)
        dyse = mechanic_model.calc_ductile_yield_strength_envelope(
            Z,
            self.geometry["Z0"],
            self.geometry["Zi"],
            self.geometry["Zm"],
            self.geometry["Zb"],
            serp_domain,
            geotherm,
        )
        return dyse

    def yield_strength_envelope(
        self, input_bounds=None, cond="tension", thermal_model=None, mechanic_model=None
    ):
        if input_bounds is None:
            Z = self.space["Z"]
        else:
            Z = self.depths_array(input_bounds)["Z"]
        byse = self.brittle_yield_strength_envelope(
            input_bounds=input_bounds, mechanic_model=mechanic_model
        )
        dyse = self.ductile_yield_strength_envelope(
            input_bounds=input_bounds,
            thermal_model=thermal_model,
            mechanic_model=mechanic_model,
        )
        # cond = 'tension'
        yse = mechanic_model.calc_yield_strength_envelope(byse, dyse, cond=cond)
        yse = yse.assign_attrs({"name": "Yield strength envelope", "units": "MPa"})
        return yse

    def yield_depths(self, input_bounds=None, thermal_model=None, mechanic_model=None):
        if input_bounds is None:
            Z = self.space["Z"]
        else:
            Z = self.depths_array(input_bounds)["Z"]
        if mechanic_model is None:
            mechanic_model = self.mechanic_model
        # byse = self.brittle_yield_strength_envelope(input_bounds=input_bounds)
        # dyse = self.ductile_yield_strength_envelope(input_bounds=input_bounds)
        geotherm = self.geotherm(input_bounds=input_bounds, thermal_model=thermal_model)
        yield_depths = mechanic_model.calc_yield_depths(
            Z, self.geometry["Z0"], geotherm, self.isosurface
        )
        return yield_depths

    def elastic_tuples(
        self,
        input_bounds=None,
        state="compression",
        thermal_model=None,
        mechanic_model=None,
    ):
        if input_bounds is None:
            Z = self.space["Z"]
        else:
            Z = self.depths_array(input_bounds)["Z"]
        yield_depths = self.yield_depths(
            input_bounds=input_bounds,
            thermal_model=thermal_model,
            mechanic_model=mechanic_model,
        )
        if mechanic_model is None:
            mechanic_model = self.mechanic_model
        elastic_tuples = mechanic_model.calc_elastic_tuples(
            self.geometry["Z0"],
            self.geometry["Zi"],
            self.geometry["Zm"],
            self.geometry["Zb"],
            yield_depths,
            state,
        )
        return elastic_tuples

    def eet(self, input_bounds=None, thermal_model=None, mechanic_model=None):
        if input_bounds is None:
            Z = self.space["Z"]
        else:
            Z = self.depths_array(input_bounds)["Z"]
        elastic_tuples = self.elastic_tuples(
            input_bounds=input_bounds,
            thermal_model=thermal_model,
            mechanic_model=mechanic_model,
        )
        if mechanic_model is None:
            mechanic_model = self.mechanic_model
        eet = mechanic_model.calc_eet(elastic_tuples)
        eet = eet.assign_attrs({"name": "Equivalent Elastic Thickness", "units": "ET"})
        return eet

    def integrated_strength(
        self, input_bounds=None, thermal_model=None, mechanic_model=None
    ):
        if input_bounds is None:
            Z = self.space["Z"]
        else:
            Z = self.depths_array(input_bounds)["Z"]
        yse = self.yiel.d_strength_envelope(
            input_bounds=input_bounds,
            thermal_model=thermal_model,
            mechanic_model=mechanic_model,
        )
        if mechanic_model is None:
            mechanic_model = self.mechanic_model
        integrated_strength = mechanic_model.calc_integrated_strength(yse)
        integrated_strength = integrated_strength.assign_attrs(
            {"name": "Integrated Strength", "units": "integrated MPa"}
        )
        return integrated_strength

    # def compare(self, array1, array2):
    #    diff = array1 - array2
    #    diff = diff.assign_attrs(
    #        {'name': 'Difference', 'units': 'Km. diff.'}
    #    )
    #    return diff

    def export_csv(self, darray, filename, name="prop", dropna=False):
        df = darray.to_dataframe(name=name)
        if dropna is True:
            df = df.dropna()
        df.to_csv(filename, sep=" ", na_rep="nan", float_format="%.2f")

    def import_csv(self, filename, name="prop"):
        df = pd.read_csv(
            filename,
            sep="\s+",
            # names=['lon','lat',name],
            index_col=[0, 1],
        )
        return df.to_xarray()

    def export_netcdf(self, filename):
        self.geometry.to_netcdf(filename)

    def stats(self, shf_data=None):
        if shf_data is None:
            shf_data = self.data["shf"]
        shf = self.surface_heat_flow()
        shf_model = shf.interp(lon=shf_data["Long"], lat=shf_data["Lat"])
        shf_model = shf_model.to_dataframe(name="SHF_model")
        shf_data = shf_data.to_dataframe()
        shf_data["SHF_model"] = shf_model["SHF_model"]
        shf_data = shf_data.dropna()
        estimators, df = evaluate_model(shf_data, return_dataframe=True)
        return estimators, df


if __name__ == "__main__":
    pass
