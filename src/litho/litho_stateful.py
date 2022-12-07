import numpy as np
import xarray as xr
import pandas as pd
from litho.data import Data, data_setup, shf_data_setup
from litho.stats import evaluate_model
from litho.thermal import TM1, TM2, TM3
from litho.mechanic import MM
from litho.plots import plot

class Lithosphere(object):

    def __init__(self, space=None):
        self.data = Data(*data_setup())
        self.shf_data = shf_data_setup()
        self.state = self.init_state()
        self.depth_axis = xr.DataArray(self.data.depth_axis)
        self.space = None
        self.plot = plot

    def init_state(self):
        state = xr.Dataset(
            {
                'Z0': (
                    ('lon', 'lat'),
                    self.data.topo,
                    {'name': 'Topography level', 'units': 'kilometers'}
                ),
                'Zi': (
                    ('lon', 'lat'),
                    self.data.icd,
                    {'name': 'ICD', 'units': 'kilometers'}
                ),
                'Zm': (
                    ('lon', 'lat'),
                    self.data.moho,
                    {'name': 'Moho', 'units': 'kilometers'}
                ),
                'Zb': (
                    ('lon', 'lat'),
                    self.data.slab_lab,
                    {'name': 'Lithosphere base', 'units': 'kilometers'}
                ),
                'lab_domain': (
                    ('lat', 'lon'),
                    self.data.lab_domain,
                    {'name': 'LAB domain'}
                )
            },
            coords={
                'lon': self.data.lon_axis,
                'lat': self.data.lat_axis,
            },
        )
        state['lab_start'] = state['lab_domain'].idxmax(dim='lon')
        return state

    def get_topo(self):
        return self.state['Z0']

    def get_icd(self):
        return self.state['Zi']

    def get_moho(self):
        return self.state['Zm']

    def get_slab_lab(self):
        return self.state['Zb']

    def get_boundaries(self):
        return {
            'Z0': self.state['Z0'],
            'Zi': self.state['Zi'],
            'Zm': self.state['Zm'],
            'Zb': self.state['Zb']
        }

    def get_lab_domain(self):
        return self.state['lab_domain']

    def get_lab_start(self):
        return self.state['lab_start']

    def get_depths_array(self, input_bounds=None):
        coords = {}
        coords['lon'] = self.state['lon']
        coords['lat'] = self.state['lat']
        coords['depth'] = xr.DataArray(
            self.data.depth_axis,
            coords=[('depth', self.data.depth_axis)]
        )
        bounds = {
            'lat': (np.inf, -np.inf),
            'lon': (-np.inf, np.inf),
            'depth': (np.inf, -np.inf),
        }
        if input_bounds is not None:
            for key, val in input_bounds.items():
                bounds[key] = val
        for key, val in bounds.items():
            if isinstance(val, xr.DataArray):
                coords[key] = val
            elif isinstance(val, tuple):
                coords[key] = coords[key].loc[val[0]:val[1]]
            else:
                coords[key] = coords[key].loc[val:val]
        XY = xr.DataArray(
            coords=[('lon', coords['lon'].data), ('lat', coords['lat'].data)]
        )
        Z = coords['depth'].reindex_like(XY).broadcast_like(XY)
        return xr.Dataset({'Z': Z})

    def set_depths_array(self, input_bounds=None):
        self.space = self.get_depths_array(input_bounds)

    def isosurface(self, array, value, lower=None, higher=None):
        # Variable                                     # Example a, Example b
        #########################################################################
        # array                                     #a) [ 50, 100, 150, 200, 250]
        #                                           #b) [250, 200, 150, 100,  50]
        # value                                     #   135
        Nm = len(array['depth']) - 1

        lslice = {'depth': slice(None, -1)}           #
        uslice = {'depth': slice(1, None)}

        prop = array - value                        #a) [-85, -35,  15,  65, 115]
        #                                           #b) [115,  65,  15, -35, -85]

        propl = prop.isel(**lslice)                 #a) [-85, -35,  15,  65]
        #                                           #b) [115,  65,  15, -35]
        propl.coords['depth'] = np.arange(Nm)
        propu = prop.isel(**uslice)                 #a) [-35,  15,  65, 115]
        #                                           #b) [ 65,  15, -35, -85]
        propu.coords['depth'] = np.arange(Nm)

        # propu*propl                               #a) [ +X,  -X,  +X,  +X]
        #                                           #b) [ +X,  +X,  -X,  +X]

        zc = xr.where((propu*propl)<0.0, 1.0, 0.0)  #a) [0.0, 1.0, 0.0, 0.0]
        #                                           #b) [0.0, 0.0, 1.0, 0.0]

        Z = self.get_depths_array(array.coords)['Z']#   [ 10,  20,  30,  40,  50]
        #                                           #

        varl = Z.isel(**lslice)                     #   [ 10,  20,  30,  40]
        varl.coords['depth'] = np.arange(Nm)
        varu = Z.isel(**uslice)                     #   [ 20,  30,  40,  50]
        varu.coords['depth'] = np.arange(Nm)

        propl = (propl*zc).sum('depth')         #a) sum([  0, -35,   0,   0]) = -35
        #                                       #b) sum([  0    0,  15,   0]) =  15

        propu = (propu*zc).sum('depth')         #a) sum([  0,  15,   0,   0]) =  15
        #                                       #b) sum([  0,   0, -35,   0]) = -35

        varl = (varl*zc).sum('depth')           #   sum([  0,  20,   0,   0]) =  20

        varu = (varu*zc).sum('depth')           #   sum([  0,  30,   0,   0]) =  30

        iso = varl+(-propl)*(varu-varl)/(propu-propl) # 20+(35)*((30-20)/(15+35))
        #                                             # 20+(-15)*((30-20)/(-35-15))
        if lower is not None:
            lower_mask = prop.min(dim='depth') > 0
            iso = iso.where(~lower_mask, other=lower)
        if higher is not None:
            higher_mask = prop.max(dim='depth') < 0
            iso = iso.where(~higher_mask, other=higher)
        return iso

    def set_thermal_state(self, thermal_model=None, age=None):
        #print("...init_thermal_state")
        print(".", end="", flush=True)
        if thermal_model is None:
            thermal_model = TM1
        self.thermal_model = thermal_model
        ##### Agregar edades de la fosa al modelo
        if age is None:
            age = self.data.trench_age[:,1]
        elif isinstance(age, list):
            age = age
        else:
            age = np.repeat(age, len(self.state['lat']))
        self.state['age'] = (
            ('lat'), age, {'name': 'Trench age', 'units':'million years'}
        )
        self.state['Tb'] = self.__set_base_temperature()

    def set_mechanic_state(self, mechanic_model=None):
        if mechanic_model is None:
            mechanic_model = MM(uc=1, lc=14, lm=30, serp=23, serp_pct=0.65)
        self.mechanic_model = mechanic_model

    def __set_base_temperature(self):
        slab_temp = self.thermal_model.calc_slab_temp(
                self.state['Zb'].where(self.state['lab_domain']==0),
                self.state['Z0'].where(self.state['lab_domain']==0),
                self.state['Zb'].sel(lon=self.state['lab_start']),
                self.state['Z0'].sel(lon=self.state['lab_start']),
                self.state['age'])
        lab_temp = self.thermal_model.calc_lab_temp(
                self.state['Zb'].where(self.state['lab_domain']==1),
                self.state['Z0'].where(self.state['lab_domain']==1))
        base_temperature = slab_temp.combine_first(lab_temp)
        base_temperature = base_temperature.assign_attrs(
            {'name': 'Lithosphere base temperature', 'units': 'celsius'}
        )
        return base_temperature

    def get_base_temperature(self):
        return self.state['Tb']

    def get_geotherm(self, input_bounds=None):
        if input_bounds is None:
            Z = self.space['Z']
        else:
            Z = self.get_depths_array(input_bounds)['Z']
        geotherm = self.thermal_model.calc_geotherm(
                Z,
                self.state['Z0'],
                self.state['Zi'],
                self.state['Zm'],
                self.state['Zb'],
                self.state['Tb'])
        geotherm = geotherm.assign_attrs({'name': 'Geotherm', 'units': 'celsius'})
        return geotherm

    def get_heat_flow(self, input_bounds=None):
        if input_bounds is None:
            Z = self.space['Z']
        else:
            Z = self.get_depths_array(input_bounds)['Z']
        heat_flow = self.thermal_model.calc_heat_flow(
                Z,
                self.state['Z0'],
                self.state['Zi'],
                self.state['Zm'],
                self.state['Zb'],
                self.state['Tb'])*1.e3
        heat_flow = heat_flow.assign_attrs({'name': 'Heat flow', 'units': 'milliwatts'})
        return heat_flow

    def get_surface_heat_flow(self):
        heat_flow = self.thermal_model.calc_heat_flow(
                self.state['Z0'],
                self.state['Z0'],
                self.state['Zi'],
                self.state['Zm'],
                self.state['Zb'],
                self.state['Tb'])*1.e3
        heat_flow = heat_flow.assign_attrs(
            {'name': 'Surface heat flow', 'units': 'milliwatts'}
        )
        return heat_flow

    def get_serp_domain(self):
        #TODO: find a better way to get serp_domain
        serp_domain = xr.DataArray(
            np.nan,
            coords=[
                ('lon', self.state['lon'].data),
                ('lat', self.state['lat'].data)
            ]
        )
        if self.thermal_model.name == 'TM1':
            max_heat_flow_moho_lon = self.get_heat_flow(
                {'depth': self.get_moho()}
            ).idxmax('lon')
            min_depth_moho_lon = self.get_moho().idxmin('lon')
            serp_domain1 = xr.where(
                serp_domain['lon'] < max_heat_flow_moho_lon, 1, 0
            )
            serp_domain2 = xr.where(
                serp_domain['lon'] < min_depth_moho_lon, 1, 0
            )
            serp_domain = xr.where(
                serp_domain2 > serp_domain1, serp_domain2, serp_domain1
            )
        else:
            max_heat_flow_icd_lon = self.get_heat_flow(
                {'depth': self.get_icd()}
            ).idxmax('lon')
            min_depth_moho_lon = self.get_moho().idxmin('lon')
            serp_domain1 = xr.where(
                serp_domain['lon'] < max_heat_flow_icd_lon, 1, 0
            )
            serp_domain2 = xr.where(
                serp_domain['lon'] < min_depth_moho_lon, 1, 0
            )
            serp_domain = xr.where(
                serp_domain2 > serp_domain1, serp_domain2, serp_domain1
            )
        return serp_domain

    def get_brittle_yield_strength_envelope(self, input_bounds=None):
        if input_bounds is None:
            Z = self.space['Z']
        else:
            Z = self.get_depths_array(input_bounds)['Z']
        byse = self.mechanic_model.calc_brittle_yield_strength_envelope(
            Z,
            self.state['Z0']
        )
        return byse

    def get_ductile_yield_strength_envelope(self, input_bounds=None):
        if input_bounds is None:
            Z = self.space['Z']
        else:
            Z = self.get_depths_array(input_bounds)['Z']
        serp_domain = self.get_serp_domain()
        geotherm = self.get_geotherm(input_bounds=input_bounds)
        dyse = self.mechanic_model.calc_ductile_yield_strength_envelope(
            Z,
            self.state['Z0'],
            self.state['Zi'],
            self.state['Zm'],
            self.state['Zb'],
            serp_domain,
            geotherm
        )
        return dyse

    def get_yield_strength_envelope(self, input_bounds=None, cond='tension'):
        if input_bounds is None:
            Z = self.space['Z']
        else:
            Z = self.get_depths_array(input_bounds)['Z']
        byse = self.get_brittle_yield_strength_envelope(input_bounds=input_bounds)
        dyse = self.get_ductile_yield_strength_envelope(input_bounds=input_bounds)
        #cond = 'tension'
        yse = self.mechanic_model.calc_yield_strength_envelope(byse, dyse, cond=cond)
        yse = yse.assign_attrs(
            {'name': 'Yield strength envelope', 'units': 'MPa'}
        )
        return yse

    def get_yield_depths(self, input_bounds=None):
        if input_bounds is None:
            Z = self.space['Z']
        else:
            Z = self.get_depths_array(input_bounds)['Z']
        #byse = self.get_brittle_yield_strength_envelope(input_bounds=input_bounds)
        #dyse = self.get_ductile_yield_strength_envelope(input_bounds=input_bounds)
        geotherm = self.get_geotherm(input_bounds=input_bounds)
        yield_depths = self.mechanic_model.calc_yield_depths(
            Z,
            self.state['Z0'],
            geotherm,
            self.isosurface
        )
        return yield_depths

    def get_elastic_tuples(self, input_bounds=None, state='compression'):
        if input_bounds is None:
            Z = self.space['Z']
        else:
            Z = self.get_depths_array(input_bounds)['Z']
        yield_depths = self.get_yield_depths(input_bounds=input_bounds)
        elastic_tuples = self.mechanic_model.calc_elastic_tuples(
            self.state['Z0'],
            self.state['Zi'],
            self.state['Zm'],
            self.state['Zb'],
            yield_depths,
            state
        )
        return elastic_tuples

    def get_eet(self, input_bounds=None):
        if input_bounds is None:
            Z = self.space['Z']
        else:
            Z = self.get_depths_array(input_bounds)['Z']
        elastic_tuples = self.get_elastic_tuples(input_bounds=input_bounds)
        eet = self.mechanic_model.calc_eet(
            elastic_tuples
        )
        eet = eet.assign_attrs(
            {'name': 'Equivalent Elastic Thickness', 'units': 'ET'}
        )
        return eet

    def get_integrated_strength(self, input_bounds=None):
        if input_bounds is None:
            Z = self.space['Z']
        else:
            Z = self.get_depths_array(input_bounds)['Z']
        yse = self.get_yield_strength_envelope(input_bounds=input_bounds)
        integrated_strength = self.mechanic_model.calc_integrated_strength(yse)
        integrated_strength = integrated_strength.assign_attrs(
            {'name': 'Integrated Strength', 'units': 'integrated MPa'}
        )
        return integrated_strength

    #def compare(self, array1, array2):
    #    diff = array1 - array2
    #    diff = diff.assign_attrs(
    #        {'name': 'Difference', 'units': 'Km. diff.'}
    #    )
    #    return diff

    def export_csv(self, darray, filename, name='prop', dropna=False):
        df = darray.to_dataframe(name=name)
        if dropna is True:
            df = df.dropna()
        df.to_csv(filename, sep=' ', na_rep='nan', float_format='%.2f')

    def import_csv(self, filename, name='prop'):
        df = pd.read_csv(
                filename,
                sep='\s+',
                #names=['lon','lat',name],
                index_col=[0,1])
        return df.to_xarray()

    def export_netcdf(self, filename):
        self.state.to_netcdf(filename)

    def stats(self):
        shf = self.get_surface_heat_flow()
        #shf_data = shf_data_setup()
        shf_data = self.shf_data
        shf_model = shf.interp(lon=shf_data['Long'], lat=shf_data['Lat'])
        shf_model = shf_model.to_dataframe(name='SHF_model')
        shf_data = shf_data.to_dataframe()
        shf_data['SHF_model'] = shf_model['SHF_model']
        shf_data = shf_data.dropna()
        estimators, df = evaluate_model(
            shf_data, return_dataframe=True
        )
        return estimators, df

if __name__ == '__main__':
    pass
