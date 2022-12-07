from pathlib import Path
import math
import numpy as np
import pandas as pd

data_dir = Path(__file__).parent / "data"

np.seterr(divide="ignore", invalid="ignore")


def geom_data_setup():
    initial_data = np.loadtxt(data_dir / "Model.dat")
    lab_domain = np.loadtxt(data_dir / "Areas.dat")
    return Data(initial_data, lab_domain)


def age_data_setup():
    trench_age = np.loadtxt(data_dir / "TrenchAge.dat")
    return trench_age


def shf_data_setup():
    shf_data_df = pd.read_csv(
        data_dir / "shf_data_with_types.dat",
        sep="\s+",
        header=0,
        names=["Long", "Lat", "SHF", "Error", "Type"],
        index_col=0,
    )
    shf_data = shf_data_df.to_xarray()
    return shf_data


def rhe_data_setup():
    rhe_data = read_rheo(data_dir / "RheParams.dat")
    return rhe_data


class Data(object):
    @staticmethod
    def round_to_step(x, step=1, prec=0):
        return (step * (np.array(x) / step).round()).round(prec)

    def __init__(self, initial_data, lab_domain):
        self.raw_data = self.__set_raw_data(initial_data)
        self.lonlat_step = 0.2
        self.depth_step = 1.0
        self.depth_precision = 0
        self.lon_axis = self.__set_axis(
            np.nanmax(self.raw_data["longitudes"]),
            np.nanmin(self.raw_data["longitudes"]),
            self.lonlat_step,
            precision=1,
        )
        self.lat_axis = self.__set_axis(
            np.nanmax(self.raw_data["latitudes"]),
            np.nanmin(self.raw_data["latitudes"]),
            self.lonlat_step,
            revert=True,
            precision=1,
        )
        self.depth_axis = self.__set_axis(
            np.nanmax(self.raw_data["topo"]),
            np.nanmin(self.raw_data["slab_lab"]),
            self.depth_step,
            revert=True,
            precision=self.depth_precision,
        )
        self.populated_area = self.__set_populated_area()
        self.sa_plate_area = self.__set_sa_plate_area()
        self.relevant_area = self.__set_relevant_area(
            [self.populated_area, self.sa_plate_area]
        )
        self.slab_lab = self.__set_boundary(self.raw_data["slab_lab"])
        self.moho = self.__set_boundary(self.raw_data["moho"])
        self.icd = self.__set_boundary(self.raw_data["icd"])
        self.topo = self.__set_boundary(self.raw_data["topo"])
        # self.areas_raw = self.reshape_data(self.raw_data['area'])
        self.lab_domain = np.asarray(lab_domain, dtype=bool)

    def __set_raw_data(self, initial_data):
        raw_data = {
            "longitudes": initial_data[:, 0],
            "latitudes": initial_data[:, 1],
            "slab_lab": initial_data[:, 2],
            "moho": initial_data[:, 3],
            "icd": initial_data[:, 4],
            "topo": initial_data[:, 5],
            #'area': initial_data[:, 6]
        }
        return raw_data

    def __set_axis(self, max_v, min_v, step, precision=None, revert=False):
        if precision is not None:
            max_v = self.round_to_step(max_v, step=step, prec=precision)
            min_v = self.round_to_step(min_v, step=step, prec=precision)
        else:
            max_v = np.ceil(max_v)
            min_v = np.floor(min_v)
        if revert is True:
            first_v, last_v = max_v, min_v
        else:
            first_v, last_v = min_v, max_v
        axis = np.linspace(
            first_v,
            last_v,
            num=int(round((abs(last_v - first_v)) / step + 1)),
            endpoint=True,
        )
        if precision is not None:
            axis = self.round_to_step(axis, step=step, prec=precision)
        return axis

    def __set_boundary(self, boundary_data):
        # return self.mask_irrelevant_data(self.reshape_data(self.round_data(boundary_data)))
        return self.mask_irrelevant_data(self.reshape_data(boundary_data))

    def __set_populated_area(self):
        slab_lab_valid = np.invert(
            np.isnan(self.reshape_data(self.raw_data["slab_lab"]))
        )
        moho_valid = np.invert(np.isnan(self.reshape_data(self.raw_data["moho"])))
        icd_valid = np.invert(np.isnan(self.reshape_data(self.raw_data["icd"])))
        topo_valid = np.invert(np.isnan(self.reshape_data(self.raw_data["topo"])))
        populated_area = np.ones(self.get_2D_shape(), dtype=bool)
        populated_area[slab_lab_valid == 0] = 0
        populated_area[moho_valid == 0] = 0
        populated_area[icd_valid == 0] = 0
        populated_area[topo_valid == 0] = 0
        return populated_area

    def __set_sa_plate_area(self):
        """False (0) in Nazca Plate, True (1) in South American Plate"""
        slab_lab = self.reshape_data(self.raw_data["slab_lab"])
        g = np.gradient(slab_lab, axis=0)
        with np.errstate(invalid="ignore"):  # error_ignore
            high_g = np.absolute(g) > 1  # type: np.ndarray
        trench_start = np.argmax(high_g, axis=0)  # gets first true value
        i_idx = self.get_2D_indices()[0]
        sa_plate_area = np.ones(self.get_2D_shape(), dtype=bool)
        sa_plate_area[i_idx < trench_start] = 0
        return sa_plate_area

    def get_2D_shape(self):
        return len(self.lon_axis), len(self.lat_axis)

    def get_3D_shape(self):
        return len(self.lon_axis), len(self.lat_axis), len(self.depth_axis)

    def get_2D_indices(self):
        return np.indices(self.get_2D_shape())

    def get_3D_indices(self):
        return np.indices(self.get_3D_shape())

    def __set_relevant_area(self, relevant_areas):
        relevant_area = np.ones(self.get_2D_shape(), dtype=bool)
        for area in relevant_areas:
            relevant_area[area == 0] = 0
        return relevant_area

    def get_relevant_area(self):
        return self.relevant_area

    def mask_irrelevant_data(self, data):
        mask = np.invert(self.get_relevant_area())
        data[mask] = np.nan
        return data

    def round_data(self, data):
        data = self.round_to_step(data, step=self.depth_step, prec=self.depth_precision)
        return data

    def reshape_data(self, data_column):
        return data_column.T.reshape(len(self.lat_axis), len(self.lon_axis)).T


def read_rheo(filename):
    f = open(filename)
    dic = {}
    line = f.readline()
    while line:
        if line[0] != "#":
            id_rh, name, h, n, a, ref = line.split()
            dic[id_rh] = {
                "name": name,
                "n": float(n),
                "A": float(a),
                "H": float(h),
                "ref": ref,
            }
        line = f.readline()
    f.close()
    return dic
