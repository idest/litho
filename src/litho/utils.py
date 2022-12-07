#Utils

class DotDict(dict):
    # dot.notation access to dictionary attributes"
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

from numpy import ma
from matplotlib import cbook
from matplotlib.colors import Normalize

class MidPointNorm(Normalize):
    def __init__(self, midpoint=0, vmin=None, vmax=None, clip=False):
        Normalize.__init__(self,vmin, vmax, clip)
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        self.autoscale_None(result)
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if not (vmin < midpoint < vmax):
            raise ValueError("midpoint must be between maxvalue and minvalue.")
        elif vmin == vmax:
            result.fill(0) # Or should it be all masked? Or 0.5?
        elif vmin > vmax:
            raise ValueError("maxvalue must be bigger than minvalue")
        else:
            vmin = float(vmin)
            vmax = float(vmax)
            if clip:
                mask = ma.getmask(result)
                result = ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                  mask=mask)

            # ma division is very slow; we can take a shortcut
            resdat = result.data

            #First scale to -1 to 1 range, than to from 0 to 1.
            resdat -= midpoint
            resdat[resdat>0] /= abs(vmax - midpoint)
            resdat[resdat<0] /= abs(vmin - midpoint)

            resdat /= 2.
            resdat += 0.5
            result = ma.array(resdat, mask=result.mask, copy=False)

        if is_scalar:
            result = result[0]
        return result

    #def inverse(self, value):
    #    if not self.scaled():
    #        raise ValueError("Not invertible until scaled")
    #    vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

    #    if cbook.iterable(value):
    #        val = ma.asarray(value)
    #        val = 2 * (val-0.5)
    #        val[val>0]  *= abs(vmax - midpoint)
    #        val[val<0] *= abs(vmin - midpoint)
    #        val += midpoint
    #        return val
    #    else:
    #        val = 2 * (val - 0.5)
    #        if val < 0:
    #            return  val*abs(vmin-midpoint) + midpoint
    #        else:
    #            return  val*abs(vmax-midpoint) + midpoint

from math import log10, floor, ceil

def get_magnitude(x):
    magnitude = int(floor(log10(abs(x)))) 
    return magnitude

def round_to_1(x, direction = round):
    # Rounds a number to one significative figure.
    # If used with direction = 'ceil' or direction = 'floor' rounds up or down
    if direction == 'ceil':
        direction = ceil
    elif direction == 'floor':
        direction = floor
    places = -get_magnitude(x)
    return direction(x * (10**places)) / float(10**places)

def round_to_step(x, step=1, prec=0):
    return (step * (np.array(x) / step).round()).round(prec)

import os

def makedir(dire):
    #TODO: use pathlib
    if not os.path.exists(dire):
        try:
            os.makedirs(dire)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    #TODO: return pathlib.Path
    return str(dire)

def makedir_from_filename(filename):
    #TODO: use pathlib
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    #TODO: return pathlib.Path
    return str(filename)


import numpy as np

def calc_deviation(array1, array2):
    sd = np.nansum(abs(array1 - array2))/array1.size
    sd = float(sd)
    return sd

import pandas as pd

def print_data_table(shf_df=None, filename=None):
    if shf_df is None:
        shf_df = pd.DataFrame.from_csv(
            'datos_Q/QsObs.txt', sep=' ', skiprows=1, header=None,
            names=['Longitud', 'Latitud', 'Flujo de Calor', 'Error',
                   'Referencia', 'Tipo'])
    if filename == None:
        filename = 'datos_Q/tabla.xlsx'
    writer = pd.ExcelWriter(filename + '.xlsx')
    df.to_excel(writer, 'Datos')
    writer.save()


def export_csv(darray, filename, name='prop', dropna=False, sep=' '):
        df = darray.to_dataframe(name=name)
        if dropna is True:
            df = df.dropna()
        df.to_csv(filename, sep=sep, na_rep='nan', float_format='%.2f')
