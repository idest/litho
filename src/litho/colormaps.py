import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

def categorical_cmap(nc, nsc, cmap="tab10", continuous=False,
    desaturated_first=False):
    if nc > plt.get_cmap(cmap).N:
        raise ValueError("Too many categories for colormap.")
    if isinstance(nsc, list):
        if len(nsc) != nc:
            raise ValueError("Length of shades array should match" +
                " number of categories")
    else:
        nsc = np.repeat(nsc,nc)
    if continuous:
        ccolors = plt.get_cmap(cmap)(np.linspace(0,1,nc))
    else:
        ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
    #cols = np.zeros((nc*sum(nsc), 3))
    cols = np.empty((0,3))
    for i, c in enumerate(ccolors):
        chsv = mcolors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv,nsc[i]).reshape(nsc[i],3)
        if desaturated_first is True:
            arhsv[:,1] = np.linspace(0.25,chsv[1],nsc[i])
            arhsv[:,2] = np.linspace(1,chsv[2],nsc[i])
        else:
            arhsv[:,1] = np.linspace(chsv[1],0.25,nsc[i])
            arhsv[:,2] = np.linspace(chsv[2],1,nsc[i])
        rgb = mcolors.hsv_to_rgb(arhsv)
        #cols[i*nsc[i]:(i+1)*nsc[i],:] = rgb       
        cols = np.vstack((cols,rgb))
    cmap = mcolors.ListedColormap(cols)
    return cmap

def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

def reverse_colourmap(cmap, name = 'my_cmap_r'):
    """
    In: 
    cmap, name 
    Out:
    my_cmap_r

    Explanation:
    t[0] goes from 0 to 1
    row i:   x  y0  y1 -> t[0] t[1] t[2]
                   /
                  /
    row i+1: x  y0  y1 -> t[n] t[1] t[2]

    so the inverse should do the same:
    row i+1: x  y1  y0 -> 1-t[0] t[2] t[1]
                   /
                  /
    row i:   x  y1  y0 -> 1-t[n] t[2] t[1]
    """        
    reverse = []
    k = []   

    for key in cmap._segmentdata:    
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []

        for t in channel:                    
            data.append((1-t[0],t[2],t[1]))            
        reverse.append(sorted(data))    

    LinearL = dict(zip(k,reverse))
    my_cmap_r = mcolors.LinearSegmentedColormap(name, LinearL) 
    return my_cmap_r

def get_diff_cmap(bins):
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # Red -> White -> Blue
    n_bins = bins  # Discretizes the interpolation into bins
    cmap_name = 'diff'
    diff_cmap = mcolors.LinearSegmentedColormap.from_list(
        cmap_name, colors, N=n_bins)
    return diff_cmap

def get_elevation_diff_cmap(bins):
    #colors = [(0, 0.433, 0), (1, 1, 1), (0.314, 0, 0.472)]# Green->White->Purple
    #colors = [(0.157, 0, 0.709), (1, 1, 1), (0, 0.433, 0)]# Blue->White->Green
    colors = [(0.098, 0.276, 0.709), (1, 1, 1), (0, 0.433, 0)]# Blue->White->Green
    n_bins = bins  # Discretizes the interpolation into bins
    cmap_name = 'diff'
    diff_cmap = mcolors.LinearSegmentedColormap.from_list(
        cmap_name, colors, N=n_bins)
    return diff_cmap

c = mcolors.ColorConverter().to_rgb

jet_white = make_colormap(
    [c('darkblue'), c('cyan'), 0.33,
     c('cyan'), c('orange'), 0.66,
     c('orange'), c('red'), 0.85,
     c('red'), c('white')])

jet_white_r = reverse_colourmap(jet_white)

jet_white_2 = make_colormap(
    [c('darkblue'), c('cyan'), 0.33,
     c('cyan'), c('orange'), 0.66,
     c('orange'), c('red'), 0.85,
     c('red'), c('white'), 0.9,
     c('white')])

jet_white_r_2 = reverse_colourmap(jet_white_2)

eet_tassara_07_printed = make_colormap(
    [c('crimson'), c('gold'),0.15,
     c('gold'), c('forestgreen'), 0.25,
     c('forestgreen'), c('xkcd:cerulean'), 0.40,
     c('xkcd:cerulean'), c('darkblue'), 0.60,
     c('darkblue'), c('xkcd:deep pink'), 0.80,
     c('xkcd:deep pink'), c('#600060'), 0.90,
     c('#600060'), c('black')])

eet_tassara_07 = make_colormap(
    [c('#ff0802'), c('#fefe00'),0.15,
     c('#fefe00'), c('#02ff00'), 0.25,
     c('#02ff00'), c('#01fffc'), 0.40,
     c('#01fffc'), c('#0101fe'), 0.60,
     c('#0101fe'), c('#fb01fc'), 0.80,
     c('#fb01fc'), c('#830082'), 0.90,
     c('#830082'), c('black')])

eet_pg_07 = make_colormap(
    [c('#ffffff'), 0.10, c('#f4d6da'), 0.15, c('#eeb3b4'), 0.20,
     c('#ef5e69'), 0.25, c('#ea7a5a'), 0.30, c('#f2d46d'), 0.35,
     c('#deed90'), 0.40, c('#88d5b3'), 0.50, c('#33baf0'), 0.60,
     c('#303e9d'), 0.70, c('#231f20')])

"""
# --- Custom colormap
custom_cmap_name = 'diff_custom'

cdict = {'red':   ((0.0, 1.00, 1.00),
                   (0.1, 1.00, 1.00),
                   (0.2, 1.00, 1.00),
                   (0.3, 1.00, 1.00),
                   (0.4, 1.00, 1.00),
                   (0.6, 1.00, 0.75),
                   (0.7, 0.75, 0.50),
                   (0.8, 0.50, 0.25),
                   (0.9, 0.25, 0.00),
                   (1.0, 0.00, 0.00)),

         'green': ((0.0, 0.00, 0.00),
                   (0.1, 0.00, 0.25),
                   (0.2, 0.25, 0.50),
                   (0.3, 0.50, 0.75),
                   (0.4, 0.75, 1.00),
                   (0.6, 1.00, 0.75),
                   (0.7, 0.75, 0.50),
                   (0.8, 0.50, 0.25),
                   (0.9, 0.25, 0.00),
                   (1.0, 0.00, 0.00)),

         'blue':  ((0.0, 0.00, 0.00),
                   (0.1, 0.00, 0.25),
                   (0.2, 0.25, 0.50),
                   (0.3, 0.50, 0.75),
                   (0.4, 0.75, 1.00),
                   (0.6, 1.00, 1.00),
                   (0.7, 1.00, 1.00),
                   (0.8, 1.00, 1.00),
                   (0.9, 1.00, 1.00),
                   (1.0, 1.00, 1.00)),

        'alpha':  ((0.0, 1.0, 1.0),
                   (0.5, 1.0, 1.0),
                   (1.0, 1.0, 1.0))
         }

diff_cmap_custom = mcolors.LinearSegmentedColormap(custom_cmap_name, cdict)
"""
