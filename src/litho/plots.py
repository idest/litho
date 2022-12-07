import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import cartopy.io.shapereader as shpreader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from litho.utils import MidPointNorm, round_to_1, get_magnitude, makedir_from_filename
from litho.colormaps import (
    jet_white_r, jet_white_r_2, get_diff_cmap, get_elevation_diff_cmap)
from matplotlib.colors import Normalize
from litho.data import geom_data_setup

data = geom_data_setup()

boundaries = xr.Dataset(
    {
        'Z0': (
            ('lon', 'lat'),
            data.topo,
            {'name': 'Topography level', 'units': 'kilometers'}
        ),
        'Zi': (
            ('lon', 'lat'),
            data.icd,
            {'name': 'ICD', 'units': 'kilometers'}
        ),
        'Zm': (
            ('lon', 'lat'),
            data.moho,
            {'name': 'Moho', 'units': 'kilometers'}
        ),
        'Zb': (
            ('lon', 'lat'),
            data.slab_lab,
            {'name': 'Lithosphere base', 'units': 'kilometers'}
        ),
        'lab_domain': (
            ('lat', 'lon'),
            data.lab_domain,
            {'name': 'LAB domain'}
        )
    },
    coords={
        'lon': data.lon_axis,
        'lat': data.lat_axis,
    },
)


def plot(
    darray, ax=None, figsize=None, vmin=None, vmax=None,
    cmap=None, trim=True, shf_data=None, diff=False, estimators=None,
    xlim=None, ylim=None, margin=True, boundaries=boundaries,
    boundaries_color=None, fill=True,
    filename=None,
    **kwargs
):
    #TODO: do not repeat trim logic everywhere
    if cmap is None:
        if 'units' in darray.attrs:
            if darray.units == 'celsius':
                cmap = get_cmap('coolwarm')
            elif darray.units == 'milliwatts':
                cmap = get_cmap('afmhot')
            elif darray.units == 'kilometers':
                cmap = get_cmap('elevation_diff')
            elif darray.units == 'MPa':
                cmap = get_cmap('jet_white_r')
                vmin = 0
                vmax = 200
            elif darray.units == 'integrated MPa':
                cmap = get_cmap('viridis')
            elif darray.units == 'ET':
                cmap = get_cmap('jet_white_r_ET')
                vmin = 0
                vmax = 100
            #elif darray.units == 'Km. diff.':
            #    cmap = get_cmap('elevation_diff')
            else:
                cmap = get_cmap('viridis')
        else:
            cmap = get_cmap('viridis')
    if cmap is 'no':
        cmap = None
    fixed_coords = {}
    for coord_key, coord_val in darray.indexes.items():
        if coord_val.size == 1:
            fixed_coords[coord_key] = coord_val
    # Line plots (1D)
    if darray.squeeze().ndim == 1:
        # Depth plots
        #if 'depth' not in fixed_coords:
        #x_dim = 'depth'
        ##x_dim = [
        ##    k for k in ('lat', 'lon', 'depth') if k not in fixed_coords
        ##][0]
        x_dim = [
            k for k in darray.dims if k not in fixed_coords
        ][0]
        if ax is None:
            if figsize is None:
                figsize = (15, 5)
            if trim is True:
                xlim=(
                    darray.dropna(x_dim)[x_dim].min(),
                    darray.dropna(x_dim)[x_dim].max()
                )
            else:
                xlim=(darray[x_dim].min(), darray[x_dim].max())
            #if ylim is None:
            #    ylim = (0, None)
            if x_dim == 'depth':
                #if 'depth' in darray.coords and 'depth' not in fixed_coords:
                xlim = xlim[::-1]
                if boundaries is not None and len(fixed_coords) == 2:
                    boundaries = [
                        boundaries['Zb'].loc[fixed_coords],
                        boundaries['Zm'].loc[fixed_coords],
                        boundaries['Zi'].loc[fixed_coords],
                        boundaries['Z0'].loc[fixed_coords]
                    ]
                else:
                    boundaries = None
        #print('1D_profile')
        ax = depth_profile(
            darray, x_dim, ax=ax, figsize=figsize,
            xlim=xlim, ylim=ylim, boundaries=boundaries
        )
        return ax
        #
    # Pcolormesh plots
    if darray.squeeze().ndim == 2:
        #Cross section plots (2D)
        if 'lat' in fixed_coords or 'lon' in fixed_coords:
            x_dim = 'lon' if 'lat' in fixed_coords else 'lat'
            y_dim = 'depth'
            if ax is None:
                if figsize is None:
                    figsize = (15, 5)
                if trim is True:
                    lonlat_margin = 0
                    depth_margin = 0
                    if margin is True:
                        lonlat_margin = data.lonlat_step*2
                        depth_margin = data.depth_step*5
                        #lonlat_margin = 0.2*2
                        #depth_margin = 1*5
                    xlim=(
                        darray.dropna(x_dim, how='all')[x_dim].min()-lonlat_margin,
                        darray.dropna(x_dim, how='all')[x_dim].max()+lonlat_margin
                    )
                    ylim=(
                        darray.dropna(y_dim, how='all')[y_dim].min()-depth_margin,
                        darray.dropna(y_dim, how='all')[y_dim].max()+depth_margin
                    )
                else:
                    xlim=(darray[x_dim].min(), darray[x_dim].max())
                    ylim=(darray[y_dim].min(), darray[y_dim].max())
            if boundaries is not None:
                boundaries = [
                    boundaries['Zb'].loc[fixed_coords],
                    boundaries['Zm'].loc[fixed_coords],
                    boundaries['Zi'].loc[fixed_coords],
                    boundaries['Z0'].loc[fixed_coords]
                ]
            #print('cross_section')
            ax = cross_section(
                darray, x_dim, y_dim, ax=ax, figsize=figsize,
                xlim=xlim, ylim=ylim, vmin=vmin, vmax=vmax, cmap=cmap,
                boundaries=boundaries, boundaries_color=boundaries_color,
                fill=fill
            )
            return ax
        # Map plots (2D)
        if all (k not in fixed_coords for k in ('lat', 'lon')):
            if figsize is None:
                figsize = (8, 10)
            ax = base_map(topo=True, figsize=figsize)
            if vmin is not None or vmax is not None:
                cbar_limits = [vmax, vmin]
            else:
                cbar_limits = [np.nanmax(darray), np.nanmin(darray)]
            heatmap_map(
                darray, ax=ax, colormap=cmap, cbar_limits=cbar_limits,
                alpha=1, **kwargs
                )
            if shf_data is not None:
                if diff is False:
                    if darray.units == 'milliwatts':
                        cbar_limits = cbar_limits
                    else:
                        cbar_limits = None
                    data_scatter_map(
                        shf_data['SHF'], shf_data['Long'], shf_data['Lat'],
                        data_types=shf_data['Type'],
                        colormap=cmap, cbar_limits=cbar_limits, ax=ax
                    )
                else:
                    diff_scatter_map(
                        shf_data['SHF_diff'], shf_data['Long'], shf_data['Lat'],
                        data_types=shf_data['Type'], estimators=estimators,
                        ax=ax)
            if filename:
                makedir_from_filename(filename)
                filename = filename + '.png'
                plt.tight_layout()
                plt.savefig(
                    filename, bbox_inches='tight')
                    #dpi=1200, transparent=True)
                    #format="eps", dpi=1200)
                    #format='svg')
                    #dpi=1200, format='pdf')
                    #dpi='figure', format='pdf')
                plt.close()
            return ax

    if darray.squeeze().ndim == 3:
        print("can't plot 3D array")

def get_cmap(colormap):
    if colormap == 'elevation_diff':
        colormap = get_elevation_diff_cmap(100)
    if colormap == 'jet_white_r':
        colormap = jet_white_r_2
    if colormap == 'jet_white_r_ET':
        colormap = jet_white_r
    return colormap

def depth_profile(
    darray, x_dim,
    ax=None, figsize=(15,5),
    xlim=None, ylim=None,
    boundaries=None
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    darray.plot(
        ax=ax, x=x_dim, #xincrease=False,
        xlim=xlim, ylim=ylim
    )
    if boundaries is not None:
        for boundary in boundaries:
            pass
            ax.axvline(boundary)
    return ax

def cross_section(
    darray, x_dim, y_dim,
    ax=None, figsize=(15, 5),
    xlim=None, ylim=None,
    vmin=None, vmax=None,
    cmap='viridis',
    boundaries=None, 
    boundaries_color=None,
    fill=None
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    if fill is True:
        darray.plot(
            ax=ax,
            x=x_dim, y=y_dim,
            xlim=xlim, ylim=ylim,
            vmin=vmin, vmax=vmax, cmap=cmap,
        )
    if boundaries is not None:
        for boundary in boundaries:
            boundary.plot(ax=ax, color=boundaries_color)
    return ax


def base_map(topo=False, figsize=None):
    xloc = np.arange(-80, -60, 4.0)
    yloc = np.arange(-45, -10, 5.0)
    if figsize is None:
        figsize=(8,8)
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    coastline = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    border = cfeature.NaturalEarthFeature(
            'cultural',
            'admin_0_boundary_lines_land',
            '10m'
            )
    ax.add_feature(border, facecolor='None', edgecolor='gray', alpha=0.7)
    ax.add_feature(coastline, facecolor='None', edgecolor='k')
    if topo is True:
        ax.stock_img()
    gl = ax.gridlines(draw_labels=True, linewidth=0.2, color='gray', alpha=0.8)
    #gl.top_label = False
    #gl.right_label = False
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlocator = mticker.FixedLocator(xloc)
    gl.ylocator = mticker.FixedLocator(yloc)
    ax.set_extent([-80, -60, -45, -10])
    return ax

def get_map_scatter_function(map_lon, map_lat, data_types, map):
    markers = ['o']
    data_markers = np.array(markers * len(map_lon))
    m_scatter = {}
    if data_types is not None:
        _, indices = np.unique(data_types, return_inverse=True)
        markers = np.array(['s', 'o', '^', 'p'])
        labels = np.array(['Shallow Marine Probe', 'Deep Marine Drillhole',
                           'Deep Land Borehole', 'Geochemestry of Hotspring'])
        data_markers = markers[indices]
    # Map Scatter Function
    def map_scatter(scatter_data, **kwargs):
        scatter_data = np.ma.masked_invalid(scatter_data)
        for m, l in zip(markers, labels):
            mask = data_markers == m
            m_scatter_data = scatter_data[mask]
            m_lon, m_lat = map_lon[mask], map_lat[mask]
            scatter = map.scatter(
                m_lon, m_lat,
                c=m_scatter_data, marker=m, label=l,
                **kwargs)
            if data_types is not None:
                m_scatter[m] = scatter
        return m_scatter, scatter
    return map_scatter


def heatmap_map(
        array_2D, colormap=None, cbar_limits=None, ax=None, alpha=1,
        filename=None, return_width_ratio=False,
        cbar_label=None, title=None,
        cbar_ticks=None, norm=None):
    cbar_max, cbar_min = np.nanmax(array_2D), np.nanmin(array_2D)
    if cbar_limits is not None:
        cbar_max, cbar_min = cbar_limits[0], cbar_limits[1]
    kwargs = {}
    if colormap is not None:
        kwargs['cmap'] = colormap
    cbar_kwargs = {}
    if cbar_label is not None:
        cbar_kwargs['label'] = cbar_label
    if cbar_ticks is not None:
        cbar_kwargs['ticks'] = cbar_ticks
    array_2D.plot(ax=ax, x='lon', alpha=alpha, vmin=cbar_min, vmax=cbar_max,
        norm=norm, cbar_kwargs=cbar_kwargs, **kwargs)
    if title is not None:
        plt.title(title)

def data_scatter_map(
        data, lon, lat, cbar_limits=None, data_types=None,
        colormap=None,
        ax=None, cbar=True, legend=True,
        rmse=None, return_width_ratio=False,
        filename=None):
    # Scatter map
    cbar_max, cbar_min = np.nanmax(data), np.nanmin(data)
    if cbar_limits is not None:
        cbar_max, cbar_min = cbar_limits[0], cbar_limits[1]
    map_scatter = get_map_scatter_function(lon, lat, data_types, ax)
    m_scatter, scatter = map_scatter(
        data, cmap='afmhot',
        vmin=cbar_min, vmax=cbar_max,
        edgecolors='black', linewidths=0.5)
    if colormap != 'afmhot':
        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes('right', '5%', '12%', axes_class=plt.Axes)
        cbar = plt.colorbar(scatter, cax=cbar_ax)# pad=0.2)
        cbar.set_label('Heat Flow [mW/m?]', rotation=90, labelpad=-45)


def diff_scatter_map(
        diff, lon, lat, data_types=None, ax=None,
        rmse=None, legend=True, return_width_ratio=False,
        filename=None,
        mse=None, sigmas=None, estimators=None):
    if estimators is not None:
        rmse=estimators['rmse']
        mse=estimators['mse']
        sigmas=estimators['sigmas']
    # Scatter map
    diff_max = np.nanmax(diff)
    diff_min = np.nanmin(diff)
    diff_limit = np.nanmax([abs(diff_max), abs(diff_min)])
    diff_limit = round_to_1(diff_limit, 'ceil')
    #diff_step = 10**get_magnitude(diff_limit)
    diff_step = 5
    divisions = np.arange(-80, 80+diff_step, diff_step)
    ticks = np.arange(-80, 80+diff_step, 10)
    #divisions = np.arange(-diff_limit, diff_limit+diff_step, diff_step)
    #ticks = np.arange(-diff_limit, diff_limit+diff_step, 2*diff_step)
    bins = len(divisions) - 1
    diff_cmap = get_diff_cmap(bins)
    norm = MidPointNorm(midpoint=0, vmin=-80, vmax=80)
    map_scatter = get_map_scatter_function(lon, lat, data_types, ax)
    m_scatter, scatter = map_scatter(
        diff, cmap=diff_cmap, norm=norm,
        edgecolors='k', linewidths=.3)
    # Colorbar
    cbar_pad = '15%'
    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes('right', '5%', pad=cbar_pad, axes_class=plt.Axes)
    plt.sca(ax)
    cbar = plt.colorbar(scatter, cax=cbar_ax)# pad=0.2)
    #cbar.set_label('Diff [mW/m?]', rotation=90, labelpad=-60)
    cbar.set_ticks([])
    ## Colorbar histogram
    hist_ax = divider.append_axes('right', '30%', pad='0%', axes_class=plt.Axes)
    plt.sca(ax)
    N, bins, patches = hist_ax.hist(
        diff, bins=divisions,
        orientation='horizontal')
    hist_ax.set_yticks(ticks)
    #hist_ax.set_ylim([-diff_limit, diff_limit])
    hist_ax.set_ylim([-80, 80])
    hist_ax.yaxis.tick_right()
    if sigmas is not None:
        hist_ax.axhline(y=sigmas.n_1_sigma)
        hist_ax.text(15,sigmas.n_1_sigma-0.5*diff_step,r'-$\sigma$',size='small')
        hist_ax.axhline(y=sigmas.p_1_sigma)
        hist_ax.text(15,sigmas.p_1_sigma+0.2*diff_step,r'+$\sigma$',size='small')
        hist_ax.axhline(y=sigmas.n_2_sigma)
        hist_ax.text(15,sigmas.n_2_sigma-0.5*diff_step,r'-2$\sigma$',size='small')
        hist_ax.axhline(y=sigmas.p_2_sigma)
        hist_ax.text(15,sigmas.p_2_sigma+0.2*diff_step,r'+2$\sigma$',size='small')
    norm = Normalize(bins.min(), bins.max())
    for bin, patch in zip(bins, patches):
        color = diff_cmap(norm(bin))
        patch.set_facecolor(color)

def estimator_plot(
        vnames, vaxes, estimator, signed=False, label='', filename=None):
    # x axis
    x_name = vnames[0]
    x_axis = vaxes[0]
    plt.xlabel(''.join(x_name))
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    if len(vnames) > 1:
        # y axis
        y_name = vnames[1]
        y_axis = vaxes[1]
        plt.ylabel(''.join(y_name))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        # 2D matrix
        vmin = np.floor(np.min(estimator))
        vmax = np.ceil(np.max(estimator))
        v = np.linspace(vmin, vmax, 100)
        #plt.contourf(
        #    x_axis, y_axis, rmses.T, v, norm=colors.PowerNorm(gamma=1./2.))
        #plt.pcolormesh(
        #    x_axis, y_axis, rmses.T, norm=colors.PowerNorm(gamma=1./2.))
        x_step = x_axis[1] - x_axis[0]
        y_step = y_axis[1] - y_axis[0]
        if signed is True:
            norm = None
            #vmax = np.nanmax([abs(vmax), abs(vmin)])
            #vmin = -vmax
            vmax = 60
            vmin = -60
            cmap = get_diff_cmap(vmax * 2 + 1)
        else:
            #norm = colors.PowerNorm(gamma=1./2.)
            norm = None
            vmin = 20
            vmax = 60
            cmap = 'viridis'
        plt.imshow(
            estimator.T, origin='lower', aspect='auto',
            norm=norm, cmap=cmap, vmin=vmin, vmax=vmax,
            extent=[
                x_axis[0] - x_step/2, x_axis[-1] + x_step/2,
                y_axis[0] - y_step/2, y_axis[-1] + y_step/2])
        #xx, yy = np.meshgrid(x_axis, y_axis)
        #xx = xx.flatten()
        #yy = yy.flatten()
        #rmses_f = rmses.T.flatten()
        #plt.scatter(xx, yy, c=rmses_f, cmap='rainbow')
        plt.colorbar()
        name = filename + '_2D'
    else:
        index = np.arange(len(x_axis))
        plt.plot(x_axis, estimator, '-r', linewidth=1.)
        plt.plot(x_axis, estimator, 'or', linewidth=1.)
        #plt.bar(x_axis, estimator, alpha=.4, width=(x_axis[1]-x_axis[0])*0.8)
        diff = max(estimator) - min(estimator)
        plt.ylim(min(estimator)-0.2*diff, max(estimator)+0.2*diff)
        plt.ylabel(label)
        plt.grid(True)
        name = filename
    plt.tight_layout()
    if filename:
        makedir_from_filename(filename)
        filename = filename + '.png'
        plt.savefig(filename)#,dpi='figure', format='pdf')
        plt.close()
