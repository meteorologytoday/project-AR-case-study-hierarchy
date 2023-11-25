import traceback
import numpy as np
import xarray as xr
import argparse
import pandas as pd
from pathlib import Path
import tool_fig_config
import scipy

from multiprocessing import Pool
import multiprocessing
import os.path
import os
import traceback

from WRFDiag import wrf_load_helper
wrf_load_helper.engine = "netcdf4"


import MITgcmDiff.loadFunctions as lf
import MITgcmDiff.mixed_layer_tools as mlt
import MITgcmDiff.calBudget as cb
import MITgcmDiag.data_loading_helper as dlh
import re


g0 = 9.81

def getApproxCurve(x, y, bin_edges_x, percentiles=[25, 50, 75]):
    
    approx_y = np.zeros((len(percentiles), len(bin_edges_x)-1))

    for i in range(len(bin_edges_x)-1):
        
        edge_l = bin_edges_x[i]
        edge_r = bin_edges_x[i+1]

        sub_y = y[ ( x >= edge_l ) & ( x < edge_r )  ]

        if len(sub_y) == 0:
            approx_y[:, i] = np.nan
        else:
            approx_y[:, i] = np.percentile(sub_y, percentiles)


            #approx_y[1, i] = np.mean(sub_y)

    return approx_y


def findfirst(a):
    return np.argmax(a)

def findlast(a):
    return (len(a) - 1) - np.argmax(a[::-1])


def findArgRange(arr, lb, ub):
    if lb > ub:
        raise Exception("Lower bound should be no larger than upper bound")

    if np.any( (arr[1:] - arr[:-1]) <= 0 ):
        raise Exception("input array should be monotonically increasing")

    idx = np.logical_and((lb <= arr),  (arr <= ub))

    idx_low = findfirst(idx)
    idx_max = findlast(idx)

    return idx_low, idx_max



def findRegion_latlon(lat_arr, lat_rng, lon_arr, lon_rng):

    lat_beg, lat_end = findArgRange(lat_arr, lat_rng[0], lat_rng[1])
    lon_beg, lon_end = findArgRange(lon_arr, lon_rng[0], lon_rng[1])

    return (lon_beg, lon_end, lat_beg, lat_end)


def decomposeRange(s):
   
    s = s.replace(' ','') 
    r = re.findall(r'([0-9]+)(?:-([0-9])+)?,?', s)

    output = []
    for i, (x1, x2) in enumerate(r):
        
        x1 = int(x1)
        if x2 == '':
            output.append(x1)
            
        else:
            x2 = int(x2)

            output.extend(list(range(x1,x2+1))) 

    return output


parser = argparse.ArgumentParser(
                    prog = 'plot_skill',
                    description = 'Plot prediction skill of GFS on AR.',
)

parser.add_argument('--date-rng', type=str, nargs=2, help='Date range.', required=True)
parser.add_argument('--skip-hrs', type=int, help='The skip in hours to do the next diag.', required=True)
parser.add_argument('--avg-hrs', type=int, help='The length of time to do the average in hours.', default=np.nan)
#parser.add_argument('--data-freq-hrs', type=int, help='The data frequency in hours.', required=True)
parser.add_argument('--sim-names', type=str, nargs='+', help='Simulation names', default=[])

parser.add_argument('--highpass-length-lat', type=float, help='The diameter of filter.', required=True)
parser.add_argument('--highpass-length-lon', type=float, help='The diameter of filter.', required=True)

parser.add_argument('--archive-root', type=str, help='Root directory that contains all runs.', required=True)
parser.add_argument('--ens-ids', type=str, help='Range of ensemble ids. Example: 1,2,5,7-14', required=True)
parser.add_argument('--output-dir-prefix', type=str, help='Output dir', required=True)
parser.add_argument('--nproc', type=int, help='Number of processors.', default=1)
parser.add_argument('--lat-rng', type=float, nargs=2, help='Latitudes in degree', default=[20, 52])
parser.add_argument('--lon-rng', type=float, nargs=2, help='Longitudes in degree', default=[360-180, 360-144])
parser.add_argument('--overwrite', action="store_true")
parser.add_argument('--no-display', action="store_true")
args = parser.parse_args()
print(args)

if np.isnan(args.avg_hrs):
    print("--avg-hrs is not set. Set it to --skip-hrs = %d" % (args.skip_hrs,))
    args.avg_hrs = args.skip_hrs

skip_hrs = pd.Timedelta(hours=args.skip_hrs)
avg_hrs  = pd.Timedelta(hours=args.avg_hrs)
dts = pd.date_range(args.date_rng[0], args.date_rng[1], freq=skip_hrs, inclusive="left")

args.lon = np.array(args.lon_rng) % 360.0

lat_n, lat_s = np.amax(args.lat_rng), np.amin(args.lat_rng)
lon_w, lon_e = np.amin(args.lon_rng), np.amax(args.lon_rng)


sim_names = args.sim_names
ens_ids = decomposeRange(args.ens_ids)
archive_root = args.archive_root
#if len(args.sim_names) == 0:
#    args.sim_names = args.input_dirs
#elif len(args.sim_names) != len(args.input_dirs):
#    raise Exception("--sim-names is provided but the number of input does not match the --input-dirs")

print("==================================")
print("Archive root: ", archive_root)
print("Output dir prefix: %s" % (args.output_dir_prefix,))
print("Date range: ", dts[0], " to ", dts[-1])
print("Skip : ", skip_hrs)
print("Avg  : ", avg_hrs)
print("Latitude  box: %.2f %.2f" % (lat_s, lat_n))
print("Longitude box: %.2f %.2f" % (lon_w, lon_e))
print("Highpass length in lat (degree): %.2f" % (args.highpass_length_lat,))
print("Highpass length in lon (degree): %.2f" % (args.highpass_length_lon,))

for i, sim_name in enumerate(sim_names):
    print("The %d-th input sim_name: %s" % (i, sim_name,))

for i, ens_id in enumerate(ens_ids):
    print("The %d-th used ens_id: %d" % (i, ens_id,))


print("==================================")

output_dir_nc = "%s_nc" % args.output_dir_prefix
output_dir_fig = "%s_fig" % args.output_dir_prefix

for _fname in [output_dir_nc, output_dir_fig]:

    if not os.path.exists(_fname):
        print("Making directory: ", _fname)
        os.makedirs(_fname)


print("Load matplotlib...")

import matplotlib as mpl
if args.no_display is False:
    print("Load TkAgg")
    mpl.use('TkAgg')
else:
    print("Load Agg")
    mpl.use('Agg')
    mpl.rc('font', size=15)
 
    
mpl.use('Agg')
# This program plots the AR event
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms
from matplotlib.dates import DateFormatter
import matplotlib.ticker as mticker

plot_infos = {

    "PSFC_high" : dict(
        factor = 1e2,
        bin_edges = np.linspace(-1, 1, 101) * 0.2,
        ticks = np.linspace(-1.0, 1.0, 11) * 0.2,
        label = "$P_{\\mathrm{sfc}}$ highpass",
        unit = "hPa",
    ),



    "SST_high" : dict(
        bin_edges = np.linspace(-1, 1, 101) * 1.0,
        ticks = np.linspace(-0.5, 0.5, 11),
        label = "SST highpass",
        unit = "K",
    ),

    "LH_high" : dict(
        bin_edges = np.linspace(-1, 1, 101) * 10.0,
        ticks = np.linspace(-10, 10, 11),
        label = "Sfc latent heat fluxes highpass",
        unit = "$\\mathrm{W} / \\mathrm{m}^2$",

    ),

    "HFX_high" : dict(
        bin_edges = np.linspace(-1, 1, 101) * 8.0,
        ticks = np.linspace(-5, 5, 11),
        label = "Surface heat fluxes highpass",
        unit = "$\\mathrm{W} / \\mathrm{m}^2$",

    ),

    "AVG_WIND_high" : dict(
        bin_edges = np.linspace(-1, 1, 101) * 1.0,
        ticks = np.linspace(-1, 1, 11) * 0.5,
        label = "Low level wind (sfc to 900hPa) highpass",
        unit = "$\\mathrm{m} / \\mathrm{s}$",
    ),

    "WIND10_high" : dict(
        bin_edges = np.linspace(-1, 1, 101) * 1.0,
        ticks = np.linspace(-1, 1, 11) * 0.5,
        label = "$\\left|\\vec{U}_{10\\mathrm{m}}\\right|$ highpass",
        unit = "$\\mathrm{m} / \\mathrm{s}$",
    ),

    "LH_full" : dict(
        bin_edges = np.linspace(-1, 1, 101) * 50.0,
        ticks = np.linspace(-1, 1, 11) * 50,
        label = "Sfc latent heat fluxes full",
        unit = "$\\mathrm{W} / \\mathrm{m}^2$",

    ),

    "HFX_full" : dict(
        bin_edges = np.linspace(-1, 1, 101) * 8.0,
        ticks = np.linspace(-5, 5, 11),
        label = "Surface heat fluxes full",
        unit = "$\\mathrm{W} / \\mathrm{m}^2$",

    ),

    "AVG_WIND_full" : dict(
        bin_edges = np.linspace(-1, 1, 101) * 1.0,
        ticks = np.linspace(-1, 1, 11) * 0.5,
        label = "Low level wind (sfc to 900hPa) full",
        unit = "$\\mathrm{m} / \\mathrm{s}$",
    ),

    "WIND10_full" : dict(
        bin_edges = np.linspace(-1, 1, 101) * 5.0,
        ticks = np.linspace(-1, 1, 11) * 5.0,
        label = "$\\left|\\vec{U}_{10\\mathrm{m}}\\right|$ full",
        unit = "$\\mathrm{m} / \\mathrm{s}$",
    ),


}

def plot_hist(dt, ds, output_fname, label=None, ens_id=None):
    
    dt_str = dt.strftime("%Y-%m-%d_%H")

    print("Plotting datetime: ", dt_str)

    plot_pairs = [
        ( ("SST", "high"), ("HFX", "high") ),
        ( ("SST", "high"), ("LH", "high") ),
        ( ("SST", "high"), ("AVG_WIND", "high") ),
        ( ("SST", "high"), ("WIND10", "high") ),
#        ( ("SST", "high"), ("PSFC", "high") ),
    ]

    nrow = len(plot_pairs)
    ncol = 1 #len(args.sim_names)

    figsize, gridspec_kw = tool_fig_config.calFigParams(
        w = 3, 
        h = 3, 
        wspace = 1.5,
        hspace = 0.5,
        w_left = 1.0,
        w_right = 1.5,
        h_bottom = 1.0,
        h_top = 1.0,
        nrow = nrow,
        ncol = ncol,
    )



    fig, ax = plt.subplots(nrow, ncol, figsize=figsize, gridspec_kw=gridspec_kw, subplot_kw=dict(aspect="auto"))


    fig.suptitle("[%s, ens=%d] %s to %s" % (
            label,
            ens_id,
            dt.strftime("%Y-%m-%d_%H"),
            (dt + avg_hrs).strftime("%Y-%m-%d_%H"), 
    ))
    for i, ( (partial_varname_X, filter_X), (partial_varname_Y, filter_Y)) in enumerate(plot_pairs):

        _ax = ax[i]
        
        varname_X = "%s_%s" % (partial_varname_X, filter_X)
        varname_Y = "%s_%s" % (partial_varname_Y, filter_Y)

        plot_info_X = plot_infos[varname_X]
        plot_info_Y = plot_infos[varname_Y]

        factor_X = 1.0
        factor_Y = 1.0


        if 'factor' in plot_info_X:
            factor_X = plot_info_X['factor']

        if 'factor' in plot_info_Y:
            factor_Y = plot_info_Y['factor']


        data_X = ds[varname_X].to_numpy().flatten() / factor_X
        data_Y = ds[varname_Y].to_numpy().flatten() / factor_Y


            

        if filter_X == "full":
            data_X -= np.mean(data_X)

        if filter_Y == "full":
            data_Y -= np.mean(data_Y)

        hist, bin_edgesX, bin_edgesY = np.histogram2d(
            data_X,
            data_Y,
            bins=(
                plot_info_X['bin_edges'],
                plot_info_Y['bin_edges'],
            ),
            density=True,
        )

        bin_midX = ( bin_edgesX[1:] + bin_edgesX[:-1] ) / 2
        bin_midY = ( bin_edgesY[1:] + bin_edgesY[:-1] ) / 2

        mappable = _ax.contourf(
            bin_midX, bin_midY,
            hist.transpose(),
            21,
            cmap="bone_r",
        )

        cax = tool_fig_config.addAxesNextToAxes(fig, _ax, "right", thickness=0.03, spacing=0.05)
        plt.colorbar(mappable, cax=cax, orientation="vertical",)

        # Plot approximated curve
        approx_curve = getApproxCurve(data_X, data_Y, bin_edgesX, percentiles=[25, 50, 75])
        _ax.plot(bin_midX, approx_curve[1, :], linestyle=':', color="dodgerblue", zorder=20)

        # plot uncertainty
        _ax.fill_between(bin_midX, approx_curve[0, :], approx_curve[2, :], zorder=2, alpha=0.2, facecolor="#888888", edgecolor="#000000")


    
        
        _ax.set_xlim(plot_info_X['ticks'][[1, -1]])
        _ax.set_ylim(plot_info_Y['ticks'][[1, -1]])

        _ax.set_xticks(plot_info_X['ticks'])
        _ax.set_yticks(plot_info_Y['ticks'])

        _ax.set_xlabel("%s [%s]" % (plot_info_X['label'], plot_info_X['unit']))
        _ax.set_ylabel("%s [%s]" % (plot_info_Y['label'], plot_info_Y['unit']))
        

    
        
    print("Saving figure: ", output_fname)
    fig.savefig(output_fname, dpi=200)



def doWork(t_idx, beg_dt, end_dt, label, ens_id, label_idx, ens_id_idx, data_dir, output_fig, output_nc):
    
    beg_dtstr = beg_dt.strftime("%Y-%m-%d_%H")
    end_dtstr = end_dt.strftime("%Y-%m-%d_%H")
       
    result = dict(t_idx=t_idx, t=beg_dt, label=label, label_idx=label_idx, ens_id=ens_id, ens_id_idx=ens_id_idx, status="UNASSIGNED", data={})
    d = result['data'] 
    print("[%s, ens_id=%d, ens_id_idx=%d] Doing date range: [%s, %s]" % (label, ens_id, ens_id_idx, beg_dtstr, end_dtstr))


    try:


        if os.path.exists(output_nc):
            print("File %s exists! Load it." % (output_nc,))
            _ds = xr.open_dataset(output_nc)
        else:


            print("Load the folder: %s" % (data_dir,))
            _ds = wrf_load_helper.loadWRFDataFromDir(data_dir, prefix="wrfout_d01_", time_rng=[beg_dt, end_dt], extend_time=pd.Timedelta(hours=3))
        
            ref_time = _ds.time.to_numpy()
            llon = _ds.coords["XLONG"].isel(time=0) % 360
            llat = _ds.coords["XLAT"].isel(time=0)

            # detect range


            lon = llon.to_numpy()[0, :]
            lat = llat.to_numpy()[:, 0]
            lon_beg, lon_end, lat_beg, lat_end = findRegion_latlon(lat, [lat_s, lat_n], lon, [lon_w, lon_e])

            print("Loaded time: ")
            print(ref_time)
            
            MU = _ds.MU + _ds.MUB
            integration_factor = - MU * _ds.DNW / g0  # DNW is negative
            integration_factor = integration_factor.transpose("time", "bottom_top", "south_north", "west_east") 
            integration_range = (_ds.PB + _ds.P) >= 900.0 * 1e2 # sfc to 900 hPa
            
            integration_factor_sum = integration_factor.where(integration_range).sum(dim="bottom_top")
            
            U = _ds.U.to_numpy()
            U = (U[:, :, :, 1:] + U[:, :, :, :-1]) / 2

            V = _ds.V.to_numpy()
            V = (V[:, :, 1:, :] + V[:, :, :-1, :]) / 2

            AVG_U    = (integration_factor * U).where(integration_range).sum(dim="bottom_top")
            AVG_V    = (integration_factor * V).where(integration_range).sum(dim="bottom_top")
            AVG_WIND = ((AVG_U**2 + AVG_V**2)**0.5 / integration_factor_sum).rename("AVG_WIND")
            WIND10 = ((_ds.U10**2 + _ds.V10**2)**0.5).rename("WIND10")
           
            need_merge = [AVG_WIND, WIND10]
            
            for v in ["SST", "U10", "V10", "LH", "HFX", "PSFC"]:
                need_merge.append(_ds[v])
            
            _ds = xr.merge(need_merge)
            #print(_ds)
            _ds = _ds.isel(west_east=slice(lon_beg, lon_end+1), south_north=slice(lat_beg, lat_end+1))
            _ds = _ds.mean(dim="time", keep_attrs=True)
       

            # low pass
            dlat = lat[1] - lat[0]
            dlon = lon[1] - lon[0]
            filter_size = (
                int(np.ceil(args.highpass_length_lat / dlat)), 
                int(np.ceil(args.highpass_length_lon / dlon)), 
            )

            print("Filter_size: ", filter_size) 

            need_merge = []
            for v in ["SST", "U10", "V10", "LH", "HFX", "AVG_WIND", "WIND10", "PSFC"]:
                
                d = _ds[v].to_numpy()
                
                low_pass = scipy.ndimage.uniform_filter(
                    _ds[v].to_numpy(),
                    size=filter_size,
                    mode='reflect',
                )
          
                data_full = _ds[v].rename("%s_full" % (v,)) 
                data_high = _ds[v].copy() - low_pass
                data_low  = _ds[v].copy()
                data_low[:, :] = low_pass

                data_high = data_high.rename("%s_high" % (v,))
                data_low  = data_low.rename("%s_low"  % (v,))
                need_merge.append(data_high)
                need_merge.append(data_low)
                need_merge.append(data_full)
           
            _ds = xr.merge(need_merge) 

            if output_nc is not None:
                print("Saving nc file: ", output_nc)
                _ds.to_netcdf(output_nc)

            #result['prec_acc_dt'] = prec_acc_dt
            print("Done data processing of ", beg_dtstr)
            result['status'] = 'OK'
 
        

        plot_hist(beg_dt, _ds, output_fig, label=label, ens_id=ens_id)

           
    except Exception as e:
        
        traceback.print_exc()
        result['status'] = 'ERROR'
        

    return result

with Pool(processes=args.nproc) as pool:

    
    input_args = []
        
    for sim_name_idx, sim_name in enumerate(sim_names):
        for ens_id_idx, ens_id in enumerate(ens_ids):
            for i, beg_dt in enumerate(dts):

                end_dt = beg_dt + avg_hrs

                beg_dtstr = beg_dt.strftime("%Y-%m-%d_%H")
                end_dtstr = end_dt.strftime("%Y-%m-%d_%H")

                label = "%s_ens%02d" % (sim_name, ens_id)
                data_dir = os.path.join(args.archive_root, label)
                output_fig = os.path.join(output_dir_fig, "%s_%s.png" % (label, beg_dtstr, ))
                output_nc  = os.path.join(output_dir_nc, "%s_%s.nc" % (label, beg_dtstr, ))

                if os.path.exists(output_fig) and os.path.exists(output_nc):

                    print("[%s_%s] Both %s and %s exists. Skip." % (label, beg_dtstr, output_fig, output_nc,))
                    continue
                    
                input_args.append((
                    i,
                    beg_dt,
                    end_dt,
                    sim_name,
                    ens_id,
                    sim_name_idx,
                    ens_id_idx,
                    data_dir,
                    output_fig,
                    output_nc,
                ))

    print("Distributing %d Jobs..." % (len(input_args),))
    results = pool.starmap(doWork, input_args)

    for i, result in enumerate(results):

        print("Extracting result %d, that contains t_idx=%d, label=%s, ens_id=%d" % (i, result['t_idx'], result['label'], result['ens_id']))

        if result['status'] == "OK":
            pass
        else: 
            print("Something wrong with time: %s (%s)" % (result['t'].strftime("%Y-%m-%d_%H"), result['status'], )) 
        




