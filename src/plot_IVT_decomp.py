import traceback
import numpy as np
import xarray as xr
import argparse
import pandas as pd
from pathlib import Path
import tool_fig_config

from multiprocessing import Pool
import multiprocessing
import os.path
import os

from WRFDiag import wrf_load_helper
wrf_load_helper.engine = "netcdf4"


import MITgcmDiff.loadFunctions as lf
import MITgcmDiff.mixed_layer_tools as mlt
import MITgcmDiff.calBudget as cb
import MITgcmDiag.data_loading_helper as dlh

import AtmosphericRiver as AR



parser = argparse.ArgumentParser(
                    prog = 'plot_skill',
                    description = 'Plot prediction skill of GFS on AR.',
)

parser.add_argument('--date-rng', type=str, nargs=2, help='Date range.', required=True)
parser.add_argument('--skip-hrs', type=int, help='The skip in hours to do the next diag.', required=True)
parser.add_argument('--avg-hrs', type=int, help='The length of time to do the average in hours.', default=np.nan)
#parser.add_argument('--data-freq-hrs', type=int, help='The data frequency in hours.', required=True)
parser.add_argument('--sim-names', type=str, nargs='*', help='Simulation names', default=[])

parser.add_argument('--pressure-factor', type=float, help='Pressure factor', default=1.0)
parser.add_argument('--input-dirs', type=str, nargs='+', help='Input dirs.', required=True)
parser.add_argument('--output-dir', type=str, help='Output dir', default="output_figure")
parser.add_argument('--nproc', type=int, help='Number of processors.', default=1)
parser.add_argument('--lat-rng', type=float, nargs=2, help='Latitudes in degree', default=[20, 52])
parser.add_argument('--lon-rng', type=float, nargs=2, help='Longitudes in degree', default=[360-180, 360-144])
parser.add_argument('--deg-lat-per-inch', type=float, help='Degree latitude per plot-inch.', default=10.0)
parser.add_argument('--deg-lon-per-inch', type=float, help='Degree longitude per plot-inch', default=10.0)
parser.add_argument('--overwrite', action="store_true")
parser.add_argument('--no-display', action="store_true")
parser.add_argument('--is-ensemble', action="store_true")
args = parser.parse_args()
print(args)


g0 = 9.81

if np.isnan(args.avg_hrs):
    print("--avg-hrs is not set. Set it to --skip-hrs = %d" % (args.skip_hrs,))
    args.avg_hrs = args.skip_hrs

skip_hrs = pd.Timedelta(hours=args.skip_hrs)
avg_hrs  = pd.Timedelta(hours=args.avg_hrs)
dts = pd.date_range(args.date_rng[0], args.date_rng[1], freq=skip_hrs, inclusive="left")

args.lon = np.array(args.lon_rng) % 360.0

lat_n, lat_s = np.amax(args.lat_rng), np.amin(args.lat_rng)
lon_w, lon_e = np.amin(args.lon_rng), np.amax(args.lon_rng)


if len(args.sim_names) == 0:
    args.sim_names = args.input_dirs
elif len(args.sim_names) != len(args.input_dirs):
    raise Exception("--sim-names is provided but the number of input does not match the --input-dirs")

levs_ps = np.arange(980, 1040, 4)
levs_ps_diff = np.concatenate(
    (
        np.arange(-20, 0, 1),
        np.arange(1, 21, 1),
    )
) * args.pressure_factor


print("==================================")
print("Date range: ", dts[0], " to ", dts[-1])
print("Skip : ", skip_hrs)
print("Avg  : ", avg_hrs)
print("Latitude  box: %.2f %.2f" % (lat_s, lat_n))
print("Longitude box: %.2f %.2f" % (lon_w, lon_e))

for i, input_dir in enumerate(args.input_dirs):
    print("The %d-th input folder: %s" % (i, input_dir,))


print("==================================")

IVT_contourf_levs = np.arange(0, 501, 50)
IVT_diff_contourf_levs = np.linspace(0, 1, 21) * 50
IVT_diff_signed_contourf_levs = np.linspace(-1, 1, 41) * 50

IVT_contourf_cmap = "GnBu"
IVT_diff_contourf_cmap = "Reds"
IVT_diff_signed_contourf_cmap = "bwr"

IVT_contourf_ticks = np.arange(0, 501, 100)
IVT_diff_contourf_ticks = np.linspace(0, 1, 11) * 50
IVT_diff_signed_contourf_ticks = np.linspace(-1, 1, 11) * 50


ncol = len(args.input_dirs)
nrow = 6

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
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

print("done")

print("Create dir: %s" % (args.output_dir,))
Path(args.output_dir).mkdir(parents=True, exist_ok=True)



def workWrap(*args):

    try:
        result = plot(*args)
    except:
        traceback.print_exc()
        
        result = "ERROR"
    result


def plot(beg_dt, end_dt, output_filename):
 
    beg_dtstr = beg_dt.strftime("%Y-%m-%d_%H")
    end_dtstr = end_dt.strftime("%Y-%m-%d_%H")
   
    print("Doing date range: [%s, %s]" % (beg_dtstr, end_dtstr))

    data = []

    ds_ref = None
    ref_time = None

    coords = dict(
        atm = dict(
            lat = None,
            lon = None,
        ),

        ocn = dict(
            lat = None,
            lon = None,
        )
    )

    

    for i, input_dir in enumerate(args.input_dirs):

        _data = dict(atm=None, ocn=None)


        print("Load the %d-th folder: %s" % (i, input_dir,))
        _ds = wrf_load_helper.loadWRFDataFromDir(input_dir, prefix="wrfout_d01_", time_rng=[beg_dt, end_dt], extend_time=pd.Timedelta(days=0))
    
        if i == 0:
            ref_time = _ds.time.to_numpy()
            coords['atm']['lon'] = _ds.coords["XLONG"].isel(time=0)
            coords['atm']['lat'] = _ds.coords["XLAT"].isel(time=0)
            
            ds_ref = _ds.mean(dim="time", keep_attrs=True)
            print("Loaded time: ")
            print(ref_time)


        if i > 0:
            if any(ref_time != _ds.time.to_numpy()):
                raise Exception("Time is not consistent between %s and %s" % (args.input_dirs[0], input_dir,))

        _ds = _ds.mean(dim="time", keep_attrs=True)
        _data['atm'] = xr.merge([_ds["PSFC"], AR.calDiffIVT(ds_ref, _ds)])

        
        data.append(_data)

    print("Data loading complete.")
    
    cent_lon = 180.0

    plot_lon_l = lon_w
    plot_lon_r = lon_e
    plot_lat_b = lat_s
    plot_lat_t = lat_n

    proj = ccrs.PlateCarree(central_longitude=cent_lon)
    proj_norm = ccrs.PlateCarree()

    
    thumbnail_height = (lat_n - lat_s) / args.deg_lat_per_inch
    thumbnail_width = (lon_e - lon_w) / args.deg_lon_per_inch

    figsize, gridspec_kw = tool_fig_config.calFigParams(
        w = thumbnail_width,
        h = thumbnail_height,
        wspace = 2.5,
        hspace = 0.5,
        w_left = 1.0,
        w_right = 1.5,
        h_bottom = 1.0,
        h_top = 1.0,
        ncol = ncol,
        nrow = nrow,
    )

    fig, ax = plt.subplots(
        nrow, ncol,
        figsize=figsize,
        subplot_kw=dict(projection=proj, aspect="auto"),
        gridspec_kw=gridspec_kw,
        constrained_layout=False,
        squeeze=False,
    )


    #print(list(ref_data.coords.keys()))

    fig.suptitle("%s ~ %s" % ( beg_dtstr, end_dtstr, ))
    
        
    print("Plot control simulation: ", args.sim_names[0])
    ref_ax = ax[:, 0]
    ref_data = data[0]


    ps_ref   = ref_data["atm"]["PSFC"]

    IVT_X = ref_data["atm"]["IVT_X1"] 
    IVT_Y = ref_data["atm"]["IVT_Y1"] 
    IVT_ref = ( IVT_X**2 + IVT_Y**2 )**0.5

    for j, _ax in enumerate(ref_ax[0:1]): 
       
        mappable = _ax.contourf(
            coords['atm']['lon'],
            coords['atm']['lat'],
            IVT_ref,
            levels=IVT_contourf_levs,
            transform=proj_norm,
            extend="both",
            cmap=IVT_contourf_cmap,
        )

        cax = tool_fig_config.addAxesNextToAxes(fig, _ax, "right", thickness=0.03, spacing=0.05)
        cb = plt.colorbar(mappable, cax=cax, ticks=IVT_contourf_ticks, orientation="vertical", pad=0.0)
        cb.ax.set_ylabel(" IVT [ $\\mathrm{kg} / \\mathrm{s} / \\mathrm{m}^2$ ]")

        cs = _ax.contour(
            coords['atm']['lon'],
            coords['atm']['lat'],
            ps_ref / 1e2,
            levels=levs_ps,
            transform=proj_norm,
            colors="black",
            linewidths = 1.0,
        )

        _ax.streamplot(
            coords['atm']['lon'],
            coords['atm']['lat'],
            IVT_X,
            IVT_Y,
            color='dodgerblue',
            linewidth=1,
            transform=proj_norm,
        )


   

    ref_ax[0].set_title(args.sim_names[0])


    for i in range(1, len(args.input_dirs)):

        print("Plot diff simulation: ", args.sim_names[i])
  
        case_ax = ax[:, i]
        _data = data[i]
 

       
        ps_diff = _data["atm"]["PSFC"] - ps_ref 
        for j, _ax in enumerate(case_ax): 
       
            varname_X, varname_Y = [
                ("IVT_X2", "IVT_Y2", ),
                ("dIVT_X", "dIVT_Y", ),
                ("dIVT_partQ_X", "dIVT_partQ_Y", ),
                ("dIVT_partV_X", "dIVT_partV_Y", ),
                ("dIVT_nonlinear_X", "dIVT_nonlinear_Y", ),
                ("dIVT_res_X", "dIVT_res_Y", ),
            ][j]

            desc = [
                "IVT",
                "$\\Delta$IVT total",
                "$\\Delta$IVT - vapor",
                "$\\Delta$IVT - velocity",
                "$\\Delta$IVT - nonlinear",
                "$\\Delta$IVT - residue",
            ][j]


            data_X = _data["atm"][varname_X]
            data_Y = _data["atm"][varname_Y]
            magnitude = (data_X**2 + data_Y**2)**0.5
            magnitude = magnitude.to_numpy()

            if j == 0:
                    
                mappable = _ax.contourf(
                    coords["atm"]['lon'],
                    coords["atm"]['lat'],
                    magnitude,
                    levels=IVT_contourf_levs,
                    transform=proj_norm,
                    extend="max",
                    cmap=IVT_contourf_cmap,
                )

                cax = tool_fig_config.addAxesNextToAxes(fig, _ax, "right", thickness=0.03, spacing=0.05)
                cb = plt.colorbar(mappable, cax=cax, ticks=IVT_contourf_ticks, orientation="vertical", pad=0.0)
                cb.ax.set_ylabel(" %s [ $\\mathrm{kg} / \\mathrm{s} / \\mathrm{m}^2$ ]" % (desc,))
 

                cs = _ax.contour(
                    coords['atm']['lon'],
                    coords['atm']['lat'],
                    _data["atm"]["PSFC"] / 1e2, 
                    levels=levs_ps,

                    transform=proj_norm,
                    colors="black",
                    linewidths = 1.0,
                )


                _ax.streamplot(
                    coords['atm']['lon'],
                    coords['atm']['lat'],
                    data_X,
                    data_Y,
                    color='dodgerblue',
                    linewidth=1,
                    transform=proj_norm,
                )

            else:            
                
                if varname_X == "dIVT_X":

                    IVT_now = (_data["atm"]["IVT_X2"]**2 + _data["atm"]["IVT_Y2"]**2)**0.5
                    IVT_now = IVT_now.to_numpy()

                    mappable = _ax.contourf(
                        coords["atm"]['lon'],
                        coords["atm"]['lat'],
                        IVT_now - IVT_ref,
                        levels=IVT_diff_signed_contourf_levs,
                        transform=proj_norm,
                        extend="both",
                        cmap=IVT_diff_signed_contourf_cmap,
                    )

                    cax = tool_fig_config.addAxesNextToAxes(fig, _ax, "right", thickness=0.03, spacing=0.05)
                    cb = plt.colorbar(mappable, cax=cax, ticks=IVT_diff_signed_contourf_ticks, orientation="vertical", pad=0.0)

                else:

                    mappable = _ax.contourf(
                        coords["atm"]['lon'],
                        coords["atm"]['lat'],
                        magnitude,
                        levels=IVT_diff_contourf_levs,
                        transform=proj_norm,
                        extend="max",
                        cmap=IVT_diff_contourf_cmap,
                    )

                    cax = tool_fig_config.addAxesNextToAxes(fig, _ax, "right", thickness=0.03, spacing=0.05)
                    cb = plt.colorbar(mappable, cax=cax, ticks=IVT_diff_contourf_ticks, orientation="vertical", pad=0.0)

                   
                cb.ax.set_ylabel(" %s [ $\\mathrm{kg} / \\mathrm{s} / \\mathrm{m}^2$ ]" % (desc,))
     
                if not ( varname_X in ["dIVT_nonlinear_X", "dIVT_res_X", ] ): 
                    _ax.streamplot(
                        coords['atm']['lon'],
                        coords['atm']['lat'],
                        data_X,
                        data_Y,
                        color='dodgerblue',
                        linewidth=1,
                        transform=proj_norm,
                    )


           

                cs = _ax.contour(
                    coords['atm']['lon'],
                    coords['atm']['lat'],
                    ps_diff / 1e2, 
                    levels=levs_ps_diff, 
                    transform=proj_norm, 
                    extend="both", 
                    colors="black",
                    linewidths = 1.0,
                )

        case_ax[0].set_title(args.sim_names[i])



    """
    _ax.quiver(coords["lon"], coords["lat"], _data.u10.to_numpy(), _data.v10.to_numpy(), scale=200, transform=proj_norm)

    cs = _ax.contourf(coords["lon"], coords["lat"], _data['map'], colors='none', levels=[0, 0.5, np.inf], hatches=[None, "."], transform=proj_norm)

    # Remove the contour lines for hatches 
    for _, collection in enumerate(cs.collections):
        collection.set_edgecolor("red")
    """
    
        

 

    for _ax in ax.flatten():

        if _ax is None:

            # This axis has been deleted
            continue

        _ax.set_global()
        #__ax.gridlines()
        _ax.coastlines()#color='gray')
        _ax.set_extent([plot_lon_l, plot_lon_r, plot_lat_b, plot_lat_t], crs=proj_norm)

        gl = _ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.5, linestyle='--')

        gl.xlabels_top   = False
        gl.ylabels_right = False

        #gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 30))
        #gl.xlocator = mticker.FixedLocator([120, 150, 180, -150, -120])#np.arange(-180, 181, 30))
        #gl.ylocator = mticker.FixedLocator([10, 20, 30, 40, 50])
        
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 10, 'color': 'black'}
        gl.ylabel_style = {'size': 10, 'color': 'black'}


    #for _ax in ax[:, 0].flatten():
    #    plt.delaxes(_ax)

    print("Output file: ", output_filename)
    fig.savefig(output_filename, dpi=200)

    if args.no_display is False:
        plt.show()
    
    plt.close(fig)

    return "DONE"    

failed_dates = []
with Pool(processes=args.nproc) as pool:

    input_args = []
    for i, beg_dt in enumerate(dts):
        
        end_dt = beg_dt + avg_hrs
        beg_dtstr = beg_dt.strftime("%Y-%m-%d_%H")
        end_dtstr = end_dt.strftime("%Y-%m-%d_%H")

        output_filename = "%s/IVT_decomp_avg-%d_%s.png" % (args.output_dir, args.avg_hrs, beg_dtstr)

        if args.overwrite is False and os.path.exists(output_filename):
            print("[%s] File %s already exists. Do not do this job." % (beg_dtstr, output_filename))

        else:
            input_args.append((beg_dt, end_dt, output_filename))

    
    result = pool.starmap(workWrap, input_args)

