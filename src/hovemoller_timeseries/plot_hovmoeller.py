import traceback
import numpy as np
import xarray as xr
import argparse
import pandas as pd
from pathlib import Path
import tool_fig_config
import range_tools

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

parser.add_argument('--archive-root', type=str, help='Root directory that contains all runs.', required=True)
parser.add_argument('--ens-ids', type=str, help='Range of ensemble ids. Example: 1,2,5,7-14', required=True)
parser.add_argument('--output', type=str, help='Output dir', default="")
parser.add_argument('--output-nc', type=str, help='Output dir', default="")
parser.add_argument('--output-stat-nc', type=str, help='Output dir', default="")
parser.add_argument('--nproc', type=int, help='Number of processors.', default=1)
parser.add_argument('--lat-rng', type=float, nargs=2, help='Latitudes in degree', default=[20, 52])
parser.add_argument('--lon-rng', type=float, nargs=2, help='Longitudes in degree', default=[360-180, 360-144])
parser.add_argument('--ttest-threshold', type=float, help='The threshold of T-test', default=0.1)
parser.add_argument('--naming-rule', type=str, help='The naming convention.', choices=["old", "standard"])
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

measured_variables_lnd = ['PREC_ACC_C', 'PREC_ACC_NC', 'SNOW_ACC_NC', 'IWV', 'IVT', 'WIND_AR']
measured_variables_ocn = ['SST', 'HFX', 'LH', 'PREC_ACC_C', 'PREC_ACC_NC', 'SNOW_ACC_NC', 'PSFC', 'IWV', 'IVT', 'WIND_AR']
measured_variables_lndocn = ['PREC_ACC_C', 'PREC_ACC_NC', 'SNOW_ACC_NC', 'IWV', 'IVT', 'WIND_AR']
computed_variables = ['IVT', 'IWV', 'WIND_AR']            

all_variables = (
          ["LND_%s" % varname for varname in measured_variables_lnd]
        + ["OCN_%s" % varname for varname in measured_variables_ocn]
        + ["LNDOCN_%s" % varname for varname in measured_variables_lndocn]
)

acc_variables = ['PREC_ACC_C', 'PREC_ACC_NC', 'SNOW_ACC_NC',]

sim_names = args.sim_names
ens_ids = decomposeRange(args.ens_ids)
archive_root = args.archive_root
#if len(args.sim_names) == 0:
#    args.sim_names = args.input_dirs
#elif len(args.sim_names) != len(args.input_dirs):
#    raise Exception("--sim-names is provided but the number of input does not match the --input-dirs")

print("==================================")
print("Archive root: ", archive_root)
print("T-test threshold: %f" % (args.ttest_threshold,))
print("Output file: %s" % (args.output,))
print("Date range: ", dts[0], " to ", dts[-1])
print("Skip : ", skip_hrs)
print("Avg  : ", avg_hrs)
print("Latitude  box: %.2f %.2f" % (lat_s, lat_n))
print("Longitude box: %.2f %.2f" % (lon_w, lon_e))

for i, sim_name in enumerate(sim_names):
    print("The %d-th input sim_name: %s" % (i, sim_name,))

for i, ens_id in enumerate(ens_ids):
    print("The %d-th used ens_id: %d" % (i, ens_id,))


print("==================================")


#print("Create dir: %s" % (args.output_dir,))
prec_acc_dt = None

def doWork(t_idx, beg_dt, end_dt, label, ens_id, label_idx, ens_id_idx, data_dir):
    
    beg_dtstr = beg_dt.strftime("%Y-%m-%d_%H")
    end_dtstr = end_dt.strftime("%Y-%m-%d_%H")
       
    result = dict(t_idx=t_idx, t=beg_dt, label=label, label_idx=label_idx, ens_id=ens_id, ens_id_idx=ens_id_idx, status="UNASSIGNED", lat=None, data={})
    d = result['data'] 
    print("[%s, ens_id=%d, ens_id_idx=%d] Doing date range: [%s, %s]" % (label, ens_id, ens_id_idx, beg_dtstr, end_dtstr))

    try:

        print("Load the folder: %s" % (data_dir,))
        _ds = wrf_load_helper.loadWRFDataFromDir(data_dir, prefix="wrfout_d01_", time_rng=[beg_dt, end_dt], extend_time=pd.Timedelta(hours=3))
    
        ref_time = _ds.time.to_numpy()
        llon = _ds.coords["XLONG"].isel(time=0) % 360
        llat = _ds.coords["XLAT"].isel(time=0)

        lat = llat[:, 0].to_numpy()
        lon = llon[0, :].to_numpy()

        print("Loaded time: ")
        print(ref_time)
        
        time_vec.append(ref_time[0])

        prec_acc_dt = _ds.attrs['PREC_ACC_DT'] * 60


        # Reference: 
        # CORDEX-WRF v1.3: development of a module for the
        # Weather Research and Forecasting (WRF) model to support
        # the CORDEX community
        # https://gmd.copernicus.org/articles/12/1029/2019/gmd-12-1029-2019.pdf
        
        MU = _ds.MU + _ds.MUB
        integration_factor = - MU * _ds.DNW / g0  # DNW is negative
        integration_factor = integration_factor.transpose("time", "bottom_top", "south_north", "west_east") 
        integration_range = _ds.PB >= 20000.0
        
        IWV = (integration_factor * _ds.QVAPOR).where(integration_range).sum(dim="bottom_top").rename("IWV")

        U = _ds.U.to_numpy()
        U = (U[:, :, :, 1:] + U[:, :, :, :-1]) / 2

        V = _ds.V.to_numpy()
        V = (V[:, :, 1:, :] + V[:, :, :-1, :]) / 2

        IVT_U = (integration_factor * _ds.QVAPOR * U).where(integration_range).sum(dim="bottom_top")
        IVT_V = (integration_factor * _ds.QVAPOR * V).where(integration_range).sum(dim="bottom_top")
        IVT = ((IVT_U**2 + IVT_V**2)**0.5).rename("IVT")
        
        U_AR = (integration_factor * U).where(integration_range).sum(dim="bottom_top") / integration_factor.sum(dim="bottom_top")
        V_AR = (integration_factor * V).where(integration_range).sum(dim="bottom_top") / integration_factor.sum(dim="bottom_top")

        WIND_AR = ((U_AR**2 + V_AR**2)**0.5).rename("WIND_AR") 
    
        #print(integration_factor)
        #print(_ds.QVAPOR)
        #print(IVT)
   
        need_merge = [IVT, IWV, WIND_AR]
        
        for v in measured_variables_lnd + measured_variables_ocn + ['LANDMASK', 'OCNMASK', 'MU', 'MUB', 'DNW']:
            if v in _ds:
                need_merge.append(_ds[v])
            else:
                if v not in computed_variables:
                    raise Exception("Cannot find variable: %s" % (v,))
        

        _ds = xr.merge(need_merge)
        _ds = _ds.mean(dim="time", keep_attrs=True)
       
        lon_beg, lon_end, lat_beg, lat_end = range_tools.findRegion_latlon(lat, [lat_s, lat_n], lon, [lon_w, lon_e])
        
        slice_lon = slice(lon_beg, lon_end+1)
        slice_lat = slice(lat_beg, lat_end+1)
        _ds = _ds.isel(west_east=slice_lon, south_north=slice_lat)

        # Update sliced lat lon
        llon = llon[slice_lat, slice_lon]
        llat = llat[slice_lat, slice_lon]
        
        lat = llat[:, 0].to_numpy()
        lon = llon[0, :].to_numpy()

        result['lat'] = lat
        result['lon'] = lon


#        _ds = _ds.where(
#            ( lat > lat_s ) 
#            & ( lat < lat_n )
#            & ( lon > lon_w )
#            & ( lon < lon_e )
#        )

        print("Doing averages...")    
        for varname in measured_variables_lnd:
        
            d["LND_%s" % varname] = _ds[varname].where(_ds.LANDMASK == 1).weighted(np.cos(llat * np.pi / 180)).mean(dim=['west_east'], skipna=True).to_numpy()
        
        
        for varname in measured_variables_ocn:
            d["OCN_%s" % varname] = _ds[varname].where(_ds.OCNMASK == 1).weighted(np.cos(llat * np.pi / 180)).mean(dim=['west_east'], skipna=True).to_numpy()
        
        for varname in measured_variables_lndocn:
            d["LNDOCN_%s" % varname] = _ds[varname].weighted(np.cos(llat * np.pi / 180)).mean(dim=['west_east'], skipna=True).to_numpy()
        
        
        print("Done averages...") 
        
        for varname in acc_variables:
            d["LND_%s" % varname] *= 1e-3 / prec_acc_dt
            d["OCN_%s" % varname] *= 1e-3 / prec_acc_dt
            d["LNDOCN_%s" % varname] *= 1e-3 / prec_acc_dt
        
        result['prec_acc_dt'] = prec_acc_dt
        result['status'] = 'OK'
        
    except Exception as e:
        
        traceback.print_exc()

        # Add nan
        for varname in all_variables:#measured_variables_lnd + measured_variables_ocn + acc_variables:
            d[varname] = np.nan
 
        result['status'] = 'ERROR'
        

    return result

if args.output_nc != "" and Path(args.output_nc).is_file():

    print("File ", args.output_nc, " already exists. Load it!")
    ds = xr.open_dataset(args.output_nc)

else:
     
    time_vec = []
    
    data = None

    lat = None

    with Pool(processes=args.nproc) as pool:

        
        input_args = []
                    
        if args.naming_rule == "old":
            fmt = "%s_ens%02d"
        elif args.naming_rule == "standard":
            fmt = "%se%02d"
        else:
            raise Exception("Unknown `--naming-rule`: %s" % (args.naming_rule)) 
            
        for sim_name_idx, sim_name in enumerate(sim_names):
            for ens_id_idx, ens_id in enumerate(ens_ids):
                for i, beg_dt in enumerate(dts):

                    end_dt = beg_dt + avg_hrs

                    beg_dtstr = beg_dt.strftime("%Y-%m-%d_%H")
                    end_dtstr = end_dt.strftime("%Y-%m-%d_%H")


                    data_dir = os.path.join(args.archive_root, fmt % (sim_name, ens_id))         

                    input_args.append((
                        i,
                        beg_dt,
                        end_dt,
                        sim_name,
                        ens_id,
                        sim_name_idx,
                        ens_id_idx,
                        data_dir,
                    ))

        print("Distributing %d Jobs..." % (len(input_args),))
        results = pool.starmap(doWork, input_args)

        for i, result in enumerate(results):

            print("Extracting result %d, that contains t_idx=%d, label=%s, ens_id=%d" % (i, result['t_idx'], result['label'], result['ens_id']))

            if result['status'] == "OK":
                if prec_acc_dt is None:
                    prec_acc_dt = result['prec_acc_dt']
                else:
                    if prec_acc_dt != result['prec_acc_dt']:
                        raise Exception('Some file does not have the same prec_acc_dt. %d and %d' % (prec_acc_dt, result['prec_acc_dt']))

                if lat is None:
                    lat = result['lat']

                if data is None:
                    data = {
                        varname : np.zeros((len(sim_names), len(ens_ids), len(dts), len(lat), ))
                        for varname in all_variables
                    }

                    for k, v in data.items():
                        
                        v.fill(np.nan)
                           
                
                for k, v in result['data'].items():
                    data[k][result['label_idx'], result['ens_id_idx'], result['t_idx'], :] = v

            else:
                
                print("Something wrong with time: %s (%s)" % (result['t'].strftime("%Y-%m-%d_%H"), result['status'], )) 


                
  
    ds = xr.Dataset(
        data_vars={
            k : (["run", "ens", "time", "lat"], v) for k, v in data.items()
        },
        coords=dict(
            time=dts,
            ens=ens_ids,
            run=sim_names,
            lat=lat,
            reference_time=pd.Timestamp('2001-01-01'),
        ),
        attrs=dict(
            PREC_ACC_DT = prec_acc_dt,
            lat_s = lat_s,
            lat_n = lat_n,
            lon_w = lon_w,
            lon_e = lon_e,
        )
    )

    if args.output_nc != "":
        print("Output to file: ", args.output_nc)
        ds.to_netcdf(args.output_nc)



# Doing stats
if args.output_stat_nc != "" and Path(args.output_stat_nc).is_file():
    
    print("File ", args.output_stat_nc, " already exists. Load it!")
    ds_stat = xr.open_dataset(args.output_stat_nc)
    
else:
    
    ds_stat = xr.Dataset(
        
        data_vars={
            k : (
                ["stat", "run", "time", "lat"],
                np.zeros(
                    (2, len(ds.coords['run']), len(ds.coords['time']), len(ds.coords['lat']))
                )) for k in ds.keys()
        },
        
        coords=dict(
            time=ds.coords['time'],
            run=ds.coords['run'],
            lat=ds.coords['lat'],
            stat=["mean", "std", ],
            reference_time=pd.Timestamp('2001-01-01'),
        ),
        
        attrs=dict(
            ens_members = len(ens_ids),
            lat_s = lat_s,
            lat_n = lat_n,
            lon_w = lon_w,
            lon_e = lon_e,
        )

    )

    for varname in ds_stat.keys():
        ds_stat[varname][0, :, :, :] = ds[varname].mean(dim='ens')
        ds_stat[varname][1, :, :, :] = ds[varname].std(dim='ens', ddof=1)

    print('Output stat file: ', args.output_stat_nc)
    ds_stat.to_netcdf(args.output_stat_nc)




plot_infos = {

    "IWV" : dict(
        label = "IWV",
        unit  = "$ \\mathrm{kg} / \\mathrm{m}^2$",
        full = dict(
            levs  = np.linspace(0, 1, 11) * 30,
            ticks = np.linspace(0, 1, 11) * 30,
        ),
        diff = dict(
            levs  = np.linspace(-1, 1, 21) * 5,
            ticks = np.linspace(-1, 1, 6) * 5,
        )
    ),

    "IVT" : dict(
        label = "IVT",
        unit  = "$ \\mathrm{kg} \\, \\mathrm{m} / \\mathrm{s} / \\mathrm{m}^2$",
        full = dict(
            levs  = np.linspace(0, 1, 26) * 500,
            ticks = np.linspace(0, 1, 11) * 500,
        ),
        diff = dict(
            levs  = np.linspace(-1, 1, 21) * 50,
            ticks = np.linspace(-1, 1, 6) * 50,
        )

    ),

    "TTL_PREC" : dict(
        label = "Rainfall",
        factor = 3600 * 1e3, 
        unit  = "$ \\mathrm{mm} / \\mathrm{hr}$",
        full = dict(
            levs  = np.linspace(0, 1, 11) * 3,
            ticks = np.linspace(0, 1, 11) * 3,
        ),
        diff = dict(
            levs  = np.linspace(-1, 1, 21) * 0.5,
            ticks = np.linspace(-1, 1, 6) * 0.5,
        )

    ),

    "WIND_AR" : dict(
        label = "Low level wind",
        unit  = "$ \\mathrm{m} / \\mathrm{s}$",
        full = dict(
            levs  = np.linspace(0, 1, 11) * 30,
            ticks = np.linspace(0, 1, 11) * 30,
        ),
        diff = dict(
            levs  = np.linspace(-1, 1, 21) * 5,
            ticks = np.linspace(-1, 1, 6) * 5,
        )

    ),

    "SST" : dict(
        label = "SST",
        offset = 273.15,
        unit  = "$ \\mathrm{K}$",
        full = dict(
            levs  = np.linspace(0, 30, 16),
            ticks = np.linspace(0, 30, 7),
        ),
        diff = dict(
            levs  = np.linspace(-1, 1, 21) * 1,
            ticks = np.linspace(-1, 1, 6) * 1,
        )
    ),

}



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
import scipy

    
ds = xr.merge([
    ds,
    (ds['LND_PREC_ACC_C'] + ds['LND_PREC_ACC_NC']).rename('LNDOCN_TTL_PREC'),
])

plot_varibales = [
    ("TTL_PREC", "LNDOCN"),
    ("IVT",      "OCN"),
    ("IWV",      "OCN"),
#    ("WIND_AR",  "OCN"),
    ("SST",      "OCN"),
]

#ncol = len(sim_names)
#nrow = len(plot_varibales)

nrow = len(sim_names)
ncol = len(plot_varibales)


figsize, gridspec_kw = tool_fig_config.calFigParams(
    w = 6,
    h = 3,
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
    subplot_kw=dict(aspect="auto"),
    gridspec_kw=gridspec_kw,
    constrained_layout=False,
    squeeze=False,
)



fig.suptitle("Lat: %d - %d, Lon: %d - %d" % (lat_s, lat_n, lon_w, lon_e) )


ref_sim = 0
ref_ds = ds.isel(run=ref_sim)
ref_ds_stat = ds_stat.isel(run=ref_sim)
for i, sim_name in enumerate(sim_names):
    
    _ds = ds.isel(run=i)
    _ds_stat = ds_stat.isel(run=i)
        
    if i == ref_sim:
        plot_type = 'full'
    else:
        plot_type = 'diff'


    for j, (varname, region) in enumerate(plot_varibales):
        
        plot_info = plot_infos[varname]
        axis_info = plot_info[plot_type]
        
        full_varname = "%s_%s" % (region, varname, )

        factor = 1.0
        offset = 0.0
        if "factor" in plot_info:
            factor = plot_info['factor']
 
        if "offset" in plot_info:
            offset = plot_info['offset']
        
        data     = _ds[full_varname].mean(dim='ens').to_numpy()
        ref_data = ref_ds[full_varname].mean(dim='ens').to_numpy()

        if plot_type == 'full':
            data = (data - offset) * factor
            cmap = 'bone_r'
        else:
            data = (data - ref_data) * factor
            cmap = 'bwr'

        if "cmap" in axis_info:
            cmap = axis_info['cmap']

        #_ax = ax[j, i] 
        _ax = ax[i, j]
        mappable = _ax.contourf(
            _ds.coords['time'],
            _ds.coords['lat'],
            data.transpose(),
            levels = axis_info['levs'],
            cmap=cmap,
            extend="both",
        )
       
        cax = tool_fig_config.addAxesNextToAxes(fig, _ax, "right", thickness=0.03, spacing=0.05,)
        cb = plt.colorbar(mappable, cax=cax, orientation="vertical")
        cb.set_ticks(axis_info['ticks']) 
    
        _ax.set_ylabel(plot_info['unit'])
        _ax.set_title("(%s) %s" % (region, plot_info['label'], ) )

        _ax.grid()
        _ax.xaxis.set_major_formatter(DateFormatter("%Y\n%m/%d"))

        if plot_type == 'diff':
            ttest = scipy.stats.ttest_ind(
                _ds[full_varname].to_numpy(),
                ref_ds[full_varname].to_numpy(),
                axis=0,
                equal_var=True,
                #nan_policy='raise',
                nan_policy='omit',
                alternative='two-sided',
            )
            print("Shape of pvalue: ", ttest.pvalue.shape)

            #scipy.stats.ttest_ind_from_stats(
            #    mean1, std1, nobs1,
            #    mean2, std2, nobs2,
            #    equal_var=True,
            #    alternative='two-sided')
            #)
            
            # T-test
            cs = _ax.contourf(
                _ds.coords["time"],
                _ds.coords["lat"],
                ttest.pvalue.transpose(),
                cmap=None,
                colors='none',
                levels=[-1, args.ttest_threshold],
                hatches=["//"],
            )

            # Remove the contour lines for hatches
            for _, collection in enumerate(cs.collections):
                collection.set_edgecolor("gray")
                #collection.set_linewidth(0.)   
     




if args.output != "":
    print("Writing to file: ", args.output)
    fig.savefig(args.output, dpi=600)

if args.no_display is False:
    plt.show()

print("done")

