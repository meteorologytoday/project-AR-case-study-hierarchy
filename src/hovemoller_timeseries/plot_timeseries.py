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
parser.add_argument('--varnames', type=str, nargs='+', help='Plotted simulation names', default=[])

parser.add_argument('--mitgcm-beg-date', type=str, help='The datetime of iteration zero in mitgcm.', required=True)
parser.add_argument('--mitgcm-deltaT', type=float, help='The timestep (sec) of mitgcm (deltaT).', required=True)
parser.add_argument('--mitgcm-dumpfreq', type=float, help='The timestep (sec) of mitgcm dump frequency.', required=True)
parser.add_argument('--mitgcm-grid-dir', type=str, help='Grid directory of MITgcm.', default="")

parser.add_argument('--archive-root', type=str, help='Root directory that contains all runs.', required=True)
parser.add_argument('--ens-ids', type=str, help='Range of ensemble ids. Example: 1,2,5,7-14', required=True)
parser.add_argument('--output', type=str, help='Output dir', default="")
parser.add_argument('--output-nc', type=str, help='Output dir', default="")
parser.add_argument('--nproc', type=int, help='Number of processors.', default=1)
parser.add_argument('--lat-rng', type=float, nargs=2, help='Latitudes in degree', default=[20, 52])
parser.add_argument('--lon-rng', type=float, nargs=2, help='Longitudes in degree', default=[360-180, 360-144])
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

atm_measured_variables = ['PREC_ACC_C', 'PREC_ACC_NC', 'SNOW_ACC_NC', 'U10', 'V10', 'HFX', 'LH', 'SWDNB', 'LWDNB', 'SST', 'PSFC']
atm_computed_variables = ['WIND10',]
atm_acc_variables = ['PREC_ACC_C', 'PREC_ACC_NC', 'SNOW_ACC_NC',]
atm_all_variables = atm_measured_variables + atm_computed_variables + atm_acc_variables

ocn_measured_variables = ['KPPhbl',]
ocn_computed_variables = []
ocn_all_variables = ocn_measured_variables + ocn_computed_variables

all_variables = atm_all_variables + ocn_all_variables


sim_names = args.sim_names
ens_ids = decomposeRange(args.ens_ids)
archive_root = args.archive_root
#if len(args.sim_names) == 0:
#    args.sim_names = args.input_dirs
#elif len(args.sim_names) != len(args.input_dirs):
#    raise Exception("--sim-names is provided but the number of input does not match the --input-dirs")

print("==================================")
print("Archive root: ", archive_root)
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
       
    result = dict(t_idx=t_idx, t=beg_dt, label=label, label_idx=label_idx, ens_id=ens_id, ens_id_idx=ens_id_idx, status=dict(atm="UNASSIGNED", ocn="UNASSIGNED"),  data=dict(atm={}, ocn={}))


    print("[%s, ens_id=%d, ens_id_idx=%d] Doing date range: [%s, %s]" % (label, ens_id, ens_id_idx, beg_dtstr, end_dtstr))
    try:
        d = result['data']['atm'] 
        print("Load the folder: %s" % (data_dir,))
        _ds = wrf_load_helper.loadWRFDataFromDir(data_dir, prefix="wrfout_d01_", time_rng=[beg_dt, end_dt], extend_time=avg_hrs)
    
        ref_time = _ds.time.to_numpy()
        llon = _ds.coords["XLONG"].isel(time=0) % 360
        llat = _ds.coords["XLAT"].isel(time=0)

        lat = llat[:, 0].to_numpy()
        lon = llon[0, :].to_numpy()

        print("Loaded time: ")
        print(ref_time)
        
        time_vec.append(ref_time[0])

        prec_acc_dt = _ds.attrs['PREC_ACC_DT'] * 60

        WIND10 = ((_ds.U10**2 + _ds.V10**2)**0.5).rename("WIND10") 
    
        need_merge = [WIND10,]
        
        for v in atm_measured_variables + ['LANDMASK', 'OCNMASK', 'MU', 'MUB', 'DNW']:
            if v in _ds:
                need_merge.append(_ds[v])
            else:
                if v not in atm_computed_variables:
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

        #result['data']['atm']['lat'] = lat
        #result['data']['atm']['lon'] = lon

        print("Doing averages...")    
        for varname in atm_measured_variables:
            d[varname] = _ds[varname].weighted(np.cos(llat * np.pi / 180)).mean(dim=['west_east', 'south_north'], skipna=True).to_numpy()
       

        print("Done averages...") 

        for varname in atm_acc_variables:
            d[varname] *= 1e-3 / prec_acc_dt
        
        result['prec_acc_dt'] = prec_acc_dt
        result['status']['atm'] = 'OK'
        
    except Exception as e:
        
        traceback.print_exc()

        # Add nan
        for varname in atm_all_variables:#measured_variables_lnd + measured_variables_ocn + acc_variables:
            d[varname] = np.nan
 
        result['status']['atm'] = 'ERROR'
        

#####################################################
    print("Load ocean data")

    d = result['data']['ocn']
    try:
        if args.mitgcm_grid_dir != "":
            mitgcm_grid_dir = args.mitgcm_grid_dir
        else:
            mitgcm_grid_dir = data_dir
            
        msm = dlh.MITgcmSimMetadata(args.mitgcm_beg_date, args.mitgcm_deltaT, args.mitgcm_dumpfreq, data_dir, mitgcm_grid_dir)
        coo, crop_kwargs = lf.loadCoordinateFromFolderAndWithRange(msm.grid_dir, nlev=None, lat_rng=args.lat_rng, lon_rng=args.lon_rng)

        #coords['ocn']['lat'] = coo.grid["YC"][:, 0]
        #coords['ocn']['lon'] = coo.grid["XC"][0, :]

        # Load average data
        datasets = ["diag_2D",]
        data_ave  = dlh.loadAveragedDataByDateRange(beg_dt, end_dt, msm, **crop_kwargs, datasets=datasets, inclusive="right")  # inclusive is right because output at time=t is the average from "before" to t
        #data_ave = produceDiagQuantities_ocn(data_ave)
        print(list(data_ave.keys()))
        
        for varname in ocn_all_variables:
            d[varname] = np.mean(data_ave[varname])

        result['status']['ocn'] = 'OK'
    except Exception as e:
        traceback.print_exc()
        print("Loading ocean data error. Still keep going...")

        for varname in ocn_all_variables:
            d[varname] = np.nan
 

    result['data']['atm'].update(result['data']['ocn'])
    result['data'] = result['data']['atm']


    return result


if args.output_nc != "" and Path(args.output_nc).is_file():

    print("File ", args.output_nc, " already exists. Load it!")
    ds = xr.open_dataset(args.output_nc)

else:
     
    time_vec = []
    
    data = None

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

            if result['status']['atm'] == "OK" and result['status']['ocn'] == "OK":
                if prec_acc_dt is None:
                    prec_acc_dt = result['prec_acc_dt']
                else:
                    if prec_acc_dt != result['prec_acc_dt']:
                        raise Exception('Some file does not have the same prec_acc_dt. %d and %d' % (prec_acc_dt, result['prec_acc_dt']))

            else:
                
                print("Something wrong with time: %s (%s)" % (result['t'].strftime("%Y-%m-%d_%H"), result['status'], )) 



            if data is None:
                
                data = {
                    varname : np.zeros((len(sim_names), len(ens_ids), len(dts), ))
                    for varname in all_variables
                }

                for k, v in data.items():
                    
                    v.fill(np.nan)
                       
            
            for k, v in result['data'].items():
                data[k][result['label_idx'], result['ens_id_idx'], result['t_idx']] = v


                
  
    ds = xr.Dataset(
        data_vars={
            k : (["run", "ens", "time",], v) for k, v in data.items()
        },
        coords=dict(
            time=dts,
            ens=ens_ids,
            run=sim_names,
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
ds_stat = xr.Dataset(
    
    data_vars={
        k : (
            ["stat", "run", "time", ],
            np.zeros(
                (2, len(ds.coords['run']), len(ds.coords['time']), )
            )) for k in ds.keys()
    },
    
    coords=dict(
        time=ds.coords['time'],
        run=ds.coords['run'],
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

plot_infos = {

    "KPPhbl" : dict(
        label = "MLD",
        unit  = "$ \\mathrm{m} $",
        full = dict(
            levs  = np.linspace(0, 1, 11) * 200,
            ticks = np.linspace(0, 1, 11) * 200,
        ),
        diff = dict(
            levs  = np.linspace(-1, 1, 21) * 5,
            ticks = np.linspace(-1, 1, 6) * 5,
        )
    ),


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

    "HFX" : dict(
        label = "HFX",
        unit  = "$ \\mathrm{W} / \\mathrm{m}^2$",
        full = dict(
            levs  = np.linspace(0, 1, 26) * 100,
            ticks = np.linspace(0, 1, 11) * 100,
        ),
        diff = dict(
            levs  = np.linspace(-1, 1, 21) * 10,
            ticks = np.linspace(-1, 1, 6) * 10,
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

    "PSFC" : dict(
        label = "$P_\\mathrm{sfc}$",
        offset = 0.0,
        factor = 1e-2,
        unit  = "$ \\mathrm{hPa}$",
        full = dict(
            levs  = np.arange(980, 1021, 4),
            ticks = np.arange(980, 1021, 4),
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
    (ds['PREC_ACC_C'] + ds['PREC_ACC_NC']).rename('TTL_PREC'),
])

plot_variables = args.varnames

ncol = len(plot_variables)
nrow = 2

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



for k, plot_type in enumerate(['full', 'diff']):
    for j, varname in enumerate(plot_variables):
        
        _ax = ax[k, j]

        plot_info = plot_infos[varname]
        axis_info = plot_info[plot_type]


        for i, sim_name in enumerate(sim_names):
            
            _ds = ds.isel(run=i)
            _ds_stat = ds_stat.isel(run=i)

            ref_ds = ds.isel(run=0)
                
            factor = 1.0
            offset = 0.0
            if "factor" in plot_info:
                factor = plot_info['factor']
     
            if "offset" in plot_info:
                offset = plot_info['offset']
            
            data = _ds[varname].mean(dim='ens').to_numpy()
            data = (data - offset) * factor

            ref_data = ref_ds[varname].mean(dim='ens').to_numpy()
            ref_data = (ref_data - offset) * factor


            
            if plot_type == "full":
                plot_data = data

            elif plot_type == "diff":
                plot_data = data - ref_data

            _ax.plot(
                _ds.coords['time'],
                plot_data,
                label=sim_name
            )
            
            _ax.set_ylabel(plot_info['unit'])
            _ax.set_title("$\\Delta$%s" % (plot_info['label'], ) )

            _ax.grid(True)
            _ax.xaxis.set_major_formatter(DateFormatter("%Y\n%m/%d"))
            
            _ax.legend()


if args.output != "":
    print("Writing to file: ", args.output)
    fig.savefig(args.output, dpi=600)

if args.no_display is False:
    plt.show()

print("done")

