import traceback
import numpy as np
import scipy
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

M_H2O = 0.018
M_DRY = 0.02896

c_p = 1004
g0  = 9.8
rho_water = 1e3
L_vap = 2.25e6

def intVert(q, p_w):
    dm = ( p_w[:-1, : , :] - p_w[1:, :, : ]) / g0

    print(q.shape)
    print(p_w.shape)

    return np.sum(c_p * dm * q, axis=0)



def saturation_vapor_pressure(T):
    T_degC = T - 273.15
    return 6.112 * np.exp(17.67 * T_degC / (T_degC + 243.5)) * 100.0

def getMixingRatio(T, p, RH):
    p_wv = saturation_vapor_pressure(T) * RH
    return p_wv / (p - p_wv) * M_H2O / M_DRY

vec_getMixingRatio = np.vectorize(getMixingRatio)
def produceDiagQuantities_atm_ens(ds):
    

    SHFLX = ds.HFX.rename("SHFLX")

    new_ds = xr.merge([
        SHFLX,
    ])


    return new_ds

def has_varnames(ds, varnames):
    has_vars = True
    for varname in varnames:
        has_vars = has_vars and (varname in ds)

    return has_vars


def produceDiagQuantities(ds):
    

    merge_vars = []

    #print("Print ds:")
    #print(ds)

    if "MU" in ds:
        MU = ds.MU + ds.MUB
        integration_factor = - MU * ds.DNW / g0  # DNW is negative
        integration_factor = integration_factor.transpose("bottom_top", "south_north", "west_east") 
        integration_range = ds.PB >= 20000.0

        #H_DIAB_TTL = - (MU * ds.H_DIABATIC * ds.DNW / g0 * c_p).sum(dim="bottom_top").rename("H_DIAB_TTL")
 
        IWV = (integration_factor * ds.QVAPOR).where(integration_range).sum(dim="bottom_top").rename("IWV")

        U = ds.U.to_numpy()
        U = (U[:, :, 1:] + U[:, :, :-1]) / 2

        V = ds.V.to_numpy()
        V = (V[:, 1:, :] + V[:, :-1, :]) / 2

        IVT_U = (integration_factor * ds.QVAPOR * U).where(integration_range).sum(dim="bottom_top")
        IVT_V = (integration_factor * ds.QVAPOR * V).where(integration_range).sum(dim="bottom_top")
        IVT = ((IVT_U**2 + IVT_V**2)**0.5).rename("IVT")
        
        merge_vars.extend([IVT, IWV])
   
    #PREC_ACC_DT = ds.attrs['PREC_ACC_DT'] * 60
    #print("PREC_ACC_DT = ", PREC_ACC_DT)

    #H_DIAB_RAIN = ( (ds.PREC_ACC_C + ds.PREC_ACC_NC) / 1e3 / PREC_ACC_DT * rho_water * L_vap ).rename("H_DIAB_RAIN")


    if has_varnames(ds, ["HFX", "LH", "SWUPB", "SWDNB", "LWUPB", "LWDNB"]):
        SHFLX = ds.HFX.rename("SHFLX")
        LHFLX = ds.LH.rename("LHFLX")
        SWFLX = (ds.SWUPB - ds.SWDNB).rename("SWFLX")
        LWFLX = (ds.LWUPB - ds.LWDNB).rename("LWFLX")
        
        TTL_HFLX = SHFLX + LHFLX + SWFLX + LWFLX
        TTL_HFLX = TTL_HFLX.rename("TTL_HFLX")
        
        SLHFLX = (ds.HFX + ds.LH).rename("SLHFLX")
            
        merge_vars.extend([SLHFLX, SHFLX, LHFLX, SWFLX, LWFLX, TTL_HFLX,])
    
    if has_varnames(ds, ["TSK", "T2", "U10", "V10", "PSFC", "Q2"]):

        AOTDIFF = ds.TSK - ds.T2
        AOTDIFF = AOTDIFF.rename("AOTDIFF")

        U10 = ds.U10
        V10 = ds.V10

        WIND10 = (U10**2 + V10**2)**0.5
        WIND10 = WIND10.rename("WIND10")


        SST  = ds.TSK.to_numpy()
        PSFC = ds.PSFC.to_numpy()
        Q2   = ds.Q2.to_numpy()
        qv_sat_tmp = vec_getMixingRatio(SST, PSFC, 1.0)
    
        AOQVDIFF = ds.PSFC.copy()
        AOQVDIFF[:, :] = qv_sat_tmp - Q2
        AOQVDIFF = AOQVDIFF.rename("AOQVDIFF")

        merge_vars.extend([WIND10, AOTDIFF, AOQVDIFF])


    VELOCE = (ds["UOCE"]**2 + ds["VOCE"]**2)**0.5
    VELOCE = VELOCE.rename("VELOCE")
    merge_vars.extend([VELOCE,])


    new_ds = xr.merge(merge_vars)


    return new_ds





def produceDiagQuantities_ocn(data):


    if "UVEL" in data:
        Usfc = data["UVEL"][0:5, :, :].mean(axis=0)
        Vsfc = data["VVEL"][0:5, :, :].mean(axis=0)
        SSC  = np.sqrt(Usfc**2 + Vsfc**2)

        data["SSC"] = SSC
        data["Usfc"] = Usfc
        data["Vsfc"] = Vsfc

    return data





parser = argparse.ArgumentParser(
                    prog = 'plot_skill',
                    description = 'Plot prediction skill of GFS on AR.',
)

parser.add_argument('--date-rng', type=str, nargs=2, help='Date range.', required=True)
parser.add_argument('--skip-hrs', type=int, help='The skip in hours to do the next diag.', required=True)
parser.add_argument('--avg-hrs', type=int, help='The length of time to do the average in hours.', default=np.nan)
#parser.add_argument('--data-freq-hrs', type=int, help='The data frequency in hours.', required=True)
parser.add_argument('--sim-names', type=str, nargs='*', help='Simulation names', default=[])

parser.add_argument('--mitgcm-beg-date', type=str, help='The datetime of iteration zero in mitgcm.', required=True)
parser.add_argument('--mitgcm-deltaT', type=float, help='The timestep (sec) of mitgcm (deltaT).', required=True)
parser.add_argument('--mitgcm-dumpfreq', type=float, help='The timestep (sec) of mitgcm dump frequency.', required=True)
parser.add_argument('--mitgcm-grid-dir', type=str, help='Grid directory of MITgcm.', default="")
parser.add_argument('--pressure-factor', type=float, help='Pressure factor', default=1.0)

parser.add_argument('--lowpass', type=float, nargs=2, help='Low pass filter spatial length (degree, lat-lon).', default=[1.0, 1.0])


parser.add_argument('--input-dirs', type=str, nargs='+', help='Input dirs.', required=True)
parser.add_argument('--output-dir', type=str, help='Output dir', default="output_figure")
parser.add_argument('--nproc', type=int, help='Number of processors.', default=1)
parser.add_argument('--lat-rng', type=float, nargs=2, help='Latitudes in degree', default=[20, 52])
parser.add_argument('--lon-rng', type=float, nargs=2, help='Longitudes in degree', default=[360-180, 360-144])
parser.add_argument('--deg-lat-per-inch', type=float, help='Degree latitude per plot-inch.', default=10.0)
parser.add_argument('--deg-lon-per-inch', type=float, help='Degree longitude per plot-inch', default=10.0)
parser.add_argument('--pvalue-threshold', type=float, help='P value threshold.', default=0.10)
parser.add_argument('--varnames', type=str, nargs='+', help='Plotted variable names', default=['TTL_HFLX',])
parser.add_argument('--overwrite', action="store_true")
parser.add_argument('--no-display', action="store_true")
parser.add_argument('--is-ensemble', action="store_true")
parser.add_argument('--ensemble-members', type=int, help="Ensemble members. Assume equal sampling members.", default=-1)

args = parser.parse_args()
print(args)

if args.is_ensemble and args.ensemble_members == -1:
    raise Exception("The option `--is-ensemble` is set but `--ensemble-members` is not given.")

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

# Plotting setup
heat_flux_setting = dict( 
    ctl = dict(
        contourf_cmap   = "bwr",
        contourf_levs   = np.linspace(-400, 400,  41),
        contourf_ticks  = np.linspace(-400, 400,  9),
    ),

    diff = dict(
        contourf_cmap   = "bwr",
        contourf_levs   = np.linspace(-20, 20, 21),
        contourf_ticks  = np.linspace(-20, 20, 9),
    )
)

levs_ps = np.arange(980, 1040, 4)
levs_ps_diff = np.concatenate(
    (
        np.arange(-20, 0, 1),
        np.arange(1, 21, 1),
    )
) * args.pressure_factor


plot_infos = dict(

    atm = {

        "IVT" : dict(
            factor = 1,
            label  = "IVT",
            unit   = "$\\mathrm{kg} \\, \\mathrm{m} / \\mathrm{s} / \\mathrm{m}^2$",
            ctl = dict(
                contourf_cmap   = "GnBu",
                contourf_levs   = np.linspace(0, 401, 50),
                contourf_ticks  = np.linspace(0, 401, 50),
            ),

            diff = dict(
                contourf_cmap   = "bwr",
                contourf_levs   = np.linspace(-1, 1, 21) * 100,
                contourf_ticks  = np.linspace(-1, 1, 11) * 100,
            )
        ),


        "PSFC" : dict(
            factor = 100,
            label  = "$P_\\mathrm{sfc}$",
            unit   = "$\\mathrm{hPa}$",
            ctl = dict(
                contourf_cmap   = "bone_r",
                contourf_levs   = np.linspace(980, 1040, 16),
                contourf_ticks  = np.linspace(980, 1040, 16),
            ),

            diff = dict(
                contourf_cmap   = "bwr",
                contourf_levs   = np.linspace(-4, 4, 21),
                contourf_ticks  = np.linspace(-4, 4, 11),
            )
        ),


        "WIND10" : dict(
            factor = 1,
            label  = "$\\left|\\vec{U}_\\mathrm{10m}\\right|$",
            unit   = "$\\mathrm{m}/\\mathrm{s}$",
            ctl = dict(
                contourf_cmap   = "hot_r",
                contourf_levs   = np.linspace(0, 20, 21),
                contourf_ticks  = np.linspace(0, 20, 11),
            ),
            diff = dict(
                contourf_cmap   = "bwr",
                contourf_levs   = np.linspace(-5, 5, 21),
                contourf_ticks  = np.linspace(-5, 5, 11),
            )
        ),


        "T2" : dict(
            factor = 1,
            label  = "$T_{\\mathrm{2m}}$",
            unit   = "$\\mathrm{K}$",
            ctl = dict(
                contourf_cmap   = "Spectral_r",
#                contourf_levs   = np.arange(285.15, 293.15, 0.1),
#                contourf_ticks   = np.arange(285.15, 293.15, 0.5),

                contourf_levs   = np.arange(273.15, 300, .5),
                contourf_ticks   = np.arange(273.15, 300, 2),
            ),
            diff = dict(
                contourf_cmap   = "bwr",
                contourf_levs   = np.linspace(-2, 2, 21),
                contourf_ticks  = np.linspace(-2, 2, 11),
            )
        ),

        "SST" : dict(
            factor = 1,
            label  = "$\\mathrm{SST}$",
            unit   = "$\\mathrm{K}$",
            ctl = dict(
                contourf_cmap   = "Spectral_r",
                contourf_levs   = np.arange(273.15, 300, .5),
                contourf_ticks   = np.arange(273.15, 300, 2),
#                contourf_levs   = np.arange(285.15, 293.15, 0.1),
#                contourf_ticks   = np.arange(285.15, 293.15, 0.5),
            ),
            diff = dict(
                contourf_cmap   = "bwr",
                contourf_levs   = np.linspace(-1, 1, 21),
                contourf_ticks  = np.linspace(-1, 1, 11),
            )
        ),

        "QFX" : dict(
            factor = 1e-4,
            label  = "QFX",
            unit   = "$\\mathrm{kg}/\\mathrm{s}/\\mathrm{m}^2$",
            ctl = dict(
                contourf_cmap   = "bwr",
                contourf_levs   = np.linspace(-1, 1, 21),
                contourf_ticks  = np.linspace(-1, 1, 11),
            ),
            diff = dict(
                contourf_cmap   = "bwr",
                contourf_levs   = np.linspace(-1, 1, 21),
                contourf_ticks  = np.linspace(-1, 1, 11),
            )
        ),


        "AOTDIFF" : dict(
            factor = 1,
            label  = "$\\mathrm{SST} - T_{\\mathrm{2m}}$",
            unit   = "$\\mathrm{K}$",
            ctl = dict(
                contourf_cmap   = "bwr",
                contourf_levs   = np.linspace(-5, 5, 21),
                contourf_ticks  = np.linspace(-5, 5, 11),
            ),
            diff = dict(
                contourf_cmap   = "bwr",
                contourf_levs   = np.linspace(-2, 2, 21),
                contourf_ticks  = np.linspace(-2, 2, 11),
            )
        ),

        "H_DIAB_RAIN" : dict(
            factor = 1000,
            label  = "H_DIAB_RAIN",
            unit   = "$\\mathrm{kW}/\\mathrm{m}^2$",
            ctl = dict(
                contourf_cmap   = "bwr",
                contourf_levs   = np.linspace(-1, 1, 21) * 5,
                contourf_ticks  = np.linspace(-1, 1, 11) * 5,
            ),
            diff = dict(
                contourf_cmap   = "bwr",
                contourf_levs   = np.linspace(-1, 1, 21) * 1,
                contourf_ticks  = np.linspace(-1, 1, 11) * 1,
            )
        ),


        "H_DIAB_TTL" : dict(
            factor = 1,
            label  = "H_DIAB_TTL",
            unit   = "$\\mathrm{W}/\\mathrm{m}^2$",
            ctl = dict(
                contourf_cmap   = "bwr",
                contourf_levs   = np.linspace(-1, 1, 21) * 1000,
                contourf_ticks  = np.linspace(-1, 1, 11) * 1000,
            ),
            diff = dict(
                contourf_cmap   = "bwr",
                contourf_levs   = np.linspace(-1, 1, 21) * 1000,
                contourf_ticks  = np.linspace(-1, 1, 11) * 1000,
            )
        ),


        "TTL_HFLX" : dict(
            factor = 1,
            label  = "HFLX",
            unit   = "$\\mathrm{W}/\\mathrm{m}^2$",
            **heat_flux_setting,
        ),

        "SLHFLX" : dict(
            factor = 1,
            label  = "SHFLX + LHFLX",
            unit   = "$\\mathrm{W}/\\mathrm{m}^2$",
            **heat_flux_setting,
        ),

        "SHFLX" : dict(
            factor = 1,
            label  = "SHFLX",
            unit   = "$\\mathrm{W}/\\mathrm{m}^2$",
            **heat_flux_setting,
        ),

        "LHFLX" : dict(
            factor = 1,
            label  = "LHFLX",
            unit   = "$\\mathrm{W}/\\mathrm{m}^2$",
            **heat_flux_setting,
        ),

        "SWFLX" : dict(
            factor = 1,
            label  = "SWFLX",
            unit   = "$\\mathrm{W}/\\mathrm{m}^2$",
            **heat_flux_setting,
        ),

        "LWFLX" : dict(
            factor = 1,
            label  = "LWFLX",
            unit   = "$\\mathrm{W}/\\mathrm{m}^2$",
            **heat_flux_setting,
        ),

        "VELOCE" : dict(
            factor = 1,
            label  = "ocean $\\left|\\vec{U}_\\mathrm{sfc}\\right|$",
            unit   = "$\\mathrm{m}/\\mathrm{s}$",
            ctl = dict(
                contourf_cmap   = "YlOrRd",
                contourf_levs   = np.linspace(0, 1, 21) * 0.5,
                contourf_ticks  = np.linspace(0, 1, 11) * 0.5,
            ),
            diff = dict(
                contourf_cmap   = "bwr",
                contourf_levs   = np.linspace(-.5, .5, 21),
                contourf_ticks  = np.linspace(-.5, .5, 11),
            )
        ),

    },

    ocn = {

        "KPPhbl" : dict(
            factor = 1,
            label  = "MLD",
            unit   = "$\\mathrm{m}$",
            ctl = dict(
                contourf_cmap   = "GnBu",
                contourf_levs   = np.linspace(0, 1, 21) * 100,
                contourf_ticks  = np.linspace(0, 1, 11) * 100,
            ),
            diff = dict(
                contourf_cmap   = "bwr",
                contourf_levs   = np.linspace(-.5, .5, 21) * 50,
                contourf_ticks  = np.linspace(-.5, .5, 11) * 50,
            )
        ),

        "SSC" : dict(
            factor = 1,
            label  = "$\\left|\\vec{U}_\\mathrm{sfc}\\right|$",
            unit   = "$\\mathrm{m}/\\mathrm{s}$",
            ctl = dict(
                contourf_cmap   = "YlOrRd",
                contourf_levs   = np.linspace(0, 1, 21) * 0.5,
                contourf_ticks  = np.linspace(0, 1, 11) * 0.5,
            ),
            diff = dict(
                contourf_cmap   = "bwr",
                contourf_levs   = np.linspace(-.5, .5, 21),
                contourf_ticks  = np.linspace(-.5, .5, 11),
            )
        ),
    }

)

print("Parse variables...")

plot_variables = []
for i, full_varname in enumerate(args.varnames):

    
    category, varname = full_varname.split(".")
   
    has_desc = varname.find("-")
    var_opt = "" 
    if has_desc != -1:
        print(varname)
        varname, var_opt = varname.split("-")
        var_opt = var_opt.upper()
        if not ( var_opt in ["HIGHPASS", "LOWPASS"] ): 
            raise Exception("Error: Option '%s' does not exists for varname '%s'." % (var_opt, varname,))
 
    if not ( (category in plot_infos) and (varname in plot_infos[category])  ):
        
        raise Exception("Error: Varname '%s' has no corresponding plot info." % (varname,))

    plot_variables.append((category, varname, var_opt))



print("==================================")
print("Date range: ", dts[0], " to ", dts[-1])
print("Skip : ", skip_hrs)
print("Avg  : ", avg_hrs)
print("Latitude  box: %.2f %.2f" % (lat_s, lat_n))
print("Longitude box: %.2f %.2f" % (lon_w, lon_e))

for i, (category, varname, var_opt) in enumerate(plot_variables):
    
    print("The %d-th plotted variable: %s - %s (%s)" % (i, category, varname, var_opt if var_opt != "" else "none"))

for i, input_dir in enumerate(args.input_dirs):
    print("The %d-th input folder: %s" % (i, input_dir,))

print("==================================")



ncol = len(args.input_dirs)
nrow = len(args.varnames)

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

            print("Loaded time: ")
            print(ref_time)


        if i > 0:
            if any(ref_time != _ds.time.to_numpy()):
                raise Exception("Time is not consistent between %s and %s" % (args.input_dirs[0], input_dir,))


        _ds = _ds.mean(dim="time", keep_attrs=True)

        if True or not args.is_ensemble:
            _data['atm'] = xr.merge([_ds, produceDiagQuantities(_ds)])
        else:
            _data['atm'] = xr.merge([_ds, produceDiagQuantities_atm_ens(_ds)])
        
        _ds_std = None
        if args.is_ensemble:
            try:
                _ds_std = wrf_load_helper.loadWRFDataFromDir(input_dir, prefix="std_wrfout_d01_", time_rng=[beg_dt, end_dt], extend_time=pd.Timedelta(days=0))
                _ds_std = xr.merge([_ds_std, produceDiagQuantities_atm_ens(_ds_std)])

            except Exception as e:
                print(e)
                traceback.print_exc()
                print("Error happens when trying to get standard deviation file. Ignore this.")

                _ds_std = None

        if _ds_std is not None:
            _ds_std = _ds_std.mean(dim="time", keep_attrs=True)
            _data['atm_std'] = _ds_std

        print("Load ocean data")

        try:
            if args.mitgcm_grid_dir != "":
                mitgcm_grid_dir = args.mitgcm_grid_dir
            else:
                mitgcm_grid_dir = input_dir
                
            msm = dlh.MITgcmSimMetadata(args.mitgcm_beg_date, args.mitgcm_deltaT, args.mitgcm_dumpfreq, input_dir, mitgcm_grid_dir)
            coo, crop_kwargs = lf.loadCoordinateFromFolderAndWithRange(msm.grid_dir, nlev=None, lat_rng=args.lat_rng, lon_rng=args.lon_rng)

            coords['ocn']['lat'] = coo.grid["YC"][:, 0]
            coords['ocn']['lon'] = coo.grid["XC"][0, :]

            #ocn_z_T = coo.grid["RC"].flatten()
            #ocn_z_W = coo.grid["RF"].flatten()
            #ocn_mask = coo.grid["maskInC"]

            # Load average data
            datasets = ["diag_state", "diag_2D"]
            data_ave  = dlh.loadAveragedDataByDateRange(beg_dt, end_dt, msm, **crop_kwargs, datasets=datasets, inclusive="right")  # inclusive is right because output at time=t is the average from "before" to t
            _data['ocn'] = produceDiagQuantities_ocn(data_ave)


            print("KEYS: ", list(_data['ocn'].keys()))
 
            # Load standard deviation
            if args.is_ensemble:
                new_datasets = []
                for j in range(len(datasets)): 
                    new_datasets.append("%s_%s" % ("std",  datasets[j],))

                data_std  = dlh.loadAveragedDataByDateRange(beg_dt, end_dt, msm, **crop_kwargs, datasets=new_datasets, inclusive="right")  # inclusive is right because output at time=t is the average from "before" to t
                _data['ocn_std'] = data_std

        except Exception as e:
            traceback.print_exc()
            print("Loading ocean data error. Still keep going...")

            _data['ocn'] = None

        data.append(_data)
        #data_ave  = dlh.loadAveragedDataByDateRange(dt, dt + avg_hrs, msm, **crop_kwargs, datasets=["diag_Tbdgt", "diag_2D", "diag_state",], inclusive="right")  # inclusive is right because output at time=t is the average from "before" to t
    print("Data loading complete.")
    
    cent_lon = 180.0

    plot_lon_l = lon_w
    plot_lon_r = lon_e
    plot_lat_b = lat_s
    plot_lat_t = lat_n

    proj = ccrs.PlateCarree(central_longitude=cent_lon)
    proj_norm = ccrs.PlateCarree()

    
    thumbnail_height = (lat_n - lat_s) / args.deg_lat_per_inch
    thumbnail_width  = (lon_e - lon_w) / args.deg_lon_per_inch

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
    
    
    fig.suptitle("%s ~ %s" % ( beg_dtstr, end_dtstr, ))
    
        
    print("Plot control simulation: ", args.sim_names[0])
    ref_ax = ax[:, 0]
    ref_data = data[0]


    ps_ref   = ref_data["atm"]["PSFC"]
    for j, _ax in enumerate(ref_ax): 
        
        category, varname, var_opt = plot_variables[j]
        plot_info = plot_infos[category][varname]

        if ref_data[category] is None:
            print("Blank data encountered. Skip this one.")
            fig.delaxes(_ax)
            continue
        elif not ( varname in _data[category] ):
            print("Varname %s does not exist. Skip this one." % (varname, ))
            fig.delaxes(_ax)
            continue



        var_ref = ref_data[category][varname] / plot_info['factor']


        if var_opt == "LOWPASS":

            dlat = coords[category]['lat'][1, 0] - coords[category]['lat'][0, 0] 
            dlon = coords[category]['lon'][0, 1] - coords[category]['lon'][0, 0] 
            filter_size = (
                int(np.ceil(args.lowpass[0] / dlat)), 
                int(np.ceil(args.lowpass[1] / dlon)), 
            )

            var_ref = scipy.ndimage.uniform_filter(
                var_ref.to_numpy(),
                size=filter_size,
                mode='reflect',
            )


        elif var_opt == "HIGHPASS":

            dlat = coords[category]['lat'][1, 0] - coords[category]['lat'][0, 0] 
            dlon = coords[category]['lon'][0, 1] - coords[category]['lon'][0, 0] 
            filter_size = (
                int(np.ceil(args.lowpass[0] / dlat)), 
                int(np.ceil(args.lowpass[1] / dlon)), 
            )

            var_ref_low = scipy.ndimage.uniform_filter(
                var_ref.to_numpy(),
                size=filter_size,
                mode='reflect',
            )

            var_ref = var_ref.to_numpy() - var_ref_low



        mappable = _ax.contourf(
            coords[category]['lon'],
            coords[category]['lat'],
            var_ref,
            levels=plot_info['ctl']['contourf_levs'],
            transform=proj_norm,
            extend="both",
            cmap=plot_info['ctl']['contourf_cmap'],
        )

        cax = tool_fig_config.addAxesNextToAxes(fig, _ax, "right", thickness=0.03, spacing=0.05)
        cb = plt.colorbar(mappable, cax=cax, ticks=plot_info['ctl']['contourf_ticks'], orientation="vertical", pad=0.0)
        cb.ax.set_ylabel(" %s [ %s ]" % (plot_info["label"], plot_info["unit"]))


        # Extra plot for current
        if category == "ocn" and varname == "SSC":

            print("Plot STREAMPLOT for SSC")
                
            _ax.streamplot(
                coords['ocn']['lon'],
                coords['ocn']['lat'],
                ref_data['ocn']['Usfc'],
                ref_data['ocn']['Vsfc'],
                color='dodgerblue',
                linewidth=1,
                transform=proj_norm,
            )

        if category == "atm" and varname == "VELOCE":

            print("Plot STREAMPLOT for VELOCE")
                
            _ax.streamplot(
                coords['atm']['lon'],
                coords['atm']['lat'],
                ref_data['atm']['UOCE'],
                ref_data['atm']['VOCE'],
                color='dodgerblue',
                linewidth=1,
                transform=proj_norm,
            )



        if category == "atm" and varname == "WIND10":

            print("Plot STREAMPLOT for WIND10")
            _ax.streamplot(
                coords['atm']['lon'],
                coords['atm']['lat'],
                ref_data['atm']['U10'],
                ref_data['atm']['V10'],
                color='dodgerblue',
                linewidth=1,
                transform=proj_norm,
            )




        cs = _ax.contour(
            coords['atm']['lon'],
            coords['atm']['lat'],
            ps_ref / 1e2,
            levels=levs_ps,
            transform=proj_norm,
            colors="black",
            linewidths = 1.0,
        )

        #plt.clabel(cs, fmt= ( "%d" if np.all(levs_ps % 1 != 0) else "%.1f" ))


    ref_ax[0].set_title(args.sim_names[0])


    for i in range(1, len(args.input_dirs)):

        print("Plot diff simulation: ", args.sim_names[i])
  
        case_ax = ax[:, i]
        _data = data[i]
 

       
        ps_diff = _data["atm"]["PSFC"] - ps_ref 
        for j, _ax in enumerate(case_ax): 
        
            category, varname, var_opt = plot_variables[j]
            plot_info = plot_infos[category][varname]

            if _data[category] is None:
                print("Blank data encountered. Skip this one.")
                fig.delaxes(_ax)
                continue

            elif not ( varname in _data[category] ):
                print("Varname %s does not exist. Skip this one." % (varname,))
                fig.delaxes(_ax)
                continue


            var_mean1 = ref_data[category][varname]
            var_mean2 = _data[category][varname]

            var_diff = ( var_mean2 - var_mean1 ) / plot_info['factor']

            if var_opt == "LOWPASS":

                dlat = coords[category]['lat'][1, 0] - coords[category]['lat'][0, 0] 
                dlon = coords[category]['lon'][0, 1] - coords[category]['lon'][0, 0] 
                filter_size = (
                    int(np.ceil(args.lowpass[0] / dlat)), 
                    int(np.ceil(args.lowpass[1] / dlon)), 
                )

                var_diff = scipy.ndimage.uniform_filter(
                    var_diff.to_numpy(),
                    size=filter_size,
                    mode='reflect',
                )

            elif var_opt == "HIGHPASS":

                dlat = coords[category]['lat'][1, 0] - coords[category]['lat'][0, 0] 
                dlon = coords[category]['lon'][0, 1] - coords[category]['lon'][0, 0] 
                filter_size = (
                    int(np.ceil(args.lowpass[0] / dlat)), 
                    int(np.ceil(args.lowpass[1] / dlon)), 
                )

                var_diff_low = scipy.ndimage.uniform_filter(
                    var_diff.to_numpy(),
                    size=filter_size,
                    mode='reflect',
                )

                var_diff = var_diff.to_numpy() - var_diff_low



            mappable = _ax.contourf(
                coords[category]['lon'],
                coords[category]['lat'],
                var_diff,
                levels=plot_info['diff']['contourf_levs'],
                transform=proj_norm,
                extend="both",
                cmap=plot_info['diff']['contourf_cmap'],
            )
            cax = tool_fig_config.addAxesNextToAxes(fig, _ax, "right", thickness=0.03, spacing=0.05)
            cb = plt.colorbar(mappable, cax=cax, ticks=plot_info['diff']['contourf_ticks'], orientation="vertical", pad=0.0)
            cb.ax.set_ylabel(" %s%s [ %s ]" % (
                plot_info["label"],
                "(%s)" % var_opt if var_opt != "" else "",
                plot_info["unit"],
            ))

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
            
            #plt.clabel(cs, fmt= ( "%d" if np.all(levs_ps % 1 != 0) else "%.1f" ))
    
            if args.is_ensemble:
                stdcat = "%s_std" % category
                if stdcat in ref_data and varname in ref_data[stdcat]:

                    print("Plotting significancy ... of variable %s " % (varname,))
                    var_std1 = ref_data[stdcat][varname]
                    var_std2 = _data[stdcat][varname]

                    # Doing T-test
                    _tscore, _pvalues = scipy.stats.ttest_ind_from_stats(
                        var_mean1, var_std1, args.ensemble_members,
                        var_mean2, var_std2, args.ensemble_members,
                        equal_var=True,
                        alternative='two-sided',
                    )

                    cs = _ax.contourf(
                        coords[category]['lon'],
                        coords[category]['lat'],
                        _pvalues,
                        cmap=None,
                        colors='none',
                        levels=[-1, args.pvalue_threshold],
                        hatches=[".."],
                        transform=proj_norm, 
                    )

                    # Remove the contour lines for hatches
                    for _, collection in enumerate(cs.collections):
                        collection.set_edgecolor("black")
                        collection.set_linewidth(0.)   
         

 
 
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

        output_filename = "%s/wrf_comparison_avg-%d_%s.png" % (args.output_dir, args.avg_hrs, beg_dtstr)

        if args.overwrite is False and os.path.exists(output_filename):
            print("[%s] File %s already exists. Do not do this job." % (beg_dtstr, output_filename))

        else:
            input_args.append((beg_dt, end_dt, output_filename))

    
    result = pool.starmap(workWrap, input_args)

