#!/bin/bash

lib_root="/cw3e/mead/projects/csg102/t2hsu/MITgcm-diagnostics"
run_root="/cw3e/mead/projects/csg102/t2hsu/AR_projects/project01/case02_NEPAC"

mitgcm_deltaT=60.0
mitgcm_dumpfreq=10800.0

export PYTHONPATH="$lib_root/src:$PYTHONPATH"

nproc=20

id=c02b04
simulations=(
    "exp04_ens_20170129" "c02b04_s01-c02b04_s05" "2017-01-29" "2017-01-29" "2017-02-08" 27 33 202 208 "3,7" do
)
    
skip_hrs=3
avg_hrs=3

figure_dir=output_ts_fig_$id
nc_dir=output_ts_nc_$id

mkdir -p $figure_dir
mkdir -p $nc_dir

nparams=11

echo "Run root: $run_root"

for (( i=0 ; i < $(( ${#simulations[@]} / $nparams )) ; i++ )); do

    exp_name="${simulations[$(( i * $nparams + 0 ))]}"
    sim_names="${simulations[$(( i * $nparams + 1 ))]}"
    sim_beg_date="${simulations[$(( i * $nparams + 2 ))]}"
    beg_date="${simulations[$(( i * $nparams + 3 ))]}"
    end_date="${simulations[$(( i * $nparams + 4 ))]}"
    
    lat_beg="${simulations[$(( i * $nparams + 5 ))]}"
    lat_end="${simulations[$(( i * $nparams + 6 ))]}"
    lon_beg="${simulations[$(( i * $nparams + 7 ))]}"
    lon_end="${simulations[$(( i * $nparams + 8 ))]}"
    
    ens_ids="${simulations[$(( i * $nparams + 9 ))]}"
    do_nodo="${simulations[$(( i * $nparams + 10 ))]}"

    if [ "$do_nodo" = "nodo" ] ; then
        echo "Case $i nodo is set. Skip"
        continue
    fi

    archive_root="$run_root/$exp_name/runs"
    output=$figure_dir/${sim_names}_lat-${lat_beg}-${lat_end}_lon-${lon_beg}-${lon_end}.png
    output_nc=$nc_dir/${sim_names}_lat-${lat_beg}-${lat_end}_lon-${lon_beg}-${lon_end}.nc
    output_stat_nc=$nc_dir/${sim_names}_lat-${lat_beg}-${lat_end}_lon-${lon_beg}-${lon_end}.stat.nc

    IFS='-' read -ra sim_names <<< "$sim_names"

    cmd="python3 plot_code/plot_timeseries.py      \
        --archive-root $archive_root   \
        --date-rng $beg_date $end_date \
        --skip-hrs $skip_hrs           \
        --avg-hrs  $avg_hrs            \
        --mitgcm-beg-date $beg_date    \
        --mitgcm-dumpfreq $mitgcm_dumpfreq \
        --mitgcm-deltaT   $mitgcm_deltaT \
        --sim-names ${sim_names[@]}    \
        --ens-ids   $ens_ids           \
        --output  $output              \
        --nproc   $nproc               \
        --output-nc  $output_nc        \
        --lat-rng $lat_beg $lat_end    \
        --lon-rng $lon_beg $lon_end    \
        --naming-rule standard         \
        --varnames KPPhbl  

    "
    eval "$cmd" 

done

wait

echo "Done!"
