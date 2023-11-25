import numpy as np
import xarray as xr

g0 = 9.81

def calDiffIVT(ds1, ds2):
    
    # Reference: 
    # CORDEX-WRF v1.3: development of a module for the
    # Weather Research and Forecasting (WRF) model to support
    # the CORDEX community
    # https://gmd.copernicus.org/articles/12/1029/2019/gmd-12-1029-2019.pdf
    
    MUB = ds1.MUB
    DNW = ds1.DNW
    MU1_STAR = (ds1.MU + ds1.MUB) / MUB
    MU2_STAR = (ds2.MU + ds2.MUB) / MUB

    integration_factor = - MUB * DNW / g0  # notice that DNW is negative, so we need to multiply by -1
    integration_factor = integration_factor.transpose("bottom_top", "south_north", "west_east") 
    integration_range = ds1.PB >= 20000.0
   

    U1 = ds1.U.to_numpy()
    U1 = (U1[:, :, 1:] + U1[:, :, :-1]) / 2

    V1 = ds1.V.to_numpy()
    V1 = (V1[:, 1:, :] + V1[:, :-1, :]) / 2

    U2 = ds2.U.to_numpy()
    U2 = (U2[:, :, 1:] + U2[:, :, :-1]) / 2

    V2 = ds2.V.to_numpy()
    V2 = (V2[:, 1:, :] + V2[:, :-1, :]) / 2
    
    Q1_STAR = ds1.QVAPOR * MU1_STAR
    Q2_STAR = ds2.QVAPOR * MU2_STAR

    dQ_STAR = Q2_STAR - Q1_STAR
    dU = U2 - U1
    dV = V2 - V1
 
    IVT_X1 = (integration_factor * U1 * Q1_STAR).where(integration_range).sum(dim="bottom_top").rename("IVT_X1")

    IVT_Y1 = (integration_factor * V1 * Q1_STAR).where(integration_range).sum(dim="bottom_top").rename("IVT_Y1")

    IVT_X2 = (integration_factor * U2 * Q2_STAR).where(integration_range).sum(dim="bottom_top").rename("IVT_X2")

    IVT_Y2 = (integration_factor * V2 * Q2_STAR).where(integration_range).sum(dim="bottom_top").rename("IVT_Y2")


    dIVT_X = (integration_factor * (U2 * Q2_STAR - U1 * Q1_STAR)).where(integration_range).sum(dim="bottom_top").rename("dIVT_X")

    dIVT_Y = (integration_factor * (V2 * Q2_STAR - V1 * Q1_STAR)).where(integration_range).sum(dim="bottom_top").rename("dIVT_Y")

    dIVT_partQ_X = (integration_factor * U1 * dQ_STAR).where(integration_range).sum(dim="bottom_top").rename("dIVT_partQ_X")
    dIVT_partQ_Y = (integration_factor * V1 * dQ_STAR).where(integration_range).sum(dim="bottom_top").rename("dIVT_partQ_Y")

    dIVT_partV_X = (integration_factor * dU * Q1_STAR).where(integration_range).sum(dim="bottom_top").rename("dIVT_partV_X")
    dIVT_partV_Y = (integration_factor * dV * Q1_STAR).where(integration_range).sum(dim="bottom_top").rename("dIVT_partV_Y")

    dIVT_nonlinear_X = (integration_factor * dU * dQ_STAR).where(integration_range).sum(dim="bottom_top").rename("dIVT_nonlinear_X")
    dIVT_nonlinear_Y = (integration_factor * dV * dQ_STAR).where(integration_range).sum(dim="bottom_top").rename("dIVT_nonlinear_Y")

    dIVT_res_X = ( dIVT_X - (dIVT_partQ_X + dIVT_partV_X + dIVT_nonlinear_X) ).rename("dIVT_res_X")
    dIVT_res_Y = ( dIVT_Y - (dIVT_partQ_Y + dIVT_partV_Y + dIVT_nonlinear_Y) ).rename("dIVT_res_Y")

    return xr.merge([
        IVT_X1,
        IVT_Y1,
        IVT_X2,
        IVT_Y2,
        dIVT_X,
        dIVT_Y,
        dIVT_partQ_X,
        dIVT_partQ_Y,
        dIVT_partV_X,
        dIVT_partV_Y,
        dIVT_nonlinear_X,
        dIVT_nonlinear_Y,
        dIVT_res_X,
        dIVT_res_Y,
    ])
