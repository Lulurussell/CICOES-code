from matplotlib import pyplot as plt
import xarray as xr
from xarray_extras import csv
import numpy as np
import dask
from dask.diagnostics import progress
from tqdm.autonotebook import tqdm
import intake
import fsspec
import seaborn as sns
import pandas as pd
from tqdm.autonotebook import tqdm  # Fancy progress bars for our loops!
from dask_gateway import Gateway
from dask.distributed import Client
import cftime
import datetime
import math


col = intake.open_esm_datastore("https://storage.googleapis.com/cmip6/pangeo-cmip6.json")

expts = ['historical', 'ssp126', 'ssp245', 'ssp370', 'ssp585', 'piControl']

query = dict(
    experiment_id = expts,
    variable_id = ['zg'],
    table_id = ['Amon'],
    source_id = ['BCC-CSM2-MR'],
    member_id = 'r1i1p1f1'
)
col_subset = col.search(**query)

# print(col_subset.df.groupby("source_id")[
#     ["experiment_id", "variable_id", "table_id"]
# ].nunique())

def drop_all_bounds(ds):
    """Drop coordinates like 'time_bounds' from datasets,
    which can lead to issues when merging."""
    drop_vars = [vname for vname in ds.coords if (('_bounds') in vname ) or ('_bnds') in vname]
    return ds.drop(drop_vars)

def open_dsets(df):
    """Open datasets from cloud storage and return xarray dataset."""
    dsets = [xr.open_zarr(fsspec.get_mapper(ds_url), consolidated=True).pipe(drop_all_bounds) for ds_url in df.zstore]
    try:
        ds = xr.merge(dsets, join='exact')
        return ds
    except ValueError:
        return None
        
def open_delayed(df):
    """A dask.delayed wrapper around `open_dsets`.
    Allows us to open many datasets in parallel."""
    return dask.delayed(open_dsets)(df)
dsets_aligned = {}

from collections import defaultdict

dsets = defaultdict(dict)
for group, df in col_subset.df.groupby(by=['source_id', 'experiment_id']):
    dsets[group[0]][group[1]] = open_delayed(df)

open_dsets(df)
dsets_ = dask.compute(dict(dsets))[0]

expt_da = xr.DataArray(expts, dims='experiment_id', name='experiment_id', coords={'experiment_id': expts})

dsets_aligned = {}



def closest(ls, a):
    """Returns closest number to a in list ls"""
    return min(ls, key=lambda x:abs(x-a))

def PNA_prelim(ds, lat, lon):
    """Return data for zg at lat, lon with elevation at 50000"""
    return ds.sel(plev=50000, lon=float(closest(list(ds.sel(plev=50000)['lon']), lon)), lat=float(closest(list(ds.sel(plev=50000)['lat']), lat)))
def PNA(ds):

    # lat, lon = 20 N, 160 W
    a = PNA_prelim(ds, 20, 205)
    # lat, lon = 45 N, 165 W
    b = PNA_prelim(ds, 45, 200)

    #lat, lon = 55 N, 115 W
    c = PNA_prelim(ds, 55, 250)

    #lat, lon = 30 N, 85 W
    d = PNA_prelim(ds, 30, 280)

    return 0.25*(a - b + c - d).rename({'zg':'PNA'})

def GBI(ds):
    return ds.sel(plev=50000, lat=slice(float(closest(list(ds.sel(plev=50000)['lat']), 60)),
        float(closest(list(ds.sel(plev=50000)['lat']), 80))), lon=slice(float(closest(list(ds.sel(plev=50000)['lon']), 280)), 
        float(closest(list(ds.sel(plev=50000)['lon']), 340)))).mean(dim='lat').mean(dim='lon').rename({'zg':'GBI'})
def PCA(ds):
    return ds.sel(lat=slice(float(closest(list(ds['lat']), 65)),
        float(closest(list(ds['lat']), 90)))).mean(dim='lat').mean(dim='lon').rename({'zg':'PCA'})




for k, v in tqdm(dsets_.items()):
    expt_dsets = v.values()
    if any([d is None for d in expt_dsets]):
        print(f"Missing experiment for {k}")
        continue
    
    # Uncomment if plotting
    # Add a 'year' axis
    # for ds in expt_dsets:
    #     ds.coords['year'] = ds.time.dt.year

    # workaround for
    # https://github.com/pydata/xarray/issues/2237#issuecomment-620961663
    # Remove 'time' dimension for year, and coarsen the 'year' axis such that it is actually yearly data, as opposed to monthly
    # dsets_ann_mean = [v[expt].pipe(PCA)
    #                   for expt in expts]
    # #                          .swap_dims({'time': 'year'})
    # #                          .drop('time') 
    # #                          .coarsen(year=12).mean()
    # dsets_aligned[k] = xr.concat(dsets_ann_mean, join='outer',
    #                              dim=expt_da)

    # Write PCA
    for expt in expts:
        # Check if the experiment actually exists, skip iteration if not.
        try:
            a = v[expt]
        except KeyError:
            continue
        # Remember to replace file locaiton with correct one when running!
        print(f"writing data to /run/media/jqiu21/Elements/Jason/Internship/PCA/mon/PCA_{k}_{expt}.nc")
        with progress.ProgressBar():
            x = PCA(v[expt]).compute().sel(time=slice('2015-01-01', '2100-12-31'))
            x.to_netcdf(path=f"/run/media/jqiu21/Elements/Jason/Internship/PCA/mon/PCA_{k}_{expt}.nc", mode='w',  engine='netcdf4', format='NETCDF4')

    #Write PNA
    for expt in expts:
        try:
            a = v[expt]
        except KeyError:
            continue
        print(f"writing data to /run/media/jqiu21/Elements/Jason/Internship/PNA/mon/PNA_{k}_{expt}.nc")
        with progress.ProgressBar():
            x = PNA(v[expt]).compute().sel(time=slice('2015-01-01', '2100-12-31'))
        x.to_netcdf(path=f"/run/media/jqiu21/Elements/Jason/Internship/PNA/mon/PNA_{k}_{expt}.nc", mode='w',  engine='netcdf4', format='NETCDF4')
    #Write GBI
    for expt in expts:
        try:
            a = v[expt]
        except KeyError:
            continue
        print(f"writing data to /run/media/jqiu21/Elements/Jason/Internship/GBI/mon/GBI_{k}_{expt}.nc")
        with progress.ProgressBar():
            x = GBI(v[expt]).compute().sel(time=slice('2015-01-01', '2100-12-31'))
            x.to_netcdf(path=f"/run/media/jqiu21/Elements/Jason/Internship/GBI/mon/GBI_{k}_{expt}.nc", mode='w',  engine='netcdf4', format='NETCDF4')

# For plotting

# Concatenate all datasets (from multiple sources) into one
# with progress.ProgressBar():
#     dsets_aligned_ = dask.compute(dsets_aligned)[0]

# source_ids = list(dsets_aligned_.keys())
# source_da = xr.DataArray(source_ids, dims='source_id', name='source_id', coords={'source_id': source_ids})

# big_ds = xr.concat([ds.reset_coords(drop=True)
#                     for ds in dsets_aligned_.values()],
#                     dim=source_da)

# Create dataframe to plot from
# df_all = big_ds.sel(year=slice(1950, 2100)).to_dataframe().reset_index()
# print(df_all.head())

# Actually plot the data
# sns.set()
# p = sns.relplot(data=df_all,
#             x="year", y="PCA", hue='experiment_id',
#             kind="line", ci="sd", size=10, aspect=2, legend=False)

# I had some issues with random '9's or '10's in the automatically generated legend, so I made my own.
# plt.legend(title='experiment_id', loc='best', labels=expts)

# plt.show()