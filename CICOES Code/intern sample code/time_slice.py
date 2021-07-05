from matplotlib import pyplot as plt
import xarray as xr
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


# df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
# print(df.head())
col = intake.open_esm_datastore("https://storage.googleapis.com/cmip6/pangeo-cmip6.json")

# print([eid for eid in col.df['experiment_id'].unique() if 'ssp' in eid])


expts = ['historical'] #'ssp126', 'ssp245', 'ssp370', 'ssp585', 

query = dict(
    experiment_id = expts,
    variable_id = ['tas'],
    table_id = ['Amon'],
    source_id = ['BCC-CSM2-MR', 'CAMS-CSM1-0', 'CESM2-WACCM', 'EC-Earth3-Veg', 'IPSL-CM6A-LR', 'MIROC6', 'MPI-ESM1-2-HR', 'MRI-ESM2-0'],
    member_id = 'r1i1p1f1'
)
col_subset = col.search(require_all_on=["source_id"], **query)

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

from collections import defaultdict

dsets = defaultdict(dict)
for group, df in col_subset.df.groupby(by=['source_id', 'experiment_id']):
    dsets[group[0]][group[1]] = open_delayed(df)

open_dsets(df)
dsets_ = dask.compute(dict(dsets))[0]

expt_da = xr.DataArray(expts, dims='experiment_id', name='experiment_id', coords={'experiment_id': expts})

dsets_aligned = {}

for k, v in tqdm(dsets_.items()):
    expt_dsets = v.values()
    if any([d is None for d in expt_dsets]):
        print(f"Missing experiment for {k}")
        continue

    climatology = v['historical'].sel(time=slice('1981-01-01', '2010-12-31')).groupby('time.month').mean('time')


    for i in v:
        if i == 'historical':
            anomaly = v[i].sel(time=slice('1981-01-01', '2010-12-31')).groupby('time.month') - climatology
        else:
            anomaly = v[i].groupby('time.month') - climatology
        # Because these files are too large to store in memory, we use the option compute=False to create a dask delayed object and then compute it later.
        delayed_obj = anomaly.to_netcdf(path=f"/run/media/jqiu21/Elements/Jason/Internship/tas_anomaly/tas-anomaly_{k}_{i}.nc", mode='w', compute=False, engine='netcdf4', format='NETCDF4')
        print(f"writing data to /run/media/jqiu21/Elements/Jason/Internship/tas_anomaly/tas-anomaly_{k}_{i}.nc")

        with progress.ProgressBar():
            results = delayed_obj.compute()