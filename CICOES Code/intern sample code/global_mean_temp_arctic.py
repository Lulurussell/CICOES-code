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


# df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
# print(df.head())
col = intake.open_esm_datastore("https://storage.googleapis.com/cmip6/pangeo-cmip6.json")

# print([eid for eid in col.df['experiment_id'].unique() if 'ssp' in eid])


expts = ['historical', 'ssp126', 'ssp245', 'ssp370', 'ssp585']

query = dict(
    experiment_id = expts,
    variable_id = ['tas'],
    table_id = ['Amon'],
    source_id = ['ACCESS-ESM1-5','CESM2-WACCM', 'MPI-ESM1-2-HR'],
    member_id = 'r1i1p1f1'
)
col_subset = col.search(require_all_on=["source_id"], **query)
# col_subset.df.groupby("source_id")[
#     ["experiment_id", "variable_id", "table_id"]
# ].nunique()

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

# calculate global means

def get_lat_name(ds):
    """Get what the dimension name for latitude is (either 'lat' or 'latitude'"""
    for lat_name in ['lat', 'latitude']:
        if lat_name in ds.coords:
            return lat_name
    raise RuntimeError("Couldn't find a latitude coordinate")
def modified_weight(lat):
    return xr.where(lat > 60, np.cos(np.deg2rad(lat)), 0)
def global_mean(ds):
    """Take the global temperature mean across latitude and longitude"""
    # To take the mean across time, you can simply use the .mean() method.
    lat = ds[get_lat_name(ds)]
    weight = np.cos(np.deg2rad(lat))
    weight /= weight.mean()
    other_dims = set(ds.dims) - {'time'}
    return (ds * weight).mean(other_dims)

def arctic_mean(ds):
    """Modified version of global mean - only considers temperatures in the arctic circle"""
    lat = ds[get_lat_name(ds)]
    weight = modified_weight(lat)
    weight /= weight.mean()
    other_dims = set(ds.dims) - {'time'}
    return (ds * weight).mean(other_dims)


expt_da = xr.DataArray(expts, dims='experiment_id', name='experiment_id', coords={'experiment_id': expts})

dsets_aligned = {}

for k, v in tqdm(dsets_.items()):
    expt_dsets = v.values()
    if any([d is None for d in expt_dsets]):
        print(f"Missing experiment for {k}")
        continue

    for ds in expt_dsets:
        ds.coords['year'] = ds.time.dt.year

    # workaround for
    # https://github.com/pydata/xarray/issues/2237#issuecomment-620961663
    dsets_ann_mean = [v[expt].pipe(arctic_mean)
                             .swap_dims({'time': 'year'})
                             .drop('time')
                             .coarsen(year=12).mean()
                      for expt in expts]

    # align everything with the 4xCO2 experiment
    dsets_aligned[k] = xr.concat(dsets_ann_mean, join='outer',
                                 dim=expt_da)

with progress.ProgressBar():
    dsets_aligned_ = dask.compute(dsets_aligned)[0]

source_ids = list(dsets_aligned_.keys())
source_da = xr.DataArray(source_ids, dims='source_id', name='source_id', coords={'source_id': source_ids})

big_ds = xr.concat([ds.reset_coords(drop=True)
                    for ds in dsets_aligned_.values()],
                    dim=source_da)
# print(big_ds)

df_all = big_ds.sel(year=slice(1900, 2100)).to_dataframe().reset_index()
# print(df_all.head())

sns.set()
p = sns.relplot(data=df_all,
            x="year", y="tas", hue='experiment_id',
            kind="line", ci="sd", size=10, aspect=2, legend=False)

plt.legend(title='experiment_id', loc='best', labels=['historical', 'ssp126', 'ssp245', 'ssp370', 'ssp585'])

plt.show()

