# This is a very basic example of using chunks to make computation easier. 
# Due to the large number of dimensions, if you didn't chunk the dataset, you'd excede 16 GB of RAM. With chunks, you can run this in 23.1s while barely using any memory.
# More info here: example of using chunking for low-memory situations

import xarray as xr
import dask
from dask.diagnostics import progress
fpath = '/home/jqiu21/Documents/Internship/'

ds = xr.open_dataset(fpath + 'Tair-1979-JFM00.nc', chunks={'time':1})

print(ds)

delayed_obj = ds.rename({'t':'tair'}).to_netcdf(fpath + 'Tair-1979-JFM00_redim.nc', mode='w', compute=False, engine='netcdf4', format='NETCDF4')
with progress.ProgressBar():
    delayed_obj.compute()