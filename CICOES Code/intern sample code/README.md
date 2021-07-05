# Some python scripts for computing climate data

## About this code

This code makes use of Pangeo's Public Dataset, and contains some code from their examples, found [here](https://gallery.pangeo.io/repos/pangeo-gallery/cmip6/). Each script has a similarly named Jupyter Notebook that explains some of the code. If you find my explainations limited, much of what I wrote is already explained in a much better way in documentation found online. This documentation is linked at some point in one of the notebooks.

## Running this code

You can run this code as any other Python script. However, do note that many of the scripts will write computed data to netCDF files. Be sure you have enough disk space to hold these, as the files for daily data can get quite large. Additionally, do be sure to change the file location to your desired location before running. You can also run this code as a notebook.

## About each file

### time_slice

This contains code for calculating the tas anomaly of a dataset. It also has some (although) limited code explaining some preliminary setup that is contained in all 3 files. 

### geopotential_height_zg

This contains code for making 3 different calculations involving geopotential.

### global_mean_temp_arctic

This contains code for plotting data, as well as for calculating the global mean temperature and arctic mean temperature.

## More info and resources

[GCPy_demo](https://gcpy-demo.readthedocs.io/en/latest/advanced_xarray.html) example of using chunking for low-memory situations. More info can be found in the [docs](https://xarray.pydata.org/en/stable/generated/xarray.Dataset.chunk.html?highlight=chunk).

