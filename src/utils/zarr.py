import zarr 
import queue
import dask.array as da
import xarray as xr
import os

class ZarrStore():
    def __init__(self, folder_path, size, n_workersd=4):

        self.folder_path = folder_path
        self.size = size
        self.n_workersd = n_workersd
    
    def zarr_store_create(self, day, label, channels):
        zarr_path = os.path.join(
        self.folder_path, f'MTG_FCI{day[0:4]}{day[5:7]}{day[8:10]}{label}.zarr'
    )
        #store = zarr.DirectoryStore(zarr_path)
        # Dimensions
        num_time = self.num_files
        cnks = 140
        height, width = self.size

        # Dummy time coordinate (optional but often required)
        time_coord = list(range(num_time))

        all_vars = {
            "vis_06", "vis_08", "vis_09",
            "ir_105", "ir_123", "ir_133",
            "ir_38", "ir_87", "ir_97",
            "wv_63", "wv_73"
        }

        assert all(v in all_vars for v in channels), "One or more channels are invalid"

        # Metadata variables always included
        meta_vars = ["identifier", "unixTimeStart", "unixTimeEnd"]

        # Filter to only requested channels
        selected_vars = set(channels)
        selected_vars.update(meta_vars)

        # Build data_vars and encoding dicts
        data_vars = {}
        encoding = {}
        compressor = zarr.Blosc(cname="zstd", clevel=9, shuffle=zarr.Blosc.SHUFFLE)

        for var in selected_vars:
            if var in meta_vars:
                shape = (num_time,)
                chunks = (1,)
                dtype = 'S143' if var == "identifier" else 'float64'
                dims = ("time",)
            else:
                shape = (num_time, height, width)
                chunks = (1, cnks, width)
                dtype = 'int16'
                dims = ("time", "y", "x")

            data_vars[var] = (dims, da.empty(shape, dtype=dtype, chunks=chunks))
            encoding[var] = {"compressor": compressor, "chunks": chunks}

        # Create and write dataset
        ds_empty = xr.Dataset(data_vars=data_vars, coords={"time": time_coord})
        ds_empty.to_zarr(zarr_path, mode='w', encoding=encoding, compute=True)

        return zarr_path, encoding
