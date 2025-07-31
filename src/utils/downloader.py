import eumdac
import datetime
import subprocess
from datetime import timedelta, datetime, timezone
from eumdac.collection import SearchResults
from eumdac.tailor_models import  RegionOfInterest
from concurrent.futures import as_completed    
from pathlib import Path
import time
import fnmatch
import eumdac
from typing import Literal
import warnings
import os
import concurrent 
from dotenv import load_dotenv
import zipfile
import shutil
from tqdm.auto import tqdm
from pydantic import validate_call, PositiveInt
import logging
import queue
import numpy as np
import threading
import xarray as xr
import dask.array as da
import pandas as pd
from dask.diagnostics import ProgressBar
from pyresample.geometry import AreaDefinition
from typing import Union
from pyresample.utils import proj4_str_to_dict

tb = ProgressBar().register()


logger = logging.getLogger(__name__)

"""
Part of the code has been adapted from monkey_wrench https://github.com/pkhalaj/monkey-wrench
"""

load_dotenv()

products_list = {
    "MTG-1-HR": {"product_id":"EO:EUM:DAT:0665",
              "product_name":"FCIL1HRFI",
              "bands":["vis_06_hr_effective_radiance", "vis_08_hr_effective_radiance",]}, 
    "MTG-1": {"product_id":"EO:EUM:DAT:0662",
                "product_name":"FCIL1FDHSI",
                "bands":["vis_06_effective_radiance", "vis_08_effective_radiance",]},
    "MTG Cloud Mask": {"product_id":"EO:EUM:DAT:0666",
                       "product_name":"FCIL2CLM"},
    }

class EUMDownloader:
    """Downloader for MTG products."""
    
    @validate_call  
    def __init__(self, 
                 product_id:str, 
                 output_dir:str,
                 format:Literal["netcdf4","geotiff"]='netcdf4',
                 sleep_time:PositiveInt=10,
                 max_parallel_conns:PositiveInt=10):

        self._token = None
        self._token_expiration = None

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.SLEEP_TIME = sleep_time # seconds
        self.max_parallel_conns = max_parallel_conns

        if format not in ["netcdf4", "geotiff"]:
            raise ValueError("Format must be either 'netcdf4' or 'geotiff'.")
        self.format = format

        self.get_token()
        self._data_store = self.initialize_datastore(self._token)
        self._data_tailor = self.initialize_datatailor(self._token)
        self._selected_collection = self._data_store.get_collection(product_id)
    
        product_key = [k for k, v in products_list.items() if v["product_id"] == product_id][0]
        self.product_name = products_list[product_key]["product_name"]
        self.product_id = product_id
        self.channels = products_list[product_key]["bands"]


    @staticmethod
    def len(product_ids: SearchResults) -> int:
        """Return the number of product IDs."""
        return product_ids.total_results
        
    def initialize_datastore(self, token):
        datastore = eumdac.DataStore(token)
        logger.info("Datastore initialized.")
        return datastore
    
    def initialize_datatailor(self, token):
        datatailor = eumdac.DataTailor(token)
        logger.info("DataTailor initialized.")
        return datatailor
    
    def get_token(self):
        if self._token is None or self._is_token_expired():
            consumer_key = os.getenv("EUMETSAT_CONSUMER_KEY")
            consumer_secret = os.getenv("EUMETSAT_CONSUMER_SECRET")
            self._token = eumdac.AccessToken((consumer_key, consumer_secret))
            self._token_expiration = self._token.expiration
            logger.info(f"New token expires at {self._token_expiration}")
    
    def _is_token_expired(self, buffer_minutes: int = 5) -> bool:
        """Check if token is expired or close to expiration (with a buffer)."""
        if self._token_expiration is None:
            return True
        now = datetime.now(datetime.timezone.utc)
        return now >= self._token_expiration - timedelta(minutes=buffer_minutes)

    def _download_products(self, interval_list, coll_obj, dest_folder, bounding_box):
        start = interval_list[0]
        end = interval_list[1]
        products = coll_obj.search(dtstart=start, dtend=end, bbox=bounding_box)
        for product in products:
            with product.open() as fsrc, open(Path(dest_folder) / fsrc.name, mode='wb') as fdst:
                shutil.copyfileobj(fsrc, fdst)

            with zipfile.ZipFile(Path(dest_folder) / fsrc.name, 'r') as zip_ref:
                for file in zip_ref.namelist():
                    if file.startswith(str(product)):
                        zip_ref.extract(file, self.output_dir)
                logger.info(f'Unzipping of product {fdst.name} finished.')

        os.remove(Path(dest_folder) / fsrc.name)
    
    @validate_call
    def _split_time_into_daily_observations(self, 
                                       start_date, 
                                       end_date, 
                                       observations_per_day=1, 
                                       start_hour=10, 
                                       interval_minutes=10,
                                       jump_minutes=0):
        """
        Generate time intervals for multiple observations per day.

        Parameters:
        - start_date, end_date: datetime.date or datetime.datetime
        - observations_per_day: number of intervals per day
        - start_hour: hour of first observation (UTC)
        - interval_hours: duration of each observation in hours

        Returns:
        - List of (start_time, end_time) tuples in '%Y%m%dT%H%M%SZ' format
        """
        intervals = []

        # Normalize to date if datetime was passed
        if isinstance(start_date, datetime):
            start_date = start_date.date()

        if isinstance(end_date, datetime):
            end_date = end_date.date()

        current_date = start_date

        while current_date <= end_date:
            interval_start = datetime.combine(current_date, datetime.min.time()) + timedelta(hours=start_hour)
            for i in range(observations_per_day):
                interval_end = interval_start + timedelta(minutes=interval_minutes)
                intervals.append((interval_start, interval_end))

                if jump_minutes > 0:
                    interval_start += timedelta(minutes=jump_minutes)
                else:
                    interval_start = interval_end

            current_date += timedelta(days=1)

        for i, (start, end) in enumerate(intervals, 1):
            logger.info(f"{i:03d}. Start: {start.strftime('%Y-%m-%d %H:%M:%S')}  →  End: {end.strftime('%Y-%m-%d %H:%M:%S')}")

        return intervals
    
    @validate_call
    def _split_intervals_for_threading(self, time_intervals: list[tuple[datetime, datetime]], n_threads: int):
        """
        Split a list of time intervals into `n_threads` roughly equal chunks for parallel processing.
    
        Parameters:
        - time_intervals: list of (start_time, end_time) tuples
        - n_threads: number of threads to divide the intervals into
    
        Returns:
        - List of lists, each containing tuples for one thread
        """
        from math import ceil
    
        total = len(time_intervals)
        chunk_size = ceil(total / n_threads)
        thread_chunks = [time_intervals[i:i + chunk_size] for i in range(0, total, chunk_size)]
    
        logger.info("\nDownload will be performed simultaneously for the following time chunks:")
        for i, chunk in enumerate(thread_chunks):
            logger.info(f"Thread {i+1}: {len(chunk)} intervals, from {chunk[0][0]} to {chunk[0][1]}")
    
        return thread_chunks
    
    def chunks_download(self, file_list, bounding_box):
        chunks = self._split_intervals_for_threading(self.intervals, self.max_parallel_conns)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel_conns) as executor:
            futures = [executor.submit(self._download_products, chunk, self._selected_collection, self.output_dir, bounding_box)
                       for chunk in chunks]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Download failed with error: {e}")
        

    def download_interval(self, 
                          start_time, 
                          end_time, 
                          bounding_box:list=None, 
                          method:Literal[None, "datatailor", "datastore"]=None,
                          observations_per_day=1, 
                          start_hour=12, 
                          interval_minutes=10,
                          jump_minutes=0
        ):

        from utils import assert_datetime_is_timezone_aware, assert_start_precedes_end

        """Download products from the datastore."""
        logger.info(f"Downloading {self.product_id} from {start_time} to {end_time}...")
                # Set sensing start and end time
        start = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        end = datetime.strptime(end_time, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)

        for t in [start, end]:
            assert_datetime_is_timezone_aware(t)

        assert_start_precedes_end(start, end)

        self.intervals = self._split_time_into_daily_observations(
            start_date=start,
            end_date=end,
            observations_per_day=observations_per_day,
            start_hour=start_hour,
            interval_minutes=interval_minutes,
            jump_minutes=jump_minutes
        )

        self.start_date = start_time
        self.end_date = end_time
        self.file_list = self._collect_products(self.intervals, bounding_box)

        if method == "datatailor":
            self._datatailor_download(
                self.file_list,
                bounding_box=bounding_box,
            )

        elif method == "datastore":
            self.chunks_download(
                self.file_list, 
                bounding_box
            )
            
            
    def _datastore_thread_download(self, intervals, bounding_box=None):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import zipfile
        from tqdm import tqdm
        """
        Download products from the Datastore service using threading and monitor with tqdm.
        """
        
        coll_obj = self.__selected_collection
        destination_folder = self.output_dir
        intervals_thread = self._split_intervals_for_threading(intervals, self.max_parallel_conns)

        def download_products_in_thread(interval_list):
            for start, end in interval_list:
                products = coll_obj.search(dtstart=start, dtend=end, bbox=bounding_box)
                for product in products:
                    with product.open() as fsrc, open(Path(destination_folder) / fsrc.name, mode='wb') as fdst:
                        shutil.copyfileobj(fsrc, fdst)

                with zipfile.ZipFile(os.path.join(destination_folder, fsrc.name), 'r') as zip_ref:
                    for file in zip_ref.namelist():
                        if file.startswith(str(product)):
                            zip_ref.extract(file, self.output_dir)
                    logger.info(f'Unzipping of product {fdst.name} finished.')

                os.remove(Path(destination_folder) / fsrc.name)

        # Use tqdm to track completion of futures
        with ThreadPoolExecutor(max_workers=self.max_parallel_conns) as executor:
            futures = [executor.submit(download_products_in_thread, chunk) for chunk in intervals_thread]

            for _ in tqdm(as_completed(futures), total=len(futures), desc="Downloading by interval batch"):
                pass  # tqdm updates for each completed thread

    def _datatailor_download(self,
                             file_list,
                             bounding_box):
        
        from eumdac.tailor_models import Chain

        # Define the chain configuration
        chain = Chain(
            product=self.product_name,
            format=self.format,
            roi=RegionOfInterest(NSWE = bounding_box),
            filter={"bands" : self.channels},
            projection='geographic',
        )

        [self._download_customisation(product, chain, self.output_dir) for product in file_list]
            
    def _collect_products(self,
                    intervals, 
                    bounding_box=None):
        
        """Download products using the DataTailor service."""
        

        file_list = []

        #loop over intervals and run the chain for each interval
        for interval in tqdm(intervals, desc="Collecting products specs..."):
            start, end = interval

            # Retrieve latest product that matches the filter
            selected_products = self._selected_collection.search(
                dtstart=start,
                dtend=end,
                bbox=bounding_box)
            
            file_list.extend(selected_products)
            
            if selected_products.total_results > 1:
                logger.debug(f'Found Multiple Datasets: {selected_products.total_results} for the given time range') 
            
        return file_list

    
    def _download_customisation(self, product, chain, dest_path):
        customisation = self._data_tailor.new_customisation(product, chain=chain)
        """Polls the status of a customisation and downloads the output once completed."""
        while True:
            status = customisation.status

            if "DONE" in status:
                logger.info(f"Customisation {customisation._id} successfully completed.")
                try:
                    # Safely get the first matching output file
                    cust_file = next(f for f in customisation.outputs if fnmatch.fnmatch(f, '*'))

                    with customisation.stream_output(cust_file) as stream, open(stream.name, mode='wb') as fdst:
                        shutil.copyfileobj(stream, fdst)

                    logger.info(f"Download finished for customisation {customisation._id}.")

                except (StopIteration, AttributeError) as e:
                    logger.error(f"No matching output found for customisation {customisation._id}.")
                except Exception as e:
                    logger.error(f"Failed to download output: {e}")
                break

            elif status in ["ERROR", "FAILED", "DELETED", "KILLED", "INACTIVE"]:
                logger.warning(f"Customisation {customisation._id} was unsuccessful. Log output:")
                logger.warning(customisation.logfile)
                break

            elif "QUEUED" in status:
                logger.info(f"Customisation {customisation._id} is queued.")
            elif "RUNNING" in status:
                logger.info(f"Customisation {customisation._id} is running.")

            time.sleep(self.SLEEP_TIME)

            for product in customisation.outputs:
                with customisation.stream_output(product) as source_file, open(os.path.join(dest_path, source_file.name), 'wb') as destination_file:
                    shutil.copyfileobj(source_file, destination_file)
                logger.info(f"Product {product} downloaded successfully.")

        customisation.delete()


    def _download_multiprocessing(self, chain_config, intervals, product):
        """Run the chain configuration for each interval in parallel."""
        input = str(product) + ".zip"
        output = str(product)
        if not os.path.exists(output):
            os.makedirs(output)
        for start, end in intervals:
            subprocess.run(["epct", "run-chain", "-f", chain_config, "--sensing-start", start, "--sensing-stop", end, input, "-o", output])


class MTGDataParallel():
    def __init__(self,  
                downloader: EUMDownloader, 
                channels:list= ['vis_06',  'vis_08'],
                area_reprojection:Union[None, str]=None,
                processes:PositiveInt=4,
                chunks: dict = {"time": 1, "lon": "auto", "lat": "auto"}
                ):
        
        from utils import compute_auto_chunks

        self.file_list = downloader.file_list
        self.output_dir = downloader.output_dir
        self.size = self._get_size(area_reprojection)
        self._reproject = area_reprojection
        self.processes = processes
        input_shape = {"time": chunks["time"], "lat": self.size[0], "lon":self.size[1]}
        self.chunks = compute_auto_chunks(shape= input_shape)

        channelsIR = ['ir_105', 'ir_123',  'ir_133',  'ir_38',  'ir_87',  'ir_97',  'wv_63',  'wv_73']    
        channelsVIS= ['nir_13', 'nir_16',  'nir_22',  'vis_04',  'vis_05', 'vis_06',  'vis_08',  'vis_09', ]
        
        assert all(f in channelsIR or f in channelsVIS for f in channels), \
           "One or more channels are not in channelsIR or channelsVIS"
        logger.info("The initialized class contains {} files".format(len(self.file_list)))
        self.channels = channels
        self.zip_path = Path(self.output_dir) / "zipfolder"
        os.makedirs(self.zip_path, exist_ok=True)
        self.nat_path = Path(self.output_dir) / "natfolder"
        os.makedirs(self.nat_path, exist_ok=True)

        self.download_to_zarr(self.file_list)

    def _get_size(self, area_reprojection:str):
        if area_reprojection == "worldeqc3km":
            return [2048, 4096]
        elif area_reprojection == "EPSG_4326_36000x18000":
            return [18000, 36000] 
        elif area_reprojection == "msg_seviri_fes_1km":
            return [11136, 11136]
        else: 
            raise NotImplementedError("Area {} not implemented".format(area_reprojection))


    def download_to_zarr(self, file_list):
        from utils import ZarrStore
        t0= time.time()
        file_list = sorted(self.file_list, 
            key=lambda p: np.datetime64(p._browse_properties['date'].split('/')[0][0:-1]))
        
        store = ZarrStore(self.output_dir, 
                          size=self.size, 
                          file_list=self.file_list,
                          channels=self.channels,
                          chunks= self.chunks)
        
        self._remove_all_tempfiles()
        download_queue = queue.Queue()
        read_pbar = tqdm(total=len(self.file_list), desc="Reading files ", position=1, leave=True)
        # Start reader thread
        reader_thread = threading.Thread(target=self.read_convert_append, args=(download_queue, read_pbar, store.path))
        reader_thread.start()
        download_pbar = tqdm(total=len(self.file_list), desc="Downloading files", position=0, leave=True)
        
        # Use ThreadPoolExecutor to download files in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.processes) as executor:
            futures = [executor.submit(self._download_file, self.file_list[t], t, download_queue) for t in range(len(file_list))]
            for _ in as_completed(futures):
                download_pbar.update(1)

        for _ in range(self.processes):
            download_queue.put((None, None, None, None))
            
        reader_thread.join()
        self._remove_all_tempfiles()

        elapsed_seconds = time.time() - t0
        hours = int(elapsed_seconds // 3600)
        minutes = int((elapsed_seconds % 3600) // 60)
        logger.info(f"Done in {hours} hours {minutes} minutes") 

    def _remove_all_tempfiles(self):
        import os
        for filename in os.listdir(self.nat_path):
           file_path = os.path.join(self.nat_path, filename)
           if os.path.isfile(file_path):
              os.remove(file_path)
        for filename in os.listdir(self.zip_path):
           file_path = os.path.join(self.zip_path, filename)
           if os.path.isfile(file_path):
              os.remove(file_path)
        return

    def _download_file(self, product, t, download_queue):
        #global download_queue
        filename = self._download_zipfile(product, self.zip_path)
        download_queue.put((filename, t, product))

    def _download_zipfile(self, product, dest_folder):
        dsnm=product.metadata["properties"]["title"]
        dssz=product.metadata["properties"]["productInformation"]["size"]
        outfilename = os.path.join(dest_folder, dsnm) +'.zip'

        if os.path.isfile(outfilename):
            szdsk=os.path.getsize(outfilename)
            if szdsk/1000 > dssz:
                return
        with product.open() as fsrc, \
                open(os.path.join(dest_folder, fsrc.name), mode='wb') as fdst:
            shutil.copyfileobj(fsrc, fdst)
        return os.path.join(dest_folder, fsrc.name)
    

    def _get_coords_area(self):
            """Create an AreaDefinition for the dataset."""
            lons, lats = self._area_def.get_lonlats()
            return lons, lats
    
    def _read_satpy_convert(self, natfolder, t):
        from satpy.scene import Scene
        from satpy import find_files_and_readers
        from satpy import _scene_converters as convert
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  

        calibration ='reflectance'
        path_to_data = natfolder
        fill_value = -32768
    
        # find files and assign the FCI reader
        files = find_files_and_readers(base_dir=path_to_data, reader='fci_l1c_nc',missing_ok=True)
        # create an FCI scene from the selected files
        scn = Scene(filenames=files)       
        scn.load(self.channels, calibration=calibration, upper_right_corner='NE')

        if self._reproject:
            logger.info(f"Reprojecting data to {self._reproject} coordinates...")
            scn_resampled = scn.resample(area=self._reproject)            
        else:
            scn_resampled = scn

        xscene = convert.to_xarray(scn_resampled)
        example = xscene[self.channels[0]]
        lats = example.coords['latitude'].values
        lons = example.coords['longitude'].values
        self._area_def =scn_resampled[self.channels[0]].attrs["area"]
                
        out={}

        fill_value=-32768.0
        for channel in self.channels:
            # _ = scn_resampled[channel].values # to trigger the loading of the data
            data = xscene[channel]
            data = data.expand_dims(dim={'time': [t]})
            data = data.where(~xr.ufuncs.isnan(data), fill_value)
            data = data.where(~xr.ufuncs.isinf(data), fill_value)
            data = data.clip(min=-32768, max=32767)
            data = data.astype('float32')
            
            data_array = xr.DataArray(
                data.drop_vars(["x","y"]).rename({"y":"lat", "x":"lon"}),
                dims=('time', 'lat', 'lon'),
                name=channel,
                attrs=data.attrs
            )

            out[channel] = data_array

        # Build dataset and rename dims to lat/lon
        ds = xr.Dataset(out).chunk(self.chunks)
        ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})

        # Assign lat/lon as 2D coordinates
        ds = ds.assign_coords(
            time=[scn_resampled["vis_06"].attrs["time_parameters"]["nominal_start_time"]],
            lat=(('lat', 'lon'), lats),
            lon=(('lat', 'lon'), lons)
        )
        ds = self._clean_metadata(ds)

        if self._reproject is not None:
            ds= ds.persist()
        else:
            ds = ds.compute()

        return t, ds

    def str2unixTime(self, stime):
        return np.datetime64(stime)
    
    def _clean_metadata(self, ds, all:bool = False):
        if all:
            # Clean up metadata
            ds.attrs = {}
            for var in ds.data_vars:
                ds[var].attrs = {}
        else:
            for a in ["FillValue", "_FillValue"]:
                for var in ds.data_vars:
                    if a in ds[var].attrs:
                        del ds[var].attrs[a]

            
        return ds

    def read_convert_append(self, download_queue, read_pbar, zarr_path):
        while True:
            filename, t, product = download_queue.get()[:3]
            if filename is None:  # Sentinel to stop thread
                break

            natfolder_t = os.path.join(self.nat_path, str(t))

            # Unzip the .nc file(s)
            with zipfile.ZipFile(filename) as zf:
                for fnat in zf.namelist():
                    if fnat.endswith('.nc'):
                        zf.extract(fnat, natfolder_t)

            # Read dataset
            t, ds = self._read_satpy_convert(natfolder_t, t)
            identifier = product._browse_properties['identifier']
            date_range = product._browse_properties['date']

            # Add metadata
            id_bytes = np.array(identifier, dtype='S143')
            t_start = self.str2unixTime(date_range.split('/')[0][:-1])
            t_end   = self.str2unixTime(date_range.split('/')[1][:-1])

            ds['identifier'] = xr.DataArray([id_bytes], dims=('time',), coords={'time': [0]})
            ds['unixTimeStart'] = xr.DataArray([t_start], dims=('time',), coords={'time': [0]})
            ds['unixTimeEnd'] = xr.DataArray([t_end], dims=('time',), coords={'time': [0]})
            # ds = ds.assign_coords(time=[pd.Timestamp(t)])

            ds.drop_vars(["lat", "lon"]).to_zarr(
                zarr_path,
                region={"time": slice(t, t + 1)},
                compute=True
            )

            read_pbar.update(1)
            download_queue.task_done()