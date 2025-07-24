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
        self.__data_store = self.initialize_datastore(self._token)
        self.__data_tailor = self.initialize_datatailor(self._token)
        self.__selected_collection = self.__data_store.get_collection(product_id)
    
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
    
    def parallel_download(self, time_intervals: list[tuple[datetime, datetime]], bounding_box):
        chunks = self._split_intervals_for_threading(time_intervals, self.max_parallel_conns)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel_conns) as executor:
            futures = [executor.submit(self._download_products, chunk, self.__selected_collection, self.output_dir, bounding_box)
                       for chunk in chunks]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Download failed with error: {e}")
        

    def download_interval(self, 
                          start_time, 
                          end_time, roi:str=None, 
                          bounding_box:list=None, 
                          method:Literal["datatailor", "datastore"]="datatailor",
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

        intervals = self._split_time_into_daily_observations(
            start_date=start,
            end_date=end,
            observations_per_day=observations_per_day,
            start_hour=start_hour,
            interval_minutes=interval_minutes,
            jump_minutes=jump_minutes
        )

        if method == "datatailor":
            self._datatailor_download(
                intervals,
                bounding_box=bounding_box,
            )

        elif method == "datastore":
            self.parallel_download(
                intervals, 
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
                    intervals, 
                    bounding_box=None):
        
        """Download products using the DataTailor service."""
        
        from eumdac.tailor_models import Chain

        # Define the chain configuration
        chain = Chain(
            product=self.product_name,
            format=self.format,
            roi=RegionOfInterest(NSWE = bounding_box),
            filter={"bands" : self.channels},
            projection='geographic',
        )
        
        #loop over intervals and run the chain for each interval
        for interval in tqdm(intervals):
            start, end = interval

            # Retrieve latest product that matches the filter
            selected_products = self.__selected_collection.search(
                dtstart=start,
                dtend=end,
                bbox=bounding_box)
            
            if selected_products.total_results > 1:
                logger.debug(f'Found Multiple Datasets: {selected_products.total_results} for the given time range') 
            
            [self._download_customisation(product, chain, self.output_dir) for product in selected_products]
            

    
    def _download_customisation(self, product, chain, dest_path):
        customisation = self.__data_tailor.new_customisation(product, chain=chain)
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


class MTGVisData(EUMDownloader):
    def __init__(self, day, size:list = [5568,5568], channels:list= ['vis_06',  'vis_08',  'vis_09']):
        super(EUMDownloader).__init__()

        self.file_list = [f for f in os.listdir(self.output_dir)]
        self.channels = channels
        self.day = day
        self.size = size
        self.zip_path = Path(self.output_dir) / "zipfile"
        os.makedirs(self.zip_path, exist_ok=True, parents=True)
        self.nat_path = Path(self.output_dir) / "natfolder"
        os.makedirs(self.nat_path, exist_ok=True, parents=True)
        

    def netcdf_zarr_pipeline(self):
        file_list = sorted(self.file_list, key=lambda p: np.datetime64(p._browse_properties['date'].split('/')[0][0:-1]))
        download_queue = queue.Queue()
        read_pbar = tqdm(total=len(file_list), desc="Reading files ", position=1, leave=True)
        # Start reader thread
        reader_thread = threading.Thread(target=self.reader, args=(download_queue, read_pbar))
        reader_thread.start()
        download_pbar = tqdm(total=len(file_list), desc="Downloading files", position=0, leave=True)
        # Use ThreadPoolExecutor to download files in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.download_file, file_list[t], t, download_queue) for t in range(len(file_list))]
            for _ in as_completed(futures):
                download_pbar.update(1)

    def download_file(self, product, t, download_queue):
        #global download_queue
        filename = self.eum_download_file(product, self.zip_path)
        download_queue.put( (filename,t,product))

    def eum_download_file(self, product, dest_folder):
        dsnm=product.metadata["properties"]["title"]
        dssz=product.metadata["properties"]["productInformation"]["size"]
        outfilename = os.path.join(dest_folder, dsnm)+'.zip'

        if os.path.isfile(outfilename):
            szdsk=os.path.getsize(outfilename)
            if szdsk/1000 > dssz:
                return
        with product.open() as fsrc, \
                open(os.path.join(dest_folder, fsrc.name), mode='wb') as fdst:
            shutil.copyfileobj(fsrc, fdst)
        return os.path.join(dest_folder, fsrc.name)
    
    def read_zarrDask_fast(self, natfolder, t):
        from satpy.scene import Scene
        from satpy import find_files_and_readers
        from satpy import _scene_converters as convert
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  

        calibration ='brightness_temperature'
    
        tt0=time.time()
        path_to_data = natfolder
        fill_value = -32768
    
        # find files and assign the FCI reader
        files = find_files_and_readers(base_dir=path_to_data, reader='fci_l1c_nc',missing_ok=True)
    
        # create an FCI scene from the selected files
        tt1=time.time()
        scn = Scene(filenames=files)
        tt2=time.time()
        # note : https://satpy.readthedocs.io/en/latest/_modules/satpy/readers/fci_l1c_nc.html#FCIL1cNCFileHandler.calibrate_rad_to_bt
        #https://satpy.readthedocs.io/en/latest/api/satpy.readers.fci_l1c_nc.html   
        
        out={}
        
        scn.load(self.channels, upper_right_corner='NE',calibration=calibration)
        tt3=time.time()
        xscene = convert.to_xarray(scn)
        fill_value=-32768.0
        for channel in self.channels:
            image = xscene[channel] 
            data = image.data * 10
            data = da.where(da.isnan(data), fill_value, data)
            data = da.where(da.isinf(data), fill_value, data)  # or 32767.0 / -32768.0 if preferred
    
            data = da.clip(data, -32768, 32767)
            data = data.astype('int16')
            data = da.expand_dims(data, axis=0) 
            data_array = xr.DataArray(
                data,
                dims=('time', 'y', 'x'),
                name=channel)
            out[channel] = data_array
        tt4=time.time()    
        ds = xr.Dataset(out)
        ds= ds.persist()

        return t,ds

    def str2unixTime(self, stime):
        return np.datetime64(stime)

    def reader(self, download_queue, read_pbar):
        while True:
            filename, t, product, set0 = download_queue.get()
            if filename is None:  # Sentinel to stop thread
                break

            natfolder_t = os.path.join(self.nat_path, str(t))

            # Unzip the .nc file(s)
            with zipfile.ZipFile(filename) as zf:
                for fnat in zf.namelist():
                    if fnat.endswith('.nc'):
                        zf.extract(fnat, natfolder_t)

            # Read dataset
            t, ds = self.read_zarrDask_fast(natfolder_t, t)
            identifier = product._browse_properties['identifier']
            date_range = product._browse_properties['date']

            # Add metadata
            id_bytes = np.array(identifier, dtype='S143')
            t_start = self.str2unixTime(date_range.split('/')[0][:-1])
            t_end   = self.str2unixTime(date_range.split('/')[1][:-1])

            ds['identifier'] = xr.DataArray([id_bytes], dims=('time',), coords={'time': [0]})
            ds['unixTimeStart'] = xr.DataArray(t_start, dims=('time',), coords={'time': [0]})
            ds['unixTimeEnd'] = xr.DataArray(t_end, dims=('time',), coords={'time': [0]})
            ds = ds.assign_coords(time=[t])

            # Write to Zarr
            from utils import ZarrStore
            zarr_store = ZarrStore(self.output_dir , self.size)
            zarr_path , encoding = zarr_store.zarr_store_create(self.day, labl='VIS', num_files=len(self.file_list), varnms=self.channels, size=self.size) 

            ds.to_zarr(
                zarr_path,
                region={"time": slice(t, t + 1)},
                compute=True
            )

            # Clean up extracted files
            for fname in os.listdir(natfolder_t):
                file_path = os.path.join(natfolder_t, fname)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            os.rmdir(natfolder_t)

            read_pbar.update(1)
            download_queue.task_done()