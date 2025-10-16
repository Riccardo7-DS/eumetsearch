import eumdac
import subprocess
from datetime import timedelta, datetime, timezone
from eumdac.collection import SearchResults
from eumdac.tailor_models import  RegionOfInterest
from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path
import time
import fnmatch
import eumdac
from typing import Literal
import netCDF4
import h5py
import multiprocessing as mp
import os
from dotenv import load_dotenv
import zipfile
import shutil
from tqdm.auto import tqdm
from pydantic import validate_call, PositiveInt
import logging
import queue
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from typing import Union
from http.client import IncompleteRead
import json 
from urllib3.exceptions import ProtocolError
import traceback
from threading import Thread, Lock
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
    """Downloader for EUMETSAT products from DataStore or DataTailor
        args:
            product_id: EUMETSAT product ID
            output_dir: directory to save the downloaded files
            format: file format to download (netcdf4 or geotiff)
            sleep_time: time to wait between status checks (in seconds)
            max_parallel_conns: maximum number of parallel connections"""
    

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

        return intervals
    
    def log_intervals(self):
        for i, (start, end) in enumerate(self.intervals, 1):
            logger.info(f"{i:03d}. Start: {start:%Y-%m-%d %H:%M:%S} -> End: {end:%Y-%m-%d %H:%M:%S}")

    
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

        with ThreadPoolExecutor(max_workers=self.max_parallel_conns) as executor:
            futures = [executor.submit(self._download_products, chunk, self._selected_collection, self.output_dir, bounding_box)
                       for chunk in chunks]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Download failed with error: {e}")
        

    def download_interval(self, 
                          start_time, 
                          end_time, 
                          bounding_box:list=None, 
                          method:Literal[None, "datatailor", "datastore"]= None,
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
        self._method = method
        self._bounding_box = bounding_box

    def initiate_download(self, aggregated_file = None):

        if os.path.isfile(aggregated_file):
            from utils import JsonDataResponse
            collected_data = JsonDataResponse(aggregated_file) 
            self.intervals = collected_data.files_exclusion(self.intervals)

        self.log_intervals()
        self.file_list = self._collect_products(self.intervals, self._bounding_box)

        if self._method == "datatailor":
            self._datatailor_download(
                self.file_list,
                bounding_box=self._bounding_box,
            )

        elif self._method == "datastore":
            self.chunks_download(
                self.file_list, 
                self._bounding_box
            )
            
            
    def _datastore_thread_download(self, intervals, bounding_box=None):
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


class ZarrExport():
    def __init__(self,
                args:dict,  
                downloader: EUMDownloader,
                label:str, 
                channels:list= ['vis_06',  'vis_08'],
                area_reprojection:Union[None, str]=None,
                reprojection="nearest",
                processes:PositiveInt=4,
                initialize_dataset:bool=False,
                chunks: dict = {"time": 1, "lon": "auto", "lat": "auto"},
                custom_size: dict | None = None
                ):
        
        """Export downloaded files to Zarr format.

        Args:
            args: command line arguments
            downloader: EUMDownloader object with downloaded files
            label: label for the Zarr file 
            channels: list of channels to include in the Zarr file
            area_reprojection: area to reproject to (e.g., "mtg_fci_latlon_1km" areas in areas.yaml file)
            reprojection: reprojection method from pyresample (e.g., "nearest", "bilinear", "cubic")
            processes: number of parallel processes
            initialize_dataset: whether to initialize the dataset with the first file
            chunks: chunk sizes for the Zarr file
            custom_size: custom size for the Zarr file (e.g., {"time": 60} for 60 time steps)
            
        """
        
        
        
        from utils import compute_auto_chunks
        from definitions import DATA_PATH


        self.output_dir = downloader.output_dir
        self.size = self._get_size(area_reprojection)
        self._reproject = area_reprojection
        self.processes = processes
        input_shape = {"time": chunks["time"], "lat": self.size[0], "lon":self.size[1]}
        self.chunks = compute_auto_chunks(shape= input_shape)
        self._reprojection = reprojection
        self._threading = args.threading

        channelsIR = ['ir_105', 'ir_123',  'ir_133',  'ir_38',  'ir_87',  'ir_97',  'wv_63',  'wv_73']    
        channelsVIS= ['nir_13', 'nir_16',  'nir_22',  'vis_04',  'vis_05', 'vis_06',  'vis_08',  'vis_09', ]
        
        assert all(f in channelsIR or f in channelsVIS for f in channels), \
           "One or more channels are not in channelsIR or channelsVIS"
        
        self.channels = channels
        self.zip_path = Path(self.output_dir) / "zipfolder"
        os.makedirs(self.zip_path, exist_ok=True)
        self.nat_path = Path(self.output_dir) / "natfolder"
        os.makedirs(self.nat_path, exist_ok=True)

        self.zarr_lock = Lock()
        self.nectdf_lock = Lock()

        downloader.initiate_download(aggregated_file=Path(DATA_PATH) / "datastore_data"/ "aggregated_data.json")

        self.file_list = downloader.file_list
        logger.info("The initialized downloader contains {} files, proceeding with export to zarr...".format(len(self.file_list)))
        self.download_to_zarr(args, self.file_list, initialize_dataset, label, custom_size)

    def _get_size(self, area_reprojection:str):
        if area_reprojection == "worldeqc3km":
            return [2048, 4096]
        elif area_reprojection == "worldeqc3km70":
            return [4096, 8192]
        elif area_reprojection == "worldeqc1km70":
            return [15585, 40075]
        elif area_reprojection == "mtg_fci_latlon_1km":
            return [14000, 14000]
        elif area_reprojection == "EPSG_4326_36000x18000":
            return [18000, 36000]
        else: 
            raise NotImplementedError("Area {} not implemented".format(area_reprojection))
        

    def _mark_done(self, task_id: str):
        if self.status_file.exists():
            status = json.loads(self.status_file.read_text())
        else:
            status = {}
        status[task_id] = "done"
        self.status_file.write_text(json.dumps(status, indent=2))


    def download_to_zarr(self, args, file_list: list, initialize_dataset: bool, label: str, custom_size: dict | None = None):
        from utils import ZarrStore
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        import numpy as np
        from pathlib import Path
        import logging

        logger = logging.getLogger(__name__)
        t0 = time.time()

        # --- Sort files by date
        file_list = sorted(
            self.file_list,
            key=lambda p: np.datetime64(p._browse_properties['date'].split('/')[0][0:-1])
        )

        # --- Initialize dataset if needed
        if initialize_dataset:
            t = 0
            filename = self._download_file(product=self.file_list[t], t=t)
            example_ds = self.read_convert_append_catch(filename=filename, t=t)

            example_ds = example_ds.assign_attrs(
                chunks=self.chunks,
                origin_size=self.size
            )

            example_ds.attrs['area_definition'] = {
                'area_id': self._area_def.area_id,
                'description': self._area_def.description,
                'proj_id': self._area_def.proj_id,
                'projection': self._area_def.proj_dict,
                'width': self._area_def.width,
                'height': self._area_def.height,
                'area_extent': self._area_def.area_extent,
            }
        else:
            example_ds = None

        # --- Initialize Zarr store
        store = ZarrStore(
            self.output_dir,
            size=self.size,
            file_list=self.file_list,
            channels=self.channels,
            chunks=self.chunks,
            label=label,
            ds=example_ds,
            custom_size=custom_size,
            remove_flag=args.yes
        )

        self.status_file = Path(store.path).with_suffix(".status.json")

        if not args.remove:
            self._remove_all_tempfiles()

        # --- Parallel downloads
        logger.info("Starting parallel downloads...")
        download_pbar = tqdm(total=len(file_list), desc="Downloading files", position=0, leave=True)

        filenames = [None] * len(file_list)
        with ThreadPoolExecutor(max_workers=self.processes) as executor:
            futures = {
                executor.submit(self._download_file, self.file_list[t], t, None): t
                for t in range(len(file_list))
            }
            for future in as_completed(futures):
                t = futures[future]
                try:
                    filenames[t] = future.result()
                    download_pbar.update(1)
                except Exception as e:
                    logger.error(f"Error downloading file {t}: {e}")
        download_pbar.close()

        logger.info("All downloads completed. Starting processing phase...")

        # --- Sequential reading/conversion after downloads complete
        read_pbar = tqdm(total=len(file_list), desc="Reading files", position=1, leave=True)
        for t, filename in enumerate(filenames):
            if filename is None:
                logger.warning(f"Skipping missing file {t}.")
                continue
            try:
                self.read_convert_append_catch(filename=filename, t=t, zarr_path=store.path)
                read_pbar.update(1)
            except Exception as e:
                logger.error(f"Error reading/processing file {filename}: {e}")
        read_pbar.close()

        # --- Done
        elapsed_seconds = time.time() - t0
        hours = int(elapsed_seconds // 3600)
        minutes = int((elapsed_seconds % 3600) // 60)
        logger.info(f"✅ Done in {hours} hours {minutes} minutes.")

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
    

    def _is_netcdf_valid(self, filepath):
        """Check if a NetCDF/HDF5 file can be opened."""
        try:
            with netCDF4.Dataset(filepath, "r"):
                return True
        except Exception as e1:
            try:
                with h5py.File(filepath, "r"):
                    return True
            except Exception as e2:
                logger.debug(f"File check failed for {filepath}: {e1} | {e2}")
                return False
            
    
    def _download_file(self, product, t, download_queue= None):
        """Download a single file and put it in the queue if provided.
        
        Args:
            product: product object to download
            t: time index
            download_queue: queue to put the result in (optional)
        Returns:
            filename if download_queue is None
        """
                 
        filename = self._download_zipfile(product, self.zip_path)

        if download_queue is not None:
            download_queue.put((filename, t, product))
        else:
            return filename

    def _download_zipfile(self, product, dest_folder):
        dsnm=product.metadata["properties"]["title"]
        dssz=product.metadata["properties"]["productInformation"]["size"]
        outfilename = os.path.join(dest_folder, dsnm) +'.zip'

        if os.path.isfile(outfilename):
            szdsk=os.path.getsize(outfilename)
            if szdsk/1000 > dssz:
                return outfilename

        return self._safe_download(product=product, dest_folder=dest_folder)
    

    def _get_coords_area(self):
            """Create an AreaDefinition for the dataset."""
            lons, lats = self._area_def.get_lonlats()
            return lons, lats
    
    def _safe_download(self, product, dest_folder, max_retries=5, backoff=5):
        """
        Download a zip product with retries in case of IncompleteRead or connection issues.

        Args:
            product: object with .open() method returning a file-like object
            dest_folder: folder to write into
            max_retries: maximum retry attempts
            backoff: seconds to wait between retries (increases linearly)

        Returns:
            str: path to downloaded file
        """
        attempt = 0
        dest_path = None

        while attempt < max_retries:
            try:
                with product.open() as fsrc, \
                     open(os.path.join(dest_folder, os.path.basename(fsrc.name)), mode="wb") as fdst:
                    shutil.copyfileobj(fsrc, fdst)

                dest_path = os.path.join(dest_folder, os.path.basename(fsrc.name))
                return dest_path

            except (IncompleteRead, OSError, ProtocolError) as e:
                attempt += 1
                logger.info(f"⚠️ Download failed (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    time.sleep(backoff * attempt)  # exponential-ish backoff
                else:
                    logger.info(f"Download failed after {max_retries} attempts: {e}")
                    return None

        return dest_path
    
    def warp_geostationary_to_regular_grid(self, src_array, area_src, target_width, target_height):
       
        """
        Reproject a geostationary source array (pyresample area) to a regular EPSG:4326 lat/lon grid.    
        Parameters:
            src_array: np.ndarray
                Source array of shape (ny, nx)
            area_src: pyresample AreaDefinition
                Source geostationary area
            target_width: int
                Width of target grid
            target_height: int
                Height of target grid    
        Returns:
            warped_array: np.ndarray
                Reprojected array in EPSG:4326
        """    
        from osgeo import gdal
        from pyproj import CRS, Transformer
        ny, nx = src_array.shape    
        # ---- Create in-memory GDAL source dataset ----
        src_ds = gdal.GetDriverByName("MEM").Create("", nx, ny, 1, gdal.GDT_Float32)
        src_ds.GetRasterBand(1).WriteArray(src_array)
        src_ds.GetRasterBand(1).FlushCache()    

        # ---- Set geotransform ----
        min_x, max_y, max_x, min_y = area_src.area_extent
        pixel_width = (max_x - min_x) / area_src.width
        pixel_height = (max_y - min_y) / area_src.height
        geo_transform_src = (min_x, pixel_width, 0, max_y, 0, -pixel_height)
        src_ds.SetGeoTransform(geo_transform_src)    
        # area_src.proj_dict is your geostationary dict
        crs_src = CRS.from_dict(area_src.proj_dict)    
        # Convert to WKT for GDAL
        wkt_src = crs_src.to_wkt()
        src_ds.SetProjection(wkt_src)    
        # Assign to your GDAL MEM dataset    
        transformer = Transformer.from_crs(crs_src, CRS.from_epsg(4326), always_xy=True
        )    
        # Source corners
        lon_min, lat_max = transformer.transform(min_x, max_y)
        lon_max, lat_min = transformer.transform(max_x, min_y)
        geo_bounds = (lon_min, lat_min, lon_max, lat_max)    
        if not isinstance(geo_bounds, (list, tuple)) or len(geo_bounds) != 4:
             raise ValueError("geo_bounds must be a 4-element tuple (minX, minY, maxX, maxY)")    
        
        # ---- GDAL Warp to target grid ----
        warp_opts = gdal.WarpOptions(
            format="MEM",
            outputBounds=geo_bounds,  # in degrees
            width=target_width,
            height=target_height,
            dstSRS="EPSG:4326",
            resampleAlg="bilinear",
            multithread=False
        )    
        dst_ds = gdal.Warp(destNameOrDestDS=None, srcDSOrSrcDSTab=src_ds, options=warp_opts)
        warped_array = dst_ds.GetRasterBand(1).ReadAsArray()    
        return warped_array
    
    def _resample_pyresample(self, scn):
        from pyresample import create_area_def
        from pyproj import CRS
        from utils import extract_custom_area

        area_def = extract_custom_area(self._reproject, "./src/utils/areas.yaml")
        return scn.resample(area_def, radius_of_influence=5000, resampler=self._reprojection)
    
    def _dataset_reproject_loop(self, scn):
        out = {}
        
        if self._reproject:
            scn_resampled = self._resample_pyresample(scn)
  
        # -------------------------------------------------------------
        # Get timestamp from scene
        # -------------------------------------------------------------
        t_value = scn[self.channels[0]].attrs["time_parameters"]["nominal_start_time"]
        t_value = self.str2unixTime(t_value)

        for channel in self.channels:
            logger.info(f"Reprojecting channel {channel}...")
            arr = scn_resampled[channel].values
            # ---- Convert to xarray ----
            data_array = xr.DataArray(
                arr[np.newaxis, :, :],
                dims=("time", "lat", "lon"),
                coords={"time": [t_value]},
                name=channel,
                attrs=scn[channel].attrs
            )
            out[channel] = data_array

        if self._reproject:
            logger.info("Reprojection of all channels complete.")

        ds = xr.Dataset(out)
        ds = self._clean_metadata(ds)

        return ds, t_value


    def _read_satpy_convert(self, natfolder, t=None):
        import warnings, numpy as np
        from satpy.scene import Scene
        from satpy import find_files_and_readers
        from utils import single_thread_env

        # -------------------------------------------------------------
        # Load Satpy scene
        # -------------------------------------------------------------
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

        calibration = "reflectance"
        files = find_files_and_readers(base_dir=natfolder, reader="fci_l1c_nc", missing_ok=True)
        scn = Scene(filenames=files)
        scn.load(self.channels, calibration=calibration)

        # -------------------------------------------------------------
        # Thread-safe environment for pyproj (no multithreading issues)
        # -------------------------------------------------------------
        if not self._threading:
            with single_thread_env():
                logger.info("Entering thread-safe reprojection mode...")

                ds, t_value = self._dataset_reproject_loop(scn)

        else:
            ds, t_value = self._dataset_reproject_loop(scn)

        return t_value, ds


    def str2unixTime(self, stime) -> np.datetime64:

        if isinstance(stime, np.datetime64):
            return stime  # already the right type
        elif isinstance(stime, datetime):
            return np.datetime64(stime, "ns")
        elif isinstance(stime, str):
            try:
                return np.datetime64(stime, "ns")
            except Exception as e:
                raise ValueError(f"Failed to parse datetime string {stime!r}: {e}")
        else:
            raise ValueError(f"Unsupported type for str2unixTime: {type(stime)} ({stime!r})")

    
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
    
    def _extract_datetime(self, s, convert_datetime:bool=False):
        import re
        matches = re.findall(r"\d{14}", s)
        if matches:
            if convert_datetime:
                return datetime.strptime(matches[1], "%Y%m%d%H%M%S")   
            else:
                return matches[1]

    def read_convert_append_catch(self, download_queue=None, read_pbar=None, zarr_path=None, filename=None, t=None):
        """
        Read, convert, and append data to Zarr store.
        Segfault-safe: if a segmentation fault occurs inside pyresample/Satpy,
        the function continues with the next file gracefully.
        """

        def _process_in_subprocess(filename, t, product):
            """Run the file processing logic in an isolated subprocess."""
            q = mp.Queue()
            p = mp.Process(target=self._safe_runner, args=(q, self._process_single_file, filename, t, product, read_pbar, zarr_path))
            p.start()
            p.join()

            if p.exitcode == 0:
                if not q.empty():
                    status, data = q.get()
                    if status == "success":
                        return data
                    else:
                        logger.error(f"Exception during processing:\n{data}")
                        return None
                else:
                    logger.warning("No output received from subprocess.")
                    return None
            elif p.exitcode == -11:  # segmentation fault
                logger.warning(f"Segmentation fault while processing {filename}, skipping it.")
                return None
            else:
                logger.warning(f"Subprocess for {filename} exited with code {p.exitcode}.")
                return None

        # --- Single-file mode ---
        if download_queue is None:
            if filename is None:
                raise ValueError("Filename must be provided when not using a download queue.")
            if t is None:
                raise ValueError("Time index 't' must be provided when not using a download queue.")

            product = self.file_list[t]
            ds = _process_in_subprocess(filename, t, product)
            return ds

        # --- Queue-based threaded mode ---
        while True:
            item = download_queue.get()
            try:
                if item[0] is None:  # Sentinel value to stop thread
                    break
                filename, t, product = item[:3]
                _process_in_subprocess(filename, t, product)
            finally:
                download_queue.task_done()
                if read_pbar:
                    read_pbar.update(1) 
    
    def read_convert_append(self, download_queue=None, read_pbar=None, zarr_path=None, filename=None,  t=None):
        """
        Read, convert, and append data to Zarr store.
        If download_queue is provided, it will read from the queue in a loop.
        Otherwise, it processes a single file specified by filename and t.
        
        Parameters:
        - download_queue: Queue for threaded processing (optional)
        - read_pbar: Progress bar for reading (optional)
        - zarr_path: Path to Zarr store (optional)
        - filename: Filename to process (required if download_queue is None)
        - t: Time index (required if download_queue is None)
                                                                
        """


        if download_queue is None:
            if filename is None:
                raise ValueError("Filename must be provided when not using a download queue.")
            if t is None:
                raise ValueError("Time index 't' must be provided when not using a download queue.")
            product = self.file_list[t]
            ds = self._process_single_file(filename, t, product)
            return ds

        # Queue-based threaded processing
        while True:
            item = download_queue.get()
            try:
                if item[0] is None:  # Sentinel to stop thread
                    break
                filename, t, product = item[:3]
                self._process_single_file(filename, t, product, read_pbar, zarr_path)
            finally:
                download_queue.task_done()

    def _safe_runner(self, q, func, *args, **kwargs):
        """Run a function and communicate results or exceptions back to parent."""
        try:
            result = func(*args, **kwargs)
            q.put(("success", result))
        except Exception:
            q.put(("error", traceback.format_exc()))

    def _extract_netcdf_files(self, filename, natfolder_t, remove_zip:bool=True):
        with zipfile.ZipFile(filename) as zf:
            for fnat in zf.namelist():
                if fnat.endswith('.nc'):
                    zf.extract(fnat, natfolder_t)
        if remove_zip:
            os.remove(filename)


    def _safe_write_to_zarr(self, ds_new, zarr_path, t, lock):
        """
        Safely writes ds_new to zarr_path at time index t, skipping if already filled.

        Parameters
        ----------
        ds_new : xarray.Dataset
            Dataset to write for a single time index (shape [time=1, ...]).
        zarr_path : str
            Path to existing zarr store.
        t : int
            Time index to write to.
        lock : threading.Lock or multiprocessing.Lock
            Lock to ensure safe concurrent writes.
        main_var : str
            Name of the primary data variable to inspect or log (e.g., "ir_105").
        """
        with lock:
            # Open lazily (fast)
            store = xr.open_zarr(zarr_path, consolidated=False)

            # Load the filled_flag lazily, then to NumPy
            filled = store["filled_flag"].load().values

            # Find first unfilled index
            unfilled_indices = np.where(~filled)[0]

            if len(unfilled_indices) == 0:
                logger.info("[done] All timesteps already filled.")
                return None

            t = int(unfilled_indices[0])

            # Write data slice
            ds_new.to_zarr(
                zarr_path,
                region={"time": slice(t, t + 1)},
                compute=True
            )

            # Update flag to True
            flag_update = xr.Dataset(
                {"filled_flag": (("time",), np.array([True], dtype=bool))},
                coords={"time": [store.time.isel(time=t).item()]}
            )
            flag_update.to_zarr(
                zarr_path,
                region={"time": slice(t, t + 1)},
                compute=True
            )

            return t  # return which index was written for logging/debug



    def _process_single_file(self, filename, t, product, read_pbar=None, zarr_path=None):
        """Process a single file: extract, read, convert, and append to Zarr or return dataset.
        
        Parameters:
        - filename: Path to the zip file
        - t: Time index
        - product: Product object for metadata
        - read_pbar: Progress bar for reading (optional)
        - zarr_path: Path to Zarr store (optional)
        """

        from utils import debug_time_vars

        # file_n = self._extract_datetime(filename)
        natfolder_t = os.path.join(self.nat_path, str(t))

        os.makedirs(natfolder_t, exist_ok=True)

        if len(os.listdir(natfolder_t)) == 0:
            self._extract_netcdf_files(filename, natfolder_t)

        # Read dataset
        t_value, ds = self._read_satpy_convert(natfolder_t, t)
        identifier = product._browse_properties['identifier']
        date_range = product._browse_properties['date']

        # Add metadata
        id_bytes = np.array(identifier, dtype='S143')
        t_start = self.str2unixTime(date_range.split('/')[0][:-1])
        t_end   = self.str2unixTime(date_range.split('/')[1][:-1])

        ds['identifier'] = xr.DataArray(
            [id_bytes], dims=['time'], coords={"time": ds.time}
        )

        # Ensure all variables share the same time coordinate
        ds["timeStart"] = xr.DataArray(
            [t_start], dims=["time"], coords={"time": ds.time}
        )
        ds["timeEnd"] = xr.DataArray(
            [t_end], dims=["time"], coords={"time": ds.time}
        )

        # Drop lat/lon if present (they are in the area definition)
        if "lat" in ds and "lon" in ds:
            ds = ds.drop_vars(["lat", "lon"])

        debug_time_vars(ds)
        
        if zarr_path is not None:
            self._safe_write_to_zarr(ds, 
                zarr_path, 
                t, 
                self.nectdf_lock
            )
            
            read_pbar.update(1)
            task_id = f"{t_start}_{t}"
            self._mark_done(task_id)
            # try:
            #     # shutil.rmtree(natfolder_t)
            # except FileNotFoundError as e:
            #     logger.warning(f"Could not remove temporary folder {natfolder_t}: {e}")
            #     pass
        else:
            return ds
