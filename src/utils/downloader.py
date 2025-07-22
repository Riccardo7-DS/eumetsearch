import eumdac
import datetime
import subprocess
from datetime import timedelta
from datetime import datetime
import time
import fnmatch
import eumdac
from typing import Literal
import os
from dotenv import load_dotenv
import shutil
from tqdm.auto import tqdm
import logging
logger = logging.getLogger(__name__)

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

    def __init__(self, 
                 product_id:str, 
                 output_dir:str,
                 format:Literal["netcdf4","geotiff"]='netcdf4',
                 sleep_time:int=10,
                 MAX_PARALLEL_CONNS=10):

        consumer_key = os.getenv("EUMETSAT_CONSUMER_KEY")
        consumer_secret = os.getenv("EUMETSAT_CONSUMER_SECRET")

        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.SLEEP_TIME = sleep_time # seconds
        self.MAX_PARALLEL_CONNS = MAX_PARALLEL_CONNS

        if format not in ["netcdf4", "geotiff"]:
            raise ValueError("Format must be either 'netcdf4' or 'geotiff'.")
        self.format = format

        self.product_id = product_id
        produc_fake_name = [k for k, v in products_list.items() if v["product_id"] == product_id][0]
        self.product_name = products_list[produc_fake_name]["product_name"]
        self.channels = products_list[produc_fake_name]["bands"]
        self.token = eumdac.AccessToken((consumer_key, consumer_secret))
        logger.info(f"This token '{self.token}' expires {self.token.expiration}")

    def initialize_datastore(self):
        datastore = eumdac.DataStore(self.token)
        logger.info("Datastore initialized.")
        return datastore
    
    def initialize_datatailor(self):
        datatailor = eumdac.DataTailor(self.token)
        logger.info("DataTailor initialized.")
        return datatailor

    def _download_products(self, interval_list, coll_obj, dest_folder):
        start = interval_list[0]
        end = interval_list[1]
        products = coll_obj.search(dtstart=start, dtend=end)
        for product in products:
            with product.open() as fsrc, open(os.path.join(dest_folder, fsrc.name), mode='wb') as fdst:
                shutil.copyfileobj(fsrc, fdst)

    def split_time_into_daily_observations(self, 
                                       start_date, 
                                       end_date, 
                                       observations_per_day=1, 
                                       start_hour=10, 
                                       interval_hours=0.167):
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
            for i in range(observations_per_day):
                interval_start = datetime.combine(current_date, datetime.min.time()) + timedelta(hours=start_hour + i * interval_hours)
                interval_end = interval_start + timedelta(hours=interval_hours)
                intervals.append((interval_start, interval_end))

            current_date += timedelta(days=1)

        for i, (start, end) in enumerate(intervals, 1):
            logger.info(f"{i:03d}. Start: {start.strftime('%Y-%m-%d %H:%M:%S')}  →  End: {end.strftime('%Y-%m-%d %H:%M:%S')}")

        return intervals
    
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

    def download_interval(self, 
                          start_time, 
                          end_time, roi:str=None, 
                          bounding_box:list=None, 
                          method:Literal["datatailor", "datastore"]="datatailor",
                          observations_per_day=1, 
                          start_hour=12, 
                          interval_hours=1
        ):

        datastore = self.initialize_datastore()
        """Download products from the datastore."""
        logger.info(f"Downloading {self.product_id} from {start_time} to {end_time}...")
        self.selected_collection = datastore.get_collection(self.product_id)

                # Set sensing start and end time
        start = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%SZ")
        end = datetime.strptime(end_time, "%Y-%m-%dT%H:%M:%SZ")
        if start >= end:
            raise ValueError("Start time must be before end time.")

        #Ensure start_time and end_time are in the correct format
        if not isinstance(start, datetime) or not isinstance(end, datetime):
            raise ValueError("Start and end times must be datetime objects.")

        intervals = self.split_time_into_daily_observations(
            start_date=start,
            end_date=end,
            observations_per_day=observations_per_day,
            start_hour=start_hour,
            interval_hours=interval_hours
        )

        if method == "datatailor":
            self._datatailor_download(
                intervals,
                bounding_box=bounding_box,
                observations_per_day=1
            )

        elif method == "datastore":
            self._datastore_thread_download(
                intervals,
                bounding_box=bounding_box)
            
            
    def _datastore_thread_download(self,
                            intervals, 
                            bounding_box=None):
        from concurrent.futures import ThreadPoolExecutor
        
        def download_products_in_thread(interval_list, coll_obj=self.selected_collection , destination_folder=self.output_dir):
            start = interval_list[0][0]
            end = interval_list[0][1]
            products = coll_obj.search(dtstart=start, dtend=end, bbox=bounding_box)
            for product in products:
                with product.open() as fsrc, open(os.path.join(destination_folder, fsrc.name), mode='wb') as fdst:
                    shutil.copyfileobj(fsrc, fdst)
        
        """Download products from the Datastore service."""
        intervals_thread = self._split_intervals_for_threading(intervals, self.MAX_PARALLEL_CONNS)

        with ThreadPoolExecutor(max_workers = self.MAX_PARALLEL_CONNS) as executor:
            executor.map(download_products_in_thread, intervals_thread)




    def _datatailor_download(self,
                    intervals, 
                    bounding_box=None):
        
        """Download products using the DataTailor service."""
        
        from eumdac.tailor_models import Chain
        datatailor = self.initialize_datatailor()

        # Define the chain configuration
        chain = Chain(
            product=self.product_name,
            format=self.format,
            filter={"bands" : self.channels},
            projection='geographic',
        )
        
        #loop over intervals and run the chain for each interval
        for interval in tqdm(intervals):
            start, end = interval

            # Retrieve latest product that matches the filter
            selected_products = self.selected_collection.search(
                dtstart=start,
                dtend=end,
                bbox=bounding_box,)
            
            if selected_products.total_results > 1:
                logger.debug(f'Found Multiple Datasets: {selected_products.total_results} for the given time range')

            customisation = datatailor.new_customisation(selected_products.first(), chain=chain)
            self._download_customisation(customisation, self.output_dir)

    
    def _download_customisation(self, customisation, dest_path):
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
    
    # def download_product(self, selected_collection, product):
    #     selected_product = self.datastore.get_product(
    #         product_id=product,
    #         collection_id=selected_collection)
    
    #     logger.info(f'Start downloading of product {selected_product}.')
        
    #     try:
    #         with selected_product.open() as fsrc, \
    #                 open(fsrc.name, mode='wb') as fdst:
    #             shutil.copyfileobj(fsrc, fdst)
    #         logger.info(f'Download of product {selected_product} finished.')
    #     except:
    #         logger.warning("Download failed")




    def _download_multiprocessing(self, chain_config, intervals, product):
        """Run the chain configuration for each interval in parallel."""
        input = str(product) + ".zip"
        output = str(product)
        if not os.path.exists(output):
            os.makedirs(output)
        for start, end in intervals:
            subprocess.run(["epct", "run-chain", "-f", chain_config, "--sensing-start", start, "--sensing-stop", end, input, "-o", output])


