import os 
from dotenv import load_dotenv
from utils import products_list
from utils import EUMDownloader, bbox_mtg, init_logging, MTGDataParallel
import argparse
from datetime import datetime
import calendar
from memory_profiler import profile

def main_monthly(args, start_date, end_date):
    logger = init_logging("./logger_mtg_fci.log", verbose=False)
    cpus = os.cpu_count()
    
    # Load environment variables from .env file
    load_dotenv()
    
    product_id = products_list["MTG-1"]["product_id"]
    
    # Define bounding box
    bbox = bbox_mtg()
    W, S, E, N = bbox[0], bbox[1], bbox[2], bbox[3]
    NSWE = [N, S, W, E]
    
    # Initialize downloader once
    downloader = EUMDownloader(
        product_id=product_id, 
        output_dir="./data/datastore_data",
        max_parallel_conns=10,
    )
    
    # Convert start/end dates to datetime objects
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    current = start

    while current <= end:
        # Compute last day of current month
        last_day = calendar.monthrange(current.year, current.month)[1]
        month_end = current.replace(day=last_day, hour=23, minute=59, second=59)
        if month_end > end:
            month_end = end
        
        # Download data for this month
        logger.info(f"Downloading data for {current.strftime('%Y-%m')}")
        downloader.download_interval(
            start_time=current.isoformat(), 
            end_time=month_end.isoformat(),
            bounding_box=NSWE, 
            observations_per_day=5,
            jump_minutes=60,
            start_hour=int(start.hour)
        )
        
        # Create MTGDataParallel object with a unique label for the month
        label = f"mtg_{current.year}_{current.month:02d}"
        logger.info(f"Processing MTGDataParallel for {label}")
        MTGDataParallel(
            args, 
            downloader, 
            area_reprojection="mtg_fci_latlon_1km",
            reprojection=args.resampler,
            chunks={"time": 10, "lat": 500, "lon": 500},
            label=label
        )
        
        # Move to first day of next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1, day=1)
        else:
            current = current.replace(month=current.month + 1, day=1)

@profile
def main(args, start_date, end_date):
    logger = init_logging("./logger_mtg_fci.log", verbose=False)
    cpus = os.cpu_count()
    
    # Load environment variables from .env file
    load_dotenv()
    
    product_id = products_list["MTG-1"]["product_id"]
    
    bbox = bbox_mtg()
    W,S,E,N = bbox[0], bbox[1], bbox[2], bbox[3]
    NSWE = [N, S, W, E]
    
    downloader = EUMDownloader(
        product_id=product_id, 
        output_dir="./data/datastore_data",
        max_parallel_conns=10,
    )
    
    downloader.download_interval(
        start_time=start_date, 
        end_time=end_date,
        bounding_box=NSWE, 
        observations_per_day=6,
        jump_minutes=60
    )
    
    MTGDataParallel(args, 
        downloader, 
        area_reprojection="mtg_fci_latlon_1km",
        reprojection=args.resampler,
        chunks={"time":10, "lat":500, "lon":500},
        processes=8
    )


def monitor_resources(interval=1.0, log_file="resource_log.txt"):
    pid = os.getpid()
    process = psutil.Process(pid)
    with open(log_file, "w") as f:
        f.write("time,cpu_percent,mem_gb\n")
        while True:
            try:
                cpu = process.cpu_percent(interval=interval)
                mem = process.memory_info().rss / 1024**3
                f.write(f"{time.strftime('%H:%M:%S')},{cpu:.2f},{mem:.3f}\n")
                f.flush()
            except Exception:
                break

if __name__ == "__main__":


    import psutil
    import time
    import threading, traceback
    import logging
    logger = logging.getLogger(__name__)

    argparser = argparse.ArgumentParser(description="MTG FCI Data Downloader")
    argparser.add_argument('-t', '--threading', action='store_false', help='Use threading for I/O operations')
    argparser.add_argument('-y', '--yes', action='store_true', help='Automatically confirm deletion of zarr')
    argparser.add_argument('-r', '--remove', action='store_true', help='Automatically confirm deletion of source files')
    argparser.add_argument("--resampler", default=os.getenv("resampler", "nearest"))
    args = argparser.parse_args()
    
    start_date = "2025-05-01T09:00:00"
    end_date = "2025-05-01T09:30:00"

    try:
        monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        monitor_thread.start()
        # with cProfile.Profile() as pr:
        main_monthly(args, start_date, end_date)
        # stats = pstats.Stats(pr)
        # stats.sort_stats("cumtime").print_stats(20)  # to
    except Exception as e:
        logger.error(f"Error {e}")
        logger.error(traceback.format_exc())
        raise


