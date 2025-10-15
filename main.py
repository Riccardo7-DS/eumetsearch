import os 
from dotenv import load_dotenv
from utils import products_list
from utils import EUMDownloader, bbox_mtg, init_logging, MTGDataParallel
import argparse
from datetime import datetime, timedelta
import calendar
from memory_profiler import profile

def main_batched(args, start_date, end_date, n_days=10):
    logger = init_logging("./logger_mtg_fci.log", verbose=False)
    load_dotenv()

    product_id = products_list["MTG-1"]["product_id"]
    bbox = bbox_mtg()
    W, S, E, N = bbox
    NSWE = [N, S, W, E]

    downloader = EUMDownloader(
        product_id=product_id,
        output_dir="./data/datastore_data",
        max_parallel_conns=10,
    )

    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    current = start

    while current <= end:
        # Define month boundaries
        year, month = current.year, current.month
        last_day = calendar.monthrange(year, month)[1]
        month_end = datetime(year, month, last_day, 23, 59, 59)
        if month_end > end:
            month_end = end

        label = f"mtg_{year}_{month:02d}"
        logger.info(f"Collecting data to data store {label}")

        batch_start = current
        while batch_start <= month_end:
            batch_end = min(batch_start + timedelta(days=n_days - 1, hours=23, minutes=59, seconds=59), month_end)
            logger.info(f"Downloading data for batch {batch_start.date()} → {batch_end.date()}")

            # 1️⃣ Download data for this batch
            downloader.download_interval(
                start_time=batch_start.isoformat(),
                end_time=batch_end.isoformat(),
                bounding_box=NSWE,
                observations_per_day=5,
                jump_minutes=60,
                start_hour=int(start.hour)
            )

            # 2️⃣ Process the downloaded data for this batch (month-level Zarr)
            # logger.info(f"Processing data for {label} ({batch_start.date()} → {batch_end.date()})")
            MTGDataParallel(
                args,
                downloader,
                area_reprojection="mtg_fci_latlon_1km",
                reprojection=args.resampler,
                chunks={"time": 1, "lat": 500, "lon": 500},
                label=label,
            )

            batch_start = batch_end + timedelta(seconds=1)

        logger.info(f"Finished processing for {label}")

        # Move to next month
        if month == 12:
            current = current.replace(year=year + 1, month=1, day=1)
        else:
            current = current.replace(month=month + 1, day=1)

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
    import dask
    import logging
    from functools import partial
    from utils import handle_exit_signal, aggregate_status_jsons
    from definitions import DATA_PATH
    from pathlib import Path
    import signal
    logger = logging.getLogger(__name__)

    argparser = argparse.ArgumentParser(description="MTG FCI Data Downloader")
    argparser.add_argument("--dask_threads", default=os.getenv("dask_threads", 0), type=int, help="Number of Dask threads (0 for single-threaded)")
    argparser.add_argument('-t', '--threading', action='store_false', help='Use threading for I/O operations')
    argparser.add_argument('-y', '--yes', action='store_true', help='Automatically confirm deletion of zarr')
    argparser.add_argument('-r', '--remove', action='store_true', help='Automatically confirm deletion of source files')
    argparser.add_argument("--resampler", default=os.getenv("resampler", "nearest"))
    args = argparser.parse_args()



    if args.yes:
        logger.warning(
        "⚠️  The '-y/--yes' flag is active. Existing Zarr datasets will be deleted automatically without confirmation!"
        )
        print("\nDeletion will start automatically in 3 seconds. Press Ctrl+C to abort.")
        try:
            for i in range(3, 0, -1):
                print(f"⏳ {i}...", end='', flush=True)
                time.sleep(1)
                print('\r', end='')
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            exit(1)

    
    aggregated_json_filename = Path(DATA_PATH) / "datastore_data" / "aggregated_data.json"

        # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, partial(handle_exit_signal, aggregated_file=aggregated_json_filename))
    signal.signal(signal.SIGTERM, partial(handle_exit_signal, aggregated_file=aggregated_json_filename))

    if not os.path.exists(aggregated_json_filename):
        aggregate_status_jsons(aggregated_json_filename)

    if args.dask_threads > 0:
        logger.info(f"Using {args.dask_threads} dask threads")
        dask.config.set(scheduler="threads", num_workers=args.dask_threads)    
    else:
        logger.info("Using single dask scheduler")
        dask.config.set(scheduler="single-threaded")

    start_date = "2025-06-01T09:00:00"
    end_date = "2025-10-13T09:30:00"

    try:
        monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        monitor_thread.start()
        # with cProfile.Profile() as pr:
        main_batched(args, start_date, end_date, n_days=1)
        # stats = pstats.Stats(pr)
        # stats.sort_stats("cumtime").print_stats(20)  # to
    except Exception as e:
        logger.error(f"Error {e}")
        logger.error(traceback.format_exc())
        aggregate_status_jsons(aggregated_json_filename)
        raise
    finally:
        # Always aggregate at the end — even if no error
        aggregate_status_jsons(aggregated_json_filename)


