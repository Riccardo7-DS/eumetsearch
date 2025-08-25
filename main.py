import os 
from dotenv import load_dotenv
from utils import products_list
from utils import EUMDownloader, bbox_mtg, init_logging, MTGDataParallel
import argparse
import cProfile
import pstats
import numpy as np
import logging 

logger = logging.getLogger(__name__)


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
        observations_per_day=1,
        jump_minutes=60
    )
    
    MTGDataParallel(args, 
        downloader, 
        area_reprojection="mtg_fci_latlon_1km",
        reprojection=args.resampler,
        chunks={"time":1, "lat":100, "lon":100}
    )

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="MTG FCI Data Downloader")
    argparser.add_argument('-y', '--yes', action='store_true', help='Automatically confirm deletion')
    argparser.add_argument("--remove", default=os.getenv("remove", False))
    argparser.add_argument("--resampler", default=os.getenv("resampler", "nearest"))
    args = argparser.parse_args()         
    
    start_date = "2025-05-02T00:00:00"
    end_date = "2025-05-03T00:00:00"

    try:
        # with cProfile.Profile() as pr:
        main(args, start_date, end_date)
        # stats = pstats.Stats(pr)
        # stats.sort_stats("cumtime").print_stats(20)  # to
    except Exception as e:
        logger.error(e)
        raise

