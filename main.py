import os 
from dotenv import load_dotenv
from utils import products_list
from utils import EUMDownloader, bbox_mtg, init_logging, MTGDataParallel
import argparse

argparser = argparse.ArgumentParser(description="MTG FCI Data Downloader")
argparser.add_argument('-y', '--yes', action='store_true', help='Automatically confirm deletion')

args = argparser.parse_args()                       
logger = init_logging()
cpus = os.cpu_count()

# Load environment variables from .env file
load_dotenv()

product_id = products_list["MTG-1"]["product_id"]

start_date = "2025-05-02T00:00:00"
end_time = "2025-05-03T00:00:00"

bbox = bbox_mtg()
W,S,E,N = bbox[0], bbox[1], bbox[2], bbox[3]
NSWE = [N, S, W, E]

downloader = EUMDownloader(product_id=product_id, 
    output_dir="./data/datastore_data",
    max_parallel_conns=cpus,
)

downloader.download_interval(
    start_time=start_date, 
    end_time=end_time,
    bounding_box=NSWE, 
    observations_per_day=1,
    jump_minutes=60
)

MTGDataParallel(args, downloader, area_reprojection="mtg_fci_latlon_1km")