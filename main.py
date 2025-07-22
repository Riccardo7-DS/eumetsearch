import os 
from dotenv import load_dotenv
import eumdac
from eumdac import DataStore
from utils import products_list


# Load environment variables from .env file
load_dotenv()

consumer_key = os.getenv("EUMETSAT_CONSUMER_KEY")
consumer_secret = os.getenv("EUMETSAT_CONSUMER_SECRET")

credentials = (consumer_key, consumer_secret)
token = eumdac.AccessToken(credentials)
datastore = DataStore(token)

product_id = products_list["MTG-1"]["product_id"]
collection = datastore.get_collection(product_id)


from utils import EUMDownloader, bbox_mtg

bbox = bbox_mtg()  # Get the bounding box for MTG-1

downloader = EUMDownloader(product_id=product_id, 
    output_dir="../data", 
    format="netcdf4",
)

downloader.download_interval(
    start_time="2025-05-21T00:00:00Z", 
    end_time="2025-05-22T00:00:00Z",
    bounding_box=bbox, 
    observations_per_day=2,
    method="datastore",
)