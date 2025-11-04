from phidown import CopernicusDataSearcher
from eumetsearch import hoa_bbox
from shapely.geometry import box
import pdb

bbox = hoa_bbox()
dl = CopernicusDataSearcher()
aoi = box(*bbox).wkt

# Search for SYNERGY vegetation products
results = dl.query_by_filter(
    collection_name='SENTINEL-3',
    aoi_wkt=aoi,
    start_date='2023-05-01T00:00:00Z',
    end_date='2023-05-03T00:00:00Z',
    top=1000,
    attributes={
        'instrumentShortName': 'SYNERGY',
        'processingLevel': '2',
        'productType' : 'SY_2_SYN___'
    }
)

df = dl.execute_query()

# Display results
print(f"Number of results: {len(df)}")
# Display the first few rows of the DataFrame
print(dl.display_results(top_n=20))