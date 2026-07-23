"""Search Copernicus Data Space for Sentinel-3 SYNERGY products over a bbox.

Requires the optional ``phidown`` extra: pip install -e ".[phidown]"
"""

from phidown import CopernicusDataSearcher
from shapely.geometry import box

from eumetsearch import hoa_bbox


def search_sentinel3_synergy(start_date: str, end_date: str, top: int = 1000):
    """Query Sentinel-3 SYNERGY L2 vegetation products over the HOA bbox."""
    aoi = box(*hoa_bbox()).wkt
    searcher = CopernicusDataSearcher()
    searcher.query_by_filter(
        collection_name="SENTINEL-3",
        aoi_wkt=aoi,
        start_date=start_date,
        end_date=end_date,
        top=top,
        attributes={
            "instrumentShortName": "SYNERGY",
            "processingLevel": "2",
            "productType": "SY_2_SYN___",
        },
    )
    return searcher, searcher.execute_query()


if __name__ == "__main__":
    searcher, df = search_sentinel3_synergy("2023-05-01T00:00:00Z", "2023-05-03T00:00:00Z")
    print(f"Number of results: {len(df)}")
    print(searcher.display_results(top_n=20))
