import geopandas as gpd
path = r"E:\Document\paper_library\5th_paper_InSAR\datasets\Slope_unit_use\su_vect_a50000_c03.shp"
try:
    gdf = gpd.read_file(path)
    print("Columns:", gdf.columns.tolist())
    print("\nData Types:")
    print(gdf.dtypes)
    print("\nFirst 3 rows:")
    print(gdf.head(3))
except Exception as e:
    print(e)
