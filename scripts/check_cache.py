import geopandas as gpd
from pathlib import Path

f = Path("data/water_bodies/wb_77.5500_12.8400_77.6200_12.9100.geojson")
gdf = gpd.read_file(str(f))
gdf_m = gdf.to_crs("EPSG:3857")

print("water_type distribution:", gdf["water_type"].value_counts().to_dict())
print()
for wtype in ["drain", "stream", "water"]:
    sub = gdf_m[gdf_m["water_type"] == wtype]
    if len(sub) > 0:
        mn = sub.area.min()
        mx = sub.area.max()
        me = sub.area.mean()
        print(f"{wtype}: n={len(sub)}, min_area={mn:.0f}m2, max={mx:.0f}m2, mean={me:.0f}m2")
