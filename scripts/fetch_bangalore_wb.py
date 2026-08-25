"""
One-time script: fetch Bangalore (Gottigere) water bodies from Overpass
and save as data/water_bodies/bangalore_fallback.geojson.

Run from repo root:
    python scripts/fetch_bangalore_wb.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

# Gottigere, Bangalore bbox — matches DEMO_REGIONS["Bangalore (Gottigere)"]
S, W, N, E = 12.84, 77.55, 12.91, 77.62

QUERY = (
    "[out:json][timeout:60];\n"
    "(\n"
    f'  way["natural"="water"]({S},{W},{N},{E});\n'
    f'  relation["natural"="water"]({S},{W},{N},{E});\n'
    f'  way["waterway"="river"]({S},{W},{N},{E});\n'
    f'  way["waterway"="canal"]({S},{W},{N},{E});\n'
    f'  way["waterway"="stream"]({S},{W},{N},{E});\n'
    f'  way["waterway"="drain"]({S},{W},{N},{E});\n'
    f'  way["landuse"="reservoir"]({S},{W},{N},{E});\n'
    f'  way["landuse"="basin"]({S},{W},{N},{E});\n'
    f'  way["natural"="bay"]({S},{W},{N},{E});\n'
    f'  way["natural"="coastline"]({S},{W},{N},{E});\n'
    ");\n"
    "out geom;"
)

MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded",
    "User-Agent": "FloodRiskZonation/1.0 (fallback-data-fetch)",
}

OUT = Path("data/water_bodies/bangalore_fallback.geojson")


def fetch() -> dict:
    for mirror in MIRRORS:
        print(f"Trying {mirror} ...", flush=True)
        try:
            r = requests.post(
                mirror,
                data=QUERY.encode("utf-8"),
                headers=HEADERS,
                timeout=45,
            )
            if r.status_code == 200:
                print(f"  OK ({r.status_code})", flush=True)
                return r.json()
            print(f"  HTTP {r.status_code}", flush=True)
        except Exception as exc:
            print(f"  Error: {exc}", flush=True)
    raise RuntimeError("All Overpass mirrors failed")


def osm_to_geojson(osm: dict) -> dict:
    """Convert Overpass JSON to a minimal GeoJSON FeatureCollection."""
    from shapely.geometry import LineString, Polygon, mapping
    from shapely.ops import polygonize, unary_union

    features = []
    for el in osm.get("elements", []):
        etype = el.get("type")
        tags = el.get("tags", {})
        wtype = tags.get("natural", tags.get("waterway", tags.get("landuse", "water")))
        try:
            if etype == "way":
                pts = el.get("geometry", [])
                if len(pts) < 2:
                    continue
                coords = [(p["lon"], p["lat"]) for p in pts]
                is_closed = len(coords) >= 4 and coords[0] == coords[-1]
                geom = Polygon(coords).buffer(0) if is_closed else LineString(coords)
            elif etype == "relation":
                outer_lines = []
                for m in el.get("members", []):
                    if m.get("role", "outer") not in {"", "outer"}:
                        continue
                    pts = m.get("geometry", [])
                    if len(pts) >= 2:
                        outer_lines.append(
                            LineString([(p["lon"], p["lat"]) for p in pts])
                        )
                polys = list(polygonize(unary_union(outer_lines))) if outer_lines else []
                if not polys:
                    continue
                geom = unary_union(polys).buffer(0)
            else:
                continue
            if not geom.is_valid or geom.is_empty:
                continue
            features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(geom),
                    "properties": {
                        "water_type": wtype,
                        "name": tags.get("name", ""),
                    },
                }
            )
        except Exception as exc:
            print(f"  Skipping element {el.get('id')}: {exc}", flush=True)

    return {"type": "FeatureCollection", "features": features}


if __name__ == "__main__":
    print(f"Fetching Bangalore water bodies (bbox {S},{W} → {N},{E})...")
    osm = fetch()
    n_elements = len(osm.get("elements", []))
    print(f"Received {n_elements} Overpass elements.")

    geojson = osm_to_geojson(osm)
    n_features = len(geojson["features"])
    print(f"Converted to {n_features} GeoJSON features.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(geojson, indent=2), encoding="utf-8")
    print(f"Saved → {OUT}  ({OUT.stat().st_size:,} bytes)")
