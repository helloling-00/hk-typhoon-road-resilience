"""
Convert raw incident GeoJSON files to parquet.

Input:  <INC_RAW>/incidents_zoom15_YYYY-MM-DD-HH-MM.geojson
Output: <INC_OUT>/date=YYYY-MM-DD/hour=H/incidents_zoom15_YYYY-MM-DD-HH-MM.parquet

The timestamp (ts) is parsed from the filename and stored as HKT datetime64[ns]
(no timezone info — the API was polled at HKT times).

Output schema (matches existing incident_parquet/ files):
  inc_id              object
  ts                  datetime64[ns]   (HKT, naive)
  magnitude_of_delay  int32
  closed              object           (always None — field reserved, not populated)
  geometry_wkb        object           (WKB bytes, LineString or Point)
  delay               float64          (seconds, NaN if absent)
  description_0       object
  icon_category_0     Int32            (pandas nullable integer)
  road_category       object
  road_subcategory    object
  left_hand_traffic   boolean
  point_type          object           (None for LineString, "start_point" etc. for Point)
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from shapely import wkb as shapely_wkb
from shapely.geometry import shape

INC_RAW = Path("/Users/helloling/workspace/HkRoadFlow/data/incidents")
INC_OUT = Path("/Users/helloling/workspace/thesis/data/incident_parquet")

FNAME_RE = re.compile(
    r"incidents_zoom15_(\d{4}-\d{2}-\d{2})-(\d{2})-(\d{2})\.geojson$"
)


def parse_geojson(path: Path, ts: pd.Timestamp) -> pd.DataFrame:
    with open(path) as f:
        fc = json.load(f)

    rows = []
    for feat in fc.get("features", []):
        props = feat["properties"]
        try:
            geom_bytes = shapely_wkb.dumps(shape(feat["geometry"]))
        except Exception:
            continue

        rows.append(
            {
                "inc_id": props.get("id"),
                "ts": ts,
                "magnitude_of_delay": props.get("magnitude_of_delay"),
                "closed": None,
                "geometry_wkb": geom_bytes,
                "delay": float(props["delay"]) if "delay" in props and props["delay"] is not None else np.nan,
                "description_0": props.get("description_0"),
                "icon_category_0": props.get("icon_category_0"),
                "road_category": props.get("road_category"),
                "road_subcategory": props.get("road_subcategory"),
                "left_hand_traffic": props.get("left_hand_traffic"),
                "point_type": props.get("point_type"),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"])
    df["magnitude_of_delay"] = df["magnitude_of_delay"].astype("Int32").astype("int32")
    df["icon_category_0"] = df["icon_category_0"].astype("Int32")
    df["left_hand_traffic"] = df["left_hand_traffic"].astype("boolean")
    df["delay"] = df["delay"].astype("float64")
    return df


def main():
    geojson_files = sorted(INC_RAW.glob("incidents_zoom15_*.geojson"))
    print(f"Found {len(geojson_files):,} incident GeoJSON files", flush=True)

    INC_OUT.mkdir(parents=True, exist_ok=True)

    done = skipped = 0
    for fp in geojson_files:
        m = FNAME_RE.match(fp.name)
        if not m:
            continue
        date_str, hh, mm = m.group(1), int(m.group(2)), int(m.group(3))
        ts = pd.Timestamp(f"{date_str} {hh:02d}:{mm:02d}:00")

        out_dir = INC_OUT / f"date={date_str}" / f"hour={hh}"
        out_path = out_dir / fp.name.replace(".geojson", ".parquet")

        if out_path.exists():
            skipped += 1
            continue

        df = parse_geojson(fp, ts)
        if df.empty:
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        done += 1
        if done % 200 == 0:
            print(f"  {done} files written ...", flush=True)

    print(f"Done. Written: {done}  Skipped (already exist): {skipped}")


if __name__ == "__main__":
    main()
