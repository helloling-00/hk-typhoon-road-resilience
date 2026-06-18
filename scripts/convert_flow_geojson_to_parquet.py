"""
Convert raw flow GeoJSON files to slot-based parquet.

Input:  <FLOW_RAW>/traffic_flow_zoom15_YYYY-MM-DD-HH-MM.geojson
Output: <FLOW_OUT>/YYYY-MM-DD/traffic_flow_zoom15_YYYY-MM-DD_slotNN_HHMM.parquet

Slot numbering: slot = hour * 2 + minute // 30  (0–47)
HHMM suffix:    slot * 30 minutes, formatted as HHMM (e.g. slot07 → 0330)

When multiple GeoJSON files fall in the same 30-min slot (same day+slot),
their records are concatenated into one parquet file.

Output schema (matches existing flow_parquet2/ files):
  layer            object
  road_category    object
  road_subcategory object
  left_hand_traffic float64
  road_closure     float64   (NaN = not closed, 1.0 = closed)
  relative_speed   float64
  geometry         object    (WKB bytes)
"""

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from shapely import wkb as shapely_wkb
from shapely.geometry import shape

FLOW_RAW = Path("/Users/helloling/workspace/HkRoadFlow/data/flow")
FLOW_OUT = Path("/Users/helloling/workspace/thesis/data/flow_parquet2")

FNAME_RE = re.compile(
    r"traffic_flow_zoom15_(\d{4}-\d{2}-\d{2})-(\d{2})-(\d{2})\.geojson$"
)


def slot_from_hhmm(hour: int, minute: int) -> int:
    return hour * 2 + minute // 30


def hhmm_suffix(slot: int) -> str:
    total_min = slot * 30
    return f"{total_min // 60:02d}{total_min % 60:02d}"


def geojson_to_rows(path: Path) -> list[dict]:
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
                "layer": props.get("layer"),
                "road_category": props.get("road_category"),
                "road_subcategory": props.get("road_subcategory"),
                "left_hand_traffic": float(props["left_hand_traffic"])
                if "left_hand_traffic" in props
                else np.nan,
                "road_closure": 1.0 if props.get("road_closure") else np.nan,
                "relative_speed": float(props["relative_speed"])
                if "relative_speed" in props
                else np.nan,
                "geometry": geom_bytes,
            }
        )
    return rows


def main():
    geojson_files = sorted(FLOW_RAW.glob("traffic_flow_zoom15_*.geojson"))
    print(f"Found {len(geojson_files):,} GeoJSON files", flush=True)

    # Group files by (date, slot)
    slot_groups: dict[tuple, list[Path]] = defaultdict(list)
    for fp in geojson_files:
        m = FNAME_RE.match(fp.name)
        if not m:
            continue
        date_str, hh, mm = m.group(1), int(m.group(2)), int(m.group(3))
        slot = slot_from_hhmm(hh, mm)
        slot_groups[(date_str, slot)].append(fp)

    multi = sum(1 for v in slot_groups.values() if len(v) > 1)
    print(f"Slot groups: {len(slot_groups):,}  (multi-file slots: {multi})", flush=True)

    FLOW_OUT.mkdir(parents=True, exist_ok=True)

    done = skipped = 0
    for (date_str, slot), files in sorted(slot_groups.items()):
        out_dir = FLOW_OUT / date_str
        out_dir.mkdir(exist_ok=True)
        suffix = hhmm_suffix(slot)
        out_path = out_dir / f"traffic_flow_zoom15_{date_str}_slot{slot:02d}_{suffix}.parquet"

        if out_path.exists():
            skipped += 1
            continue

        all_rows = []
        for fp in files:
            all_rows.extend(geojson_to_rows(fp))

        if not all_rows:
            continue

        df = pd.DataFrame(all_rows)
        df["left_hand_traffic"] = df["left_hand_traffic"].astype("float64")
        df["road_closure"] = df["road_closure"].astype("float64")
        df["relative_speed"] = df["relative_speed"].astype("float64")
        df.to_parquet(out_path, index=False)
        done += 1
        if done % 100 == 0:
            print(f"  {done} slots written ...", flush=True)

    print(f"Done. Written: {done}  Skipped (already exist): {skipped}")


if __name__ == "__main__":
    main()
