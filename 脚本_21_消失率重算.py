"""
重新计算三次台风各信号期的路段消失率
参考基准：baseline_speed.parquet 中 n_obs≥2 的 (day_type, slot) 对应路段集合
处理跨日类型的信号期：按每个 slot 的实际日期确定 day_type
"""
import glob, pickle, pandas as pd, numpy as np
from datetime import datetime, timedelta

FLOW = "/Users/helloling/workspace/thesis/data/flow_parquet2"
DATA = "/Users/helloling/workspace/thesis/data"

# ── 日期 → day_type 映射 ──────────────────────────────────────────────────────
DAY_TYPE = {}
# 香港公众假期（数据范围内）
HK_HOLIDAYS = {"2025-10-01"}  # 国庆日
import calendar
from datetime import date as Date
for y in [2025]:
    for m in range(9, 11):
        for d in range(1, 32):
            try:
                dt = Date(y, m, d)
            except ValueError:
                continue
            s = dt.strftime("%Y-%m-%d")
            if s in HK_HOLIDAYS:
                DAY_TYPE[s] = "SUNDAY_HOLIDAY"
            elif dt.weekday() == 5:
                DAY_TYPE[s] = "SATURDAY"
            elif dt.weekday() == 6:
                DAY_TYPE[s] = "SUNDAY_HOLIDAY"
            else:
                DAY_TYPE[s] = "WORKDAY"

# ── 信号期定义（按台风完整升降过程） ─────────────────────────────────────────
PERIODS = {
    "Mitag": [
        ("S1↑",  "2025-09-17 21:20", "2025-09-19 09:20"),
        ("S3",   "2025-09-19 09:20", "2025-09-20 09:20"),
        ("S1↓",  "2025-09-20 09:20", "2025-09-20 10:40"),
    ],
    "Ragasa": [
        ("S1↑",  "2025-09-22 12:20", "2025-09-22 21:40"),
        ("S3↑",  "2025-09-22 21:40", "2025-09-23 14:20"),
        ("S8↑",  "2025-09-23 14:20", "2025-09-24 01:40"),
        ("S10",  "2025-09-24 02:40", "2025-09-24 13:20"),  # S9 treated same as S10 visually
        ("S8↓",  "2025-09-24 13:20", "2025-09-24 20:20"),
        ("S3↓",  "2025-09-24 20:20", "2025-09-25 08:20"),
        ("S1↓",  "2025-09-25 08:20", "2025-09-25 11:20"),
    ],
    "Matmo": [
        ("S1↑",  "2025-10-03 19:40", "2025-10-04 12:20"),
        ("S3",   "2025-10-04 12:20", "2025-10-05 15:40"),
        ("S1↓",  "2025-10-05 15:40", "2025-10-05 22:20"),
    ],
}

# ── 辅助：展开时段到 (date_str, slot_int) 列表 ────────────────────────────────
def slots_in_period(start_str, end_str):
    fmt = "%Y-%m-%d %H:%M"
    start = datetime.strptime(start_str, fmt)
    end   = datetime.strptime(end_str,   fmt)
    cur = start.replace(minute=(start.minute // 30) * 30)
    result = []
    while cur < end:
        result.append((cur.strftime("%Y-%m-%d"), cur.hour * 2 + cur.minute // 30))
        cur += timedelta(minutes=30)
    return result

# ── 加载 baseline_speed，构建 (day_type, slot) → set[road_id] ─────────────────
print("Loading baseline_speed.parquet...", flush=True)
bl = pd.read_parquet(f"{DATA}/baseline_speed.parquet")
bl2 = bl[bl["n_obs"] >= 2]
ref_lookup = {}   # (day_type, slot) → frozenset of road_ids
for (dt, sl), grp in bl2.groupby(["day_type", "slot"]):
    ref_lookup[(dt, sl)] = frozenset(grp["road_id"].values)
print(f"  Reference lookup built: {len(ref_lookup)} (day_type, slot) combos", flush=True)
del bl, bl2

# ── 加载 ep_to_road，构建 ep_key → road_id 映射 ───────────────────────────────
print("Loading ep_to_road...", flush=True)
ep = pd.read_parquet(f"{DATA}/ep_to_road.parquet")
ep_lkp = ep.set_index("ep_key")["road_id"]
del ep

# ── 读取单个 slot 文件，返回 road_id 集合 ────────────────────────────────────
def get_ep_key(wkb_bytes):
    try:
        from shapely import wkb as swkb
        g = swkb.loads(wkb_bytes)
        coords = list(g.coords)
        s, e = coords[0], coords[-1]
        s4 = (round(s[0],4), round(s[1],4))
        e4 = (round(e[0],4), round(e[1],4))
        return str((min(s4, e4), max(s4, e4)))
    except:
        return None

def roads_observed_in_slots(slot_list, show_missing=False):
    seen = set()
    missing = []
    for date_str, slot_i in slot_list:
        pat = f"{FLOW}/{date_str}/traffic_flow_zoom15_{date_str}_slot{slot_i:02d}_*.parquet"
        files = glob.glob(pat)
        if not files:
            missing.append((date_str, slot_i))
            continue
        df = pd.read_parquet(files[0], columns=["geometry", "road_closure"])
        df = df[df["road_closure"] != 1]
        for wb in df["geometry"]:
            epk = get_ep_key(wb)
            if epk and epk in ep_lkp.index:
                seen.add(int(ep_lkp[epk]))
    if show_missing and missing:
        print(f"    [missing {len(missing)} slots: {missing[:3]}{'...' if len(missing)>3 else ''}]", flush=True)
    return seen

# ── 对每个信号期计算消失率 ───────────────────────────────────────────────────
print("\n=== Computing disappearance rates ===\n", flush=True)

RESULTS = {}

for typhoon, periods in PERIODS.items():
    print(f"── {typhoon} ──────────────────────────────────", flush=True)
    RESULTS[typhoon] = []
    for label, start_s, end_s in periods:
        slot_list = slots_in_period(start_s, end_s)
        n_slots = len(slot_list)

        # 按 slot 的 day_type 分组，累积参考集
        ref_set = set()
        dt_breakdown = {}
        for date_str, slot_i in slot_list:
            dt = DAY_TYPE.get(date_str, "WORKDAY")
            key = (dt, slot_i)
            rds = ref_lookup.get(key, frozenset())
            ref_set.update(rds)
            dt_breakdown[dt] = dt_breakdown.get(dt, 0) + 1
        n_ref = len(ref_set)

        # 读实际台风观测
        obs_set = roads_observed_in_slots(slot_list, show_missing=True)
        n_obs_roads = len(obs_set)

        disappeared = ref_set - obs_set
        n_dis = len(disappeared)
        pct = n_dis / n_ref * 100 if n_ref > 0 else 0

        dt_str = ", ".join(f"{v}×{k[:3]}" for k,v in dt_breakdown.items())
        print(f"  {label:8s} {start_s[5:16]} → {end_s[5:16]}  [{n_slots} slots, {dt_str}]")
        print(f"           ref={n_ref:,}  obs={n_obs_roads:,}  disappeared={n_dis:,}  ({pct:.1f}%)", flush=True)

        RESULTS[typhoon].append({
            "label": label, "start": start_s, "end": end_s,
            "n_slots": n_slots, "day_types": dt_breakdown,
            "n_ref": n_ref, "n_obs_roads": n_obs_roads,
            "n_disappeared": n_dis, "pct_disappeared": round(pct, 1),
        })
    print()

# ── 汇总表 ──────────────────────────────────────────────────────────────────
print("\n=== SUMMARY TABLE ===\n")
print(f"{'Typhoon':<10} {'Period':<8} {'Ref':>7} {'Observed':>9} {'Disappeared':>12} {'Pct':>6}  {'Date range'}")
print("-" * 80)
for typhoon, rows in RESULTS.items():
    for r in rows:
        print(f"{typhoon:<10} {r['label']:<8} {r['n_ref']:>7,} {r['n_obs_roads']:>9,} "
              f"{r['n_disappeared']:>12,} {r['pct_disappeared']:>5.1f}%  "
              f"{r['start'][5:16]} → {r['end'][5:16]}")
    print()

# 保存结果
import json, os
out_path = f"/Users/helloling/workspace/thesis/data/osm_cache/disappearance_rates.pkl"
with open(out_path, "wb") as f:
    pickle.dump(RESULTS, f)
print(f"Saved to {out_path}")
