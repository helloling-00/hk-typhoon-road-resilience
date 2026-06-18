"""
构建回归分析宽表：每行 = road_id × signal_period × time_group
Y = 速度偏差均值（台风实测 − 对应 day_type/slot 基线）
X = 道路特征 + 信号等级 + 人口特征 + 事故 + 土地利用

5 段时间划分：
  NIGHT    = slot 0–13   （00:00–07:00）凌晨
  AM_PEAK  = slot 14–18  （07:00–09:30）早高峰
  MIDDAY   = slot 19–33  （09:30–17:00）日间
  PM_PEAK  = slot 34–38  （17:00–19:30）晚高峰
  EVENING  = slot 39–47  （19:30–24:00）夜间

输出：data/regression_table.parquet
"""
import ast, glob, pickle, pandas as pd, numpy as np
from shapely import wkb as shapely_wkb
from shapely.geometry import LineString
import warnings; warnings.filterwarnings("ignore")

DATA = "/Users/helloling/workspace/thesis/data"
FLOW = f"{DATA}/flow_parquet2"

# ── 信号期定义（date, slot 粒度，每行精确到 0.5h） ─────────────────────────
# slot = 0-based 30min index from 00:00
def t2slot(hhmm):
    h, m = int(hhmm[:2]), int(hhmm[3:])
    return h * 2 + (1 if m >= 30 else 0)

SIGNAL_PERIODS = [
    # (typhoon, signal, date, slot_start, slot_end_excl)
    # ── Mitag ─────────────────────────────────────────────────────────────
    # S1: 09-17 21:20 → 09-19 09:20  (夜间，多天)
    ("Mitag", 1, "2025-09-17", t2slot("21:20"), 48),
    ("Mitag", 1, "2025-09-18", 0, 48),
    ("Mitag", 1, "2025-09-19", 0, t2slot("09:20")),
    # S3: 09-19 09:20 → 09-20 09:20
    ("Mitag", 3, "2025-09-19", t2slot("09:20"), 48),
    ("Mitag", 3, "2025-09-20", 0, t2slot("09:20")),
    # S1: 09-20 09:20 → 10:40
    ("Mitag", 1, "2025-09-20", t2slot("09:20"), t2slot("10:40")+1),
    # ── Ragasa ────────────────────────────────────────────────────────────
    # S1: 09-22 12:20 → 21:40 (注：09-22 数据大量缺失，但仍纳入)
    ("Ragasa", 1, "2025-09-22", t2slot("12:20"), t2slot("21:40")+1),
    # S3: 09-22 21:40 → 09-23 14:20
    ("Ragasa", 3, "2025-09-22", t2slot("21:40"), 48),
    ("Ragasa", 3, "2025-09-23", 0, t2slot("14:20")),
    # S8: 09-23 14:20 → 09-24 01:40
    ("Ragasa", 8, "2025-09-23", t2slot("14:20"), 48),
    ("Ragasa", 8, "2025-09-24", 0, t2slot("01:40")+1),
    # S9: 09-24 01:40 → 02:40
    ("Ragasa", 9, "2025-09-24", t2slot("01:40"), t2slot("02:40")+1),
    # S10: 09-24 02:40 → 13:20
    ("Ragasa", 10, "2025-09-24", t2slot("02:40"), t2slot("13:20")),
    # S8: 09-24 13:20 → 20:20
    ("Ragasa", 8, "2025-09-24", t2slot("13:20"), t2slot("20:20")),
    # S3: 09-24 20:20 → 09-25 08:20
    ("Ragasa", 3, "2025-09-24", t2slot("20:20"), 48),
    ("Ragasa", 3, "2025-09-25", 0, t2slot("08:20")),
    # S1: 09-25 08:20 → 11:20
    ("Ragasa", 1, "2025-09-25", t2slot("08:20"), t2slot("11:20")+1),
    # ── Matmo ─────────────────────────────────────────────────────────────
    # S1: 10-03 19:40 → 10-04 12:20
    ("Matmo", 1, "2025-10-03", t2slot("19:40"), 48),
    ("Matmo", 1, "2025-10-04", 0, t2slot("12:20")),
    # S3: 10-04 12:20 → 10-05 15:40
    ("Matmo", 3, "2025-10-04", t2slot("12:20"), 48),
    ("Matmo", 3, "2025-10-05", 0, t2slot("15:40")),
    # S1: 10-05 15:40 → 22:20
    ("Matmo", 1, "2025-10-05", t2slot("15:40"), t2slot("22:20")+1),
]

DATE_DAY_TYPE = {
    "2025-09-17": "WORKDAY", "2025-09-18": "WORKDAY", "2025-09-19": "WORKDAY",
    "2025-09-20": "SATURDAY",
    "2025-09-22": "WORKDAY", "2025-09-23": "WORKDAY", "2025-09-24": "WORKDAY",
    "2025-09-25": "WORKDAY",
    "2025-10-03": "WORKDAY", "2025-10-04": "SATURDAY",
    "2025-10-05": "SUNDAY_HOLIDAY",
}

def time_group(slot):
    if  0 <= slot <= 13: return "NIGHT"    # 00:00–07:00
    if 14 <= slot <= 18: return "AM_PEAK"  # 07:00–09:30
    if 19 <= slot <= 33: return "MIDDAY"   # 09:30–17:00
    if 34 <= slot <= 38: return "PM_PEAK"  # 17:00–19:30
    return "EVENING"                        # 19:30–24:00

# ── 基线 ─────────────────────────────────────────────────────────────────────
print("Loading baseline...", flush=True)
bl = pd.read_parquet(f"{DATA}/baseline_speed.parquet",
                     columns=["day_type","slot","road_id","mean_speed","n_obs"])
bl_lkp = bl.set_index(["road_id","day_type","slot"])["mean_speed"]
print(f"  baseline rows: {len(bl):,}")
del bl

# ep → road_id
ep = pd.read_parquet(f"{DATA}/ep_to_road.parquet")
ep_lkp = ep.set_index("ep_key")["road_id"]

def get_ep_key(wkb_bytes):
    try:
        g = shapely_wkb.loads(wkb_bytes)
        c = list(g.coords)
        s4 = (round(c[0][0],4), round(c[0][1],4))
        e4 = (round(c[-1][0],4), round(c[-1][1],4))
        return str((min(s4,e4), max(s4,e4)))
    except: return None

# ── 读一天一个 slot 的速度 ─────────────────────────────────────────────────
def read_slot(date_str, slot):
    pat = f"{FLOW}/{date_str}/traffic_flow_zoom15_{date_str}_slot{slot:02d}_*.parquet"
    fs = glob.glob(pat)
    if not fs: return {}
    df = pd.read_parquet(fs[0], columns=["geometry","road_closure","relative_speed"])
    df = df[df["road_closure"] != 1].dropna(subset=["relative_speed"])
    result = {}
    for _, row in df.iterrows():
        epk = get_ep_key(row["geometry"])
        if epk and epk in ep_lkp.index:
            rid = int(ep_lkp[epk])
            if rid not in result:
                result[rid] = []
            result[rid].append(row["relative_speed"])
    return {k: np.mean(v) for k, v in result.items()}

# ── 主循环：按信号期 × time_group 聚合 ──────────────────────────────────────
print("Processing signal periods...", flush=True)

# 将 SIGNAL_PERIODS 按 (typhoon, signal) 分组合并
from collections import defaultdict
period_slots = defaultdict(list)  # key=(typhoon, signal_str) → [(date, slot, day_type), ...]

for typhoon, signal, date, s_start, s_end in SIGNAL_PERIODS:
    dt = DATE_DAY_TYPE.get(date, "WORKDAY")
    for slot in range(s_start, min(s_end, 48)):
        period_slots[(typhoon, signal)].append((date, slot, dt))

# 对每个 (typhoon, signal) × time_group，读速度→算偏差→取均值
rows = []
for (typhoon, signal), slot_list in sorted(period_slots.items()):
    # 按 time_group 分组
    tg_slots = defaultdict(list)
    for date, slot, dt in slot_list:
        tg_slots[time_group(slot)].append((date, slot, dt))

    for tg, date_slot_list in tg_slots.items():
        print(f"  {typhoon} S{signal} {tg}: {len(date_slot_list)} slots", flush=True)

        # 逐 slot 读速度，汇总 per road
        road_speeds = defaultdict(list)
        road_devs   = defaultdict(list)

        for date, slot, dt in date_slot_list:
            spd_dict = read_slot(date, slot)
            for rid, spd in spd_dict.items():
                road_speeds[rid].append(spd)
                key = (rid, dt, slot)
                if key in bl_lkp.index:
                    road_devs[rid].append(spd - bl_lkp[key])

        for rid in road_devs:
            if len(road_devs[rid]) == 0: continue
            rows.append({
                "road_id":        rid,
                "typhoon":        typhoon,
                "signal_level":   signal,
                "time_group":     tg,
                "mean_speed":     float(np.mean(road_speeds[rid])),
                "mean_deviation": float(np.mean(road_devs[rid])),
                "n_slots":        len(road_devs[rid]),
            })

reg = pd.DataFrame(rows)
print(f"\n回归表行数: {len(reg):,}  ({reg['road_id'].nunique():,} 条路)")

# ── 合并 X 变量 ────────────────────────────────────────────────────────────────
print("Merging road features...", flush=True)

# road_category + 道路长度
with open(f"{DATA}/osm_cache/road_wkb_store.pkl","rb") as f:
    wkb_store = pickle.load(f)
ep_df = pd.read_parquet(f"{DATA}/ep_to_road.parquet")
rr = pd.read_parquet(f"{DATA}/road_registry.parquet")[
        ["road_id","road_category"]].drop_duplicates("road_id")

def road_length_m(row):
    rid = row["road_id"]
    if rid in wkb_store:
        try:
            g = shapely_wkb.loads(wkb_store[rid])
            # 转 EPSG:3857 近似米距
            from shapely.ops import transform
            import pyproj
            proj = pyproj.Transformer.from_crs("EPSG:4326","EPSG:3857",always_xy=True).transform
            g2 = transform(proj, g)
            return g2.length
        except: pass
    try:
        pts = ast.literal_eval(row["ep_key"])
        ls = LineString([pts[0], pts[1]])
        # rough degree→m at HK latitude (~22°): 1° lat ≈ 110km, 1° lon ≈ 101km
        dx = (pts[1][0]-pts[0][0])*101000
        dy = (pts[1][1]-pts[0][1])*110000
        return float(np.sqrt(dx**2+dy**2))
    except: return np.nan

# 只计算回归表中出现的 road_id，避免对所有 166k 条路操作
reg_rids = set(reg["road_id"].unique())
ep_sub = ep_df[ep_df["road_id"].isin(reg_rids)].drop_duplicates("road_id").copy()
print(f"  Computing road lengths for {len(ep_sub):,} roads...", flush=True)
ep_sub["road_length_m"] = ep_sub.apply(road_length_m, axis=1)

road_feat = ep_sub[["road_id","road_length_m"]].merge(rr, on="road_id", how="left")

# 基线均速（WORKDAY 基线，所有 slot 均值，代表该路"正常"速度水平）
bl2 = pd.read_parquet(f"{DATA}/baseline_speed.parquet",
                      columns=["road_id","day_type","mean_speed"])
baseline_avg = (bl2[bl2["day_type"]=="WORKDAY"]
                .groupby("road_id")["mean_speed"].mean()
                .rename("baseline_avg_speed").reset_index())
road_feat = road_feat.merge(baseline_avg, on="road_id", how="left")

# 人口特征
demo = pd.read_parquet(f"{DATA}/road_demo_features.parquet",
                       columns=["road_id","population_density_500m",
                                "median_income_500m","elderly_ratio_500m"])
road_feat = road_feat.merge(demo, on="road_id", how="left")

# 事故特征
inc_feat = pd.read_parquet(f"{DATA}/road_incident_features.parquet",
                            columns=["road_id","incident_count_500m",
                                     "severe_incident_500m","closure_nearby_500m"])
road_feat = road_feat.merge(inc_feat, on="road_id", how="left")

# POI 特征
poi = pd.read_parquet(f"{DATA}/road_poi_features.parquet",
                      columns=["road_id","poi_count_500m","poi_diversity_500m"])
road_feat = road_feat.merge(poi, on="road_id", how="left")

# 合并进回归表
reg = reg.merge(road_feat, on="road_id", how="left")

# ── 编码分类变量 ──────────────────────────────────────────────────────────────
# signal_level 分组：low(1), medium(3), high(8+)
def signal_group(s):
    if s <= 1:  return "S1"
    if s <= 3:  return "S3"
    if s <= 8:  return "S8"
    return "S10"
reg["signal_group"] = reg["signal_level"].apply(signal_group)

# road_category → 宽类
def road_broad(cat):
    if pd.isna(cat): return "other"
    c = str(cat)
    if "motorway" in c: return "motorway"
    if "trunk" in c:    return "trunk"
    if "primary" in c:  return "primary"
    if "secondary" in c: return "secondary"
    if "tertiary" in c:  return "tertiary"
    if "residential" in c: return "residential"
    return "other"
reg["road_broad"] = reg["road_category"].apply(road_broad)

print(f"\n回归表概况:")
print(reg[["mean_deviation","mean_speed","n_slots"]].describe().round(3))
print(f"\n按 typhoon × signal_group 分布:")
print(reg.groupby(["typhoon","signal_group"])["road_id"].count().to_string())
print(f"\n缺失值统计:")
missing = reg.isnull().sum()
print(missing[missing>0].to_string())

out = f"{DATA}/regression_table.parquet"
reg.to_parquet(out, index=False)
print(f"\nSaved: {out}")
