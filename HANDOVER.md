# 香港台风道路韧性研究 · 项目交接文档

**项目背景**：本文档记录整个研究过程中的探索思路、关键发现和遗留问题，供后续研究者接手使用。

**核心问题**：台风期间香港道路速度为什么会"变好"？这个"变好"是真实的需求抑制（出行减少→路变畅），还是路段采样偏差（慢路消失→均值虚高）？台风过后恢复有多快？哪些地方恢复慢？

---

## 目录

- [1. 数据](#1-数据)
  - [1.1 原始数据](#11-原始数据)
  - [1.2 处理后的 Parquet](#12-处理后的-parquet)
  - [1.3 三场台风的信号时间表](#13-三场台风的信号时间表)
  - [1.4 路网覆盖验证](#14-路网覆盖验证)
  - [1.5 数据缺失](#15-数据缺失)
- [2. 稳定路段 ID 的构建](#2-稳定路段-id-的构建)
- [3. 基线构建](#3-基线构建)
- [4. 分析框架](#4-分析框架)
- [5. Pre-Event 探索](#5-pre-event-探索)
- [6. During-Event 探索](#6-during-event-探索)
- [7. Post-Event 恢复分析](#7-post-event-恢复分析)
- [8. 应急可达性分析](#8-应急可达性分析)
- [9. 建成环境特征的构建](#9-建成环境特征的构建)
- [10. 有效脚本清单](#10-有效脚本清单)
- [11. 后续研究可以做什么](#11-后续研究可以做什么)
- [12. 技术坑与注意事项](#12-技术坑与注意事项)

---

## 1. 数据

### 1.1 原始数据

数据来自 TomTom Traffic API 抓取，覆盖 **2025 年 9 月 13 日至 10 月 10 日**，共 28 天。分两类：

**交通流速（Flow）**
- 1,474 个 GeoJSON，约 35.9 GB，命名 `traffic_flow_zoom15_YYYY-MM-DD-HH-MM.geojson`
- 采集频率约 30 分钟一次（:16 和 :46），正常天 48 个文件/天
- 每个文件约 4,700 条路段 Feature，几何类型 MultiLineString

| 字段 | 类型 | 说明 |
|------|------|------|
| `road_category` | string | motorway / trunk / primary / secondary / tertiary / street / service 及各 `_link` |
| `road_subcategory` | string / 缺失 | residential / unclassified / driveway / living_street，部分 feature 无此字段 |
| `left_hand_traffic` | bool | 香港全部 true |
| `relative_speed` | float [0, 1] | 当前速度 ÷ 自由流速度 |
| `road_closure` | bool / **缺失** | 仅在路段关闭时出现（约 0.2% feature），未关闭时此字段不存在 |

**交通事件（Incidents）**
- 2,030 个 GeoJSON，约 40 MB，命名 `incidents_zoom15_YYYY-MM-DD-HH-MM.geojson`
- 每个 30 分钟槽内约两个文件（:15 和 :45），覆盖至 2025-10-09
- 几何类型 LineString（路段）或 Point（起点标记）

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 事件唯一 ID（TTI- 前缀） |
| `icon_category_0` | int | 事件类型（6=拥堵，8=封路/事故） |
| `description_0` | string | Slow / Queuing / Stationary / Closed |
| `magnitude_of_delay` | int [1, 4] | 延误等级 |
| `delay` | float / 缺失 | 延误秒数，约 30% 有值 |

原始 GeoJSON 中事件的时间戳**不在 properties 里**，来自文件名，即文件名时间就是采集时间（HKT）。

### 1.2 处理后的 Parquet

原始 GeoJSON 35.9 GB，转 Parquet 后 3.1 GB（约 10:1 压缩）。转换脚本为 `convert_flow_geojson_to_parquet.py` 和 `convert_incidents_geojson_to_parquet.py`。

```
data/
├── flow_parquet2/             ← 3.1 GB，按天分文件夹
│   └── 2025-09-13/
│       ├── traffic_flow_zoom15_2025-09-13_slot00_0000.parquet
│       └── ... （slot01~slot47，HHMM 后缀 = slot×30 分钟）
└── incident_parquet/          ← 16 MB，date=/hour= 两级分区
    └── date=2025-09-13/hour=19/incidents_zoom15_2025-09-13-19-45.parquet
```

Flow parquet 字段同原始，几何改为 WKB bytes，`road_closure` 规范化为 NaN（未关闭）/ 1.0（关闭）。

**重要**：过滤封闭路段用 `df["road_closure"] != 1`，不要用 `== 0`，因为未关闭是 NaN 不是 0。

Incident parquet 关键字段：`inc_id`、`ts`（HKT datetime64[ns]）、`magnitude_of_delay`（int32）、`delay`（float64）、`geometry_wkb`、`description_0`、`icon_category_0`（Int32 nullable）。

### 1.3 三场台风的信号时间表

数据范围内三场台风（所有时间 HKT）：

| 台风 | 英文名 | 最高信号 | 信号段 |
|------|--------|----------|--------|
| 米娜 | **Mitag** | S3 | S1: 09-17 21:20→09-19 09:20；S3: 09-19 09:20→09-20 09:20；S1: →10:40 |
| 樺加沙 | **Ragasa** | S10 | S1: 09-22 12:20→21:40；S3: →09-23 14:20；S8: →09-24 01:40；S9: →02:40；**S10: 02:40→13:20**；S8: →20:20；S3: →09-25 08:20；S1: →11:20 |
| 麥德姆 | **Matmo** | S3 | S1: 10-03 19:40→10-04 12:20；S3: →10-05 15:40；S1: →22:20 |

Ragasa 是三者中唯一到 S10 的，提供了从 S1 到 S10 的完整强度梯度，是分析主体。Mitag 和 Matmo 都只到 S3，用作弱台风对照。

### 1.4 路网覆盖验证

**静态覆盖（vs OSM）**：以 OSM 机动车路网（159,946 条边）为参照，50m 空间匹配（`脚本_08` → 图16，`脚本_19` → 图19）：

| 道路类别 | 覆盖率 |
|----------|--------|
| Motorway | 100% |
| Trunk / Primary / Secondary / Tertiary | 99–99.6% |
| Street / residential / unclassified | 64–80% |
| Service / other | 58.7% |

主干路几乎全覆盖，本地支路偏低——这是浮动车数据的结构性限制，低流量道路 probe vehicle 不足。

**动态覆盖（台风期间路段消失）**：`脚本_21` 对每个信号期做了定量计算（参照 n_obs≥2 的基线路段集）：

| 台风 | 信号期 | 参照路段 | 实际观测 | 消失率 |
|------|--------|---------|---------|--------|
| Mitag | S1↑ | 43,759 | 32,296 | 46.1% |
| Mitag | S3 | 40,573 | 28,638 | 47.7% |
| Ragasa | S1↑ | 37,011 | 5,735 | **85.5%**（叠加 09-22 文件级缺口）|
| Ragasa | S3↑ | 35,151 | 21,742 | 52.1% |
| Ragasa | S8↑ | 36,305 | 13,804 | 67.2% |
| Ragasa | S10 | 32,523 | 13,853 | 66.1% |
| Ragasa | S3↓ | 27,412 | 15,544 | 55.6% |
| Matmo | S3 | 21,493 | 28,250 | 35.1% |

`脚本_20`（→ 图18）的五面板分析证明消失是**采样稀疏**而非偏差：
- 消失于 S3 的路段，中位基线 slot 数 ≈ 1（本来就很少被采到）
- 存活至 S10 的路段，中位 slot 数 ≈ 46
- 基线速度与消失概率相关仅 r = −0.13（快路慢路消失率相近）

**结论**：网络均值在台风期抬升是真实的需求抑制，不是"慢路消失"造成的虚高。

### 1.5 数据缺失

**文件级缺口**：
- 09-22（Ragasa S1↑期间）：05:00–20:30，32 个 slot 完全缺失——最严重
- 09-14 下午 19 槽；09-15 散乱缺 24 槽；09-16 缺 slot04；09-17 缺 slot37

09-22 的缺口直接影响 Ragasa pre-event 分析：pre-event 实际从 S3 开始，S1 期间几乎没有数据。

**路段级消失**（文件存在但路段未出现）：见 1.4 节。

---

## 2. 稳定路段 ID 的构建

### 2.1 问题

TomTom 不给路段稳定 ID，每次快照只有 WKB 几何。同一条物理道路在不同时刻的 WKB 会略有不同（顶点数差一两个、坐标精度微小浮动）。实测：466,187 个唯一 WKB 对应 166,159 条物理道路，平均每条路有 2.8 个 WKB 变体（中位数 1，极端情况 516）。

直接用 WKB 匹配会把同一条路当多条路分别统计，时序分析全部出错。

### 2.2 解决方案：端点键（ep_key）

提取 WKB 首尾端点坐标（四舍五入到小数点后 4 位），用端点对作为物理道路的稳定标识符：

```python
from shapely import wkb as shapely_wkb

def get_ep_key(wb):
    g = shapely_wkb.loads(bytes(wb))
    coords = (list(g.coords) if g.geom_type == "LineString"
              else [c for line in g.geoms for c in line.coords])
    s = (round(coords[0][0], 4), round(coords[0][1], 4))
    e = (round(coords[-1][0], 4), round(coords[-1][1], 4))
    return str((min(s, e), max(s, e)))  # min/max 统一方向
```

这个函数在几乎所有分析脚本里都会出现。

### 2.3 生成的文件

扫描全部 28 天 × 48 槽后生成：

- `data/road_registry.parquet`（466,187 行）：字段 `wkb_hash`、`road_id`、`road_category`、`road_subcategory`、`ep_key`
- `data/ep_to_road.parquet`（166,159 行）：ep_key → road_id 的 1:1 映射

标准使用方式：读出 WKB → 算 ep_key → merge ep_to_road → 得 road_id → groupby road_id 取 mean（同一条路的多个 WKB 变体在同一 slot 内可能都出现）。

> ⚠️ `road_registry.parquet` 里有 `wkb_hash` 列，来自 Python 内置 `hash()`，跨进程不稳定。**只用 ep_key 做连接**。

---

## 3. 基线构建

### 3.1 逻辑

基线定义：同一条路（road_id）、同一个 30 分钟 slot、同一天类型，在所有非台风日的观测均值：

```
baseline[road_id, slot, day_type] = mean(relative_speed)  跨所有非台风日
```

台风日（排除出基线）：09-17~20, 09-22~25, 10-03~05, 10-10。

### 3.2 日类型：为什么要分三类

画出实际速度曲线后发现周六是独立形态——早高峰比周日明显但比工作日平缓，强行并入任何一类都会引入偏差。最终三分：

- **WORKDAY**：周一至周五，排除公众假期
- **SATURDAY**：周六单独一类
- **SUNDAY_HOLIDAY**：周日 + 法定节假日

数据范围内香港法定假期：**10-01（国庆节）** 和 **10-07（中秋节翌日）**。

### 3.3 基线质量

`data/baseline_speed.parquet`（1,677,503 行），字段：`road_id`、`day_type`、`slot`、`mean_speed`、`std_speed`、`n_obs`。

| n_obs 阈值 | 比例 |
|-----------|------|
| ≥ 1 | 100% |
| ≥ 2 | 53.4% |
| ≥ 3 | 30.3% |
| ≥ 5 | 14.5% |

SATURDAY 类最多只有 2 个周六，n_obs 最多为 2。回归分析用 `max_days_seen ≥ 3` 过滤，分析 n_obs=1 的格子应标注"不可靠"。

### 3.4 标准加载模板

```python
def load_slot(day, slot, day_type):
    pat = f"{FLOW}/{day}/traffic_flow_zoom15_{day}_slot{slot:02d}_*.parquet"
    fs = glob.glob(pat)
    if not fs: return None

    df = pd.read_parquet(fs[0], columns=["relative_speed", "geometry", "road_closure"])
    df = df[df["road_closure"] != 1].dropna(subset=["relative_speed"])
    if len(df) < 50: return None

    df["ep_key"] = df["geometry"].apply(get_ep_key)
    df = df.merge(ep[["ep_key", "road_id"]], on="ep_key", how="inner")
    obs = df.groupby("road_id")["relative_speed"].mean()

    bl_vals = bl_idx.reindex(
        pd.MultiIndex.from_arrays([[day_type]*len(obs), [slot]*len(obs), obs.index],
                                   names=["day_type","slot","road_id"])
    ).values
    deviation = obs.values - bl_vals   # 正值=比常态快，负值=比常态慢
    ...
```

---

## 4. 分析框架

所有分析的核心是**偏差（deviation）**：观测 relative_speed 减去同时段基线。

道路状态分类用 ±0.03 对称阈值（F/N/S）：

```python
DEV_HI, DEV_LO = 0.03, -0.03
# F = Faster than baseline (dev > +0.03)
# N = Near baseline (|dev| ≤ 0.03)
# S = Slower than baseline (dev < -0.03)
```

**0.03 的依据**：用这个阈值统计正常工作日，F 和 S 各约占 25–26%（"自然分化率"37–40%）。报告台风日 F/S 比例时，必须和对照工作日对比，不能直接读绝对数字。`脚本_51` 专门计算了这个对照基准。

三阶段框架：
- **Pre-event**：S8 发布前（S1/S3 期间），看低信号是否已改变出行
- **During-event**：S8/S9/S10 期间，看需求抑制强度
- **Post-event**：S8 降级后到 all-clear，看恢复速度

---

## 5. Pre-Event 探索

### 5.1 发现问题

`脚本_01` 画了 09-22 到 09-25 的网络平均速度时序，在 09-23（S3 有效，S8 尚未发布）出现两个反常现象：

- **08:30 早高峰清空**：台风速度高于基线（+0.021），正常早高峰是速度最低点
- **13:00 午间逆转**：到中午速度跌到基线以下（−0.010）

两个信号出现在 S8 发布（14:20）之前，说明 S3 就已改变出行行为。`脚本_47` 对比了只到 S3 的 Mitag——Mitag 的 pre-signal 期没有类似午间下沉，进一步证明这个模式与"Ragasa 将升 S8"的预期行为相关，而非 S3 本身直接导致。

### 5.2 量化：长度加权 F/N/S

`脚本_53` 把分析降到路段级，用 ±0.03 阈值统计 08:30 和 13:00 两时刻的路段状态，**按路段长度加权**：

| 时刻 | 组别 | Faster | Near | Slower |
|------|------|--------|------|--------|
| 08:30 | 对照工作日均值（8天）| 25.6% | 49.7% | 24.7% |
| 08:30 | Sep 23 S3 | **45.6%** | 29.2% | 25.2% |
| 13:00 | 对照工作日均值 | 26.1% | 49.6% | 24.3% |
| 13:00 | Sep 23 S3 | 33.9% | 30.3% | **35.8%** |

对照工作日取 8 天（Sep 16, 26, 29, 30; Oct 02, 06, 08, 09）。关键是差值：08:30 的 Faster 比对照高 20pp；13:00 的 Slower 比对照高约 11pp。

### 5.3 转移矩阵

上午畅通、中午反慢——在同一条路上有多大的关联？`脚本_56` 构建了 08:30→13:00 的 3×3 转移矩阵（长度加权）：

| 08:30 → 13:00 | → F | → N | → S |
|---------------|-----|-----|-----|
| F（42.4%）| 14.9% | 11.0% | **16.5%** |
| N（35.3%）| 8.4% | 21.4% | 5.5% |
| S（22.2%）| 8.3% | 3.5% | 10.5% |

最大单一转变是 **F→S（16.5%）**：上午被提前清空、中午反而变慢，是 midday reversal 的核心路段群，空间上主要集中在九龙北部和新界西南。

### 5.4 回归：哪些特征预测清空/逆转

`脚本_59d`（最终版）用二元 logit 解释 **08:30 Faster（早高峰清空）** 和 **13:00 Slower（午间变慢）**。

**Morning clearance（n=1,465，AUC=0.621，McFadden R²=0.041）**：
- 职住比 OR=2.69***：职住不均的区域更容易清空
- 劳动人口比 OR=0.35*：劳动密集区反而不容易清空（台风天仍需出行）
- 学校数 1km OR=1.30**：学校附近更容易清空（停课减少接送）
- 距海岸线 OR=1.18*：内陆比沿海更容易清空

**Midday slowdown（n=1,721，AUC=0.637，McFadden R²=0.048）**：
- 学校数 1km OR=1.46***
- 零售密度 OR=1.51*：商业街午间更易变慢
- 事故数 500m OR=1.16*
- 旅游密度 OR=0.46***：旅游区反而不易变慢（游客出行已被抑制）

AUC 0.62–0.64 不高，McFadden R² 0.04–0.05 在路段级空间行为分析中是正常范围——这些模型是"关联证据"而不是预测工具。

> **尝试过的方向**：网格 OLS（500m 网格聚合后做回归，结果不如路段 logit 清晰，放弃）；POI 按"出行必要性"分三组再回归（结果和分类别 POI 差不多，无额外信息量，放弃）；多项 logit 预测转移类型（F→S vs F→F），预测力不比单阶段 logit 高，没有采用。

---

## 6. During-Event 探索

### 6.1 基线拥堵水平是最强调节变量

`脚本_02` 最先发现剂量-响应关系：S10 下 45.3% 路段更快，S1 下只有 32.9%。

`脚本_06` 按基线拥堵水平五分位分组看偏差（→ 图13）：

| 基线拥堵分位 | 平均偏差 | % Faster |
|-------------|---------|----------|
| Q1（最拥堵）| +0.096 | 44% |
| Q2 | +0.061 | 23% |
| Q3 | +0.009 | 9% |
| Q4 | −0.001 | 5% |
| Q5（最畅通）| −0.002 | 5% |

Q1（原本最拥堵的路）改善最大；Q5（高速路，本来就接近自由流）几乎没变化——天花板效应，速度最高只到 1.0，原来就快的路没有上升空间。这成为 during-event 分析的核心框架。

### 6.2 早高峰的残存结构

`脚本_52b` 比较 S3、S10 和基线的早高峰形态（→ 图25e）：

| 组 | 06:00 速度 | 低谷速度 | 下降幅度 |
|----|-----------|---------|---------|
| 基线工作日 | 0.835 | 0.716 | −0.119 |
| Signal 3 | 0.922 | 0.782 | −0.140 |
| Signal 10 | 0.970 | 0.829 | −0.141 |

S10 整体速度极高（0.970），但早上 8 点仍有下沉，幅度比基线还大（−0.141 vs −0.119）——即使在极端需求抑制下，仍有必要出行（必要工种、紧急服务）在产生微弱早高峰。

`脚本_52d` 的晚高峰分析（→ 图25f）：Ragasa S10 后 falling S8 晚高峰速度比 rising S8 更低（0.822 vs 0.846）、观测路段数更多——S10 过去后需求已经开始回流。

### 6.3 异质性面板

`脚本_62` 把 Ragasa 三天（rising/peak/falling）× 6 个时段 × 五分位做成 panel（→ 图25h），三个稳健模式：
1. 基线拥堵的主导性贯穿全程（Q1 peak 早高峰偏差超 +0.15，Q5 全程接近零）
2. 商业密度梯度在 peak day 最明显
3. falling 阶段（09-25）偏差已显著回落，出行恢复早于官方 all-clear

---

## 7. Post-Event 恢复分析

### 7.1 恢复定义

恢复起算：Ragasa S8→S3 降级（09-24 20:20），终点 all-clear（09-25 11:20），15h 窗口。未恢复标记为右删失。

恢复定义：|dev| ≤ 0.03 连续两槽（双向恢复，即速度既不显著偏高也不显著偏低于基线）。

### 7.2 关键发现（`脚本_03` → 图06/07/08）

- 15h 窗口内仅 **28.7%** 的路段达到双向恢复
- 中位恢复时间 **35.7 小时**（远超官方 all-clear 时间）
- 恢复曲线有**两波**：S8→S3 降级后一波，次日早高峰一波——第二波说明恢复与日常节奏的重启绑定
- Link road 恢复最快，arterial 最慢（主干道承接了回流需求）
- Ragasa（S10）偏差幅度最大（+0.075），正偏差残留最久；Mitag/Matmo（S3）整个周期偏差接近零

### 7.3 残余 disruption 初探

恢复最慢的 200 条路段 spot check：87% 的 post-event 低谷与正常低谷的差距在 0.02 以内（需求回流而非路坏）。剩余 13% 低谷显著低于正常——这些是真正的残余 disruption 候选，值得用 incident 数据做 500m overlay 验证。**这个分析没有做完。**

### 7.4 Cox PH 模型（只写了 spec，没跑）

计划用 Cox 比例风险模型解释路段恢复时间：

```
h_i(t) = h_0(t) × exp(β1·Cat_i + β2·BCQ_i + β3·X_built + β4·X_incident + β5·Coast_i)
```

其中 Cat = 道路类别，BCQ = 基线拥堵五分位，X_built 含职住比/零售/旅游密度，X_incident = 台风期 500m 内事故数，Coast = 距海岸线距离。所有变量在 `regression_table.parquet` 里已经准备好，只需准备路段级恢复时间数据（从 `脚本_03` 输出提取）就可以跑。**这是后续最直接可以接着做的工作。**

---

## 8. 应急可达性分析

`脚本_63`（→ 图25j）：台风期间速度变化是否改变了屋苑到应急服务的可达性？

方法：每个屋苑 3km 缓冲区内主干路平均速度，对比基线（聚焦 09-24 midday S10）。

主要发现：
- 基线可达性最差的屋苑（Q1）改善最大（+0.018），最好的（Q5）只 +0.003——拥堵分层的逻辑再次出现
- 大部分屋苑基线可达性未恶化

局限：只有一个时间截面；没有考虑路段消失的屋苑；没有区分 emergency vehicle 与一般车辆。

---

## 9. 建成环境特征的构建

回归分析的特征变量分散在多个脚本中：

| 脚本 | 产物 | 说明 |
|------|------|------|
| `脚本_25_屋苑宽表.py` | `estate_features.parquet` | 人口普查网格 → 屋苑级人口属性 |
| `脚本_26_道路人口特征.py` | `road_demo_features.parquet` | 人口网格数据附加到路段（500m 缓冲） |
| `脚本_27a_事故特征.py` | `road_incident_features.parquet` | 路段 500m 内 incident 数 |
| `脚本_27b_POI特征.py` | `road_poi_features.parquet` | OSM 八类 POI 密度（学校/零售/餐饮/旅游/医疗/交通/金融/市政） |
| `脚本_27c_道路结构特征.py` | `road_structural_features.parquet` | 交叉口度数、道路长度、距海岸线 |
| `脚本_64_school_elderly_features.py` | `road_school_elderly_features.parquet` | 补充学校和长者设施数量 |
| `脚本_28b_更新回归表.py` | `regression_table.parquet` | 拼接所有特征（最终版宽表） |

OSM 数据已下载缓存至 `data/osm_cache/`（water, shops, emergency 等）。

---

## 10. 有效脚本清单

只列产生了实质结果的脚本。其余文件夹中的 .py（PPT 版本、迭代中间版、放弃方向）未收录。

### 数据转换

| 脚本 | 输出 | 说明 |
|------|------|------|
| `convert_flow_geojson_to_parquet.py` | `flow_parquet2/` | Flow GeoJSON → slot parquet |
| `convert_incidents_geojson_to_parquet.py` | `incident_parquet/` | Incidents GeoJSON → parquet |

### 路网覆盖与消失分析

| 脚本 | 输出 | 说明 |
|------|------|------|
| `脚本_08_路网覆盖可视化.py` | 图16 | OSM vs TomTom 双图 |
| `脚本_09_数据覆盖质量图.py` | 图17 | 覆盖完整度地图（五类着色） |
| `脚本_19_路网覆盖分布图.py` | 图19 | 覆盖率分类地图（更精细版） |
| `脚本_20_路段消失演变图.py` | 图18 | 消失分析五面板 |
| `脚本_21_消失率重算.py` | `disappearance_rates.pkl` | 各信号期消失率表 |
| `脚本_22_消失路分布图.py` | 图22 | 消失路空间分布 |

### 建成环境特征

| 脚本 | 说明 |
|------|------|
| `脚本_25_屋苑宽表.py` | 人口属性 |
| `脚本_26_道路人口特征.py` | 500m 缓冲人口附加 |
| `脚本_27a_事故特征.py` | incident 数聚合 |
| `脚本_27b_POI特征.py` | OSM POI 密度 |
| `脚本_27c_道路结构特征.py` | 结构特征 |
| `脚本_64_school_elderly_features.py` | 学校/长者设施 |
| `脚本_28b_更新回归表.py` | `regression_table.parquet`（最终版） |

### Pre-Event 分析

| 脚本 | 输出 | 说明 |
|------|------|------|
| `脚本_01_台风前行为分析.py` | 早期时序图 | 发现 09-23 早高峰清空和午间逆转 |
| `脚本_47_速度形状对比_米娜vs叶加沙.py` | 对比图 | 验证 midday reversal 是 Ragasa 特有 |
| `脚本_51_对照工作日share.py` | 对照基准数字 | 正常工作日 F/N/S 自然分化率 |
| `脚本_53_长度加权share表.py` | 表格 | 长度加权 F/N/S 占比 |
| `脚本_55_两时段地图.py` | 图 | 08:30 / 13:00 路段状态地图 |
| `脚本_56_transition_matrix.py` | 图50f/g | 转移矩阵 + 空间分布 |
| `脚本_59_binary_logit.py` | 回归结果 | 早高峰清空 / 午间变慢二元 logit |
| `脚本_59d_morning_clearance_regression.py` | 系数图 | Morning clearance 最终版 + 图 |
| `脚本_66_morning_clearance_3panel.py` | 图50d | 清空三面板地图 |
| `脚本_67_midday_churn_3panel.py` | 图50e | 逆转三面板地图 |

### During-Event 分析

| 脚本 | 输出 | 说明 |
|------|------|------|
| `脚本_02_速度极化分析.py` | — | 剂量-响应关系，基线拥堵五分位分析 |
| `脚本_06_信号升级前行为与五分位分析.py` | 图13 | 五分位天花板效应 |
| `脚本_49_during_speed_shape.py` | — | 三台风 during 速度形态 |
| `脚本_52_Ragasa_pre_signals.py` | 图25d | Ragasa 全程时序含信号着色 |
| `脚本_52b_Ragasa_morning_compare.py` | 图25e | 早高峰 S3/S10/基线对比 |
| `脚本_52d_evening_S8_compare.py` | 图25f | 晚高峰 rising/falling S8 对比 |
| `脚本_62_heterogeneity_panel.py` | 图25h | 三天 × 六时段 × 五分位面板 |

### Post-Event 分析

| 脚本 | 输出 | 说明 |
|------|------|------|
| `脚本_03_台风后恢复分析.py` | 图06/07/08 | 三台风恢复曲线、道路类别恢复速度 |

### 应急可达性

| 脚本 | 输出 | 说明 |
|------|------|------|
| `脚本_63_emergency_accessibility.py` | 图25j | 屋苑应急可达性分析 |

---

## 11. 后续研究可以做什么

### 立即可做（数据和代码都准备好了）

1. **Cox PH 模型**：`regression_table.parquet` 里所有变量都在，从 `脚本_03` 输出提取路段级恢复时间后即可跑。见 7.4 节的 spec。

2. **修复 10-07 基线分类 bug**：核查基线计算逻辑是否把 10-07（中秋节翌日）误分为 WORKDAY，修复后重跑所有分析确认结果变化幅度。

3. **残余 disruption 的 incident overlay**：恢复最慢的路段列表（`脚本_03` 输出）× incident 数据 500m 空间连接，判断是路坏了还是需求回流。见 7.3 节。

### 需要更多数据的方向

4. **多年比较**：现在只有 2025 年数据。拿到 2023、2024 年同期数据可做跨年比较，验证结论稳健性。

5. **OD 层面分析**：现在的分析单元是路段，但需求变化本质上是 OD 层面的。如果能获取 OD 流量数据（手机信令），可以区分"起点减少出行"vs"路段被绕行"。

6. **台风预期与实际信号的分离**：市民的出行调整可能早于官方信号（知道会挂 S8 就提前不出门）。预期效应很难从现有数据中单独分离。

7. **跨城市比较**：TomTom 在台北、厦门、深圳等有台风的城市也有数据，是自然的扩展方向。

### 方法论上还可以探索的

8. **分位数回归**：对"最慢的那 10% 路段"或"最快的那 10% 路段"的分析可能比均值回归更有政策意义。

9. **机器学习方法**：AUC 0.62 不高，Random Forest / XGBoost 在特征选择和非线性关系上可能有改善空间。

---

## 12. 技术坑与注意事项

- **`road_closure` 过滤**：`!= 1` 不是 `== 0`，未关闭是 NaN 不是 0
- **`wkb_hash` 不能跨脚本用**：Python `hash()` 不稳定，只用 ep_key
- **台风期间未观测道路填 NaN 不填 0**：填 0 会压低均值
- **09-22 大缺口**：Ragasa S1 期间（05:00–20:30）32 槽完全没有数据，pre-event 分析从 S3 开始
- **10-07 节假日**：中秋节翌日可能被误分为 WORKDAY，待核查
- **道路过滤阈值**：回归用 `max_days_seen ≥ 3`，可按需调整
- **Ragasa 的中文名**：文件和脚本里有时叫"叶加沙"，是同一台风的另一翻译

---

*Sun Ling，HKU MDUM 2024–25，2025 年 6 月*
