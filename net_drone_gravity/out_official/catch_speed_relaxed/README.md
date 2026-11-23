# 放宽安全线换转移速度（catch_speed_relaxed）

> **注意（2026-11-23）：**本目录存档的结果都是修复前（boxfix 关）跑的。规划器出界和高度下限的修复 `boxfix` 现在默认开启。
> - **按原命令重跑：**触发过兜底的个别试验结果会不同。
> - **逐位复现存档：**在设置文件或 `--catch-limits` 里加 `"boxfix": false`。
> - **修复前后对比：**见 `../catch_boxfix/`。

## 安全线分档（数值由协调会话定，用户未给具体值）

| 档 | 电机线 | 桨盘线 | 说明 |
|---|---|---|---|
| 0 | 90 % | 30 mm | 现有线；设置沿用 `../catch_speed/settings/catch_speed_x1.json`、`catch_speed_x5.json` |
| 1 | 95 % | 25 mm | |
| 2 | 98 % | 20 mm | |
| 3 | 无电机线 | 15 mm | 只统计饱和（指令 ≥ f_max）的累计时长和最长一次 |

**电机线判据：**超线连续 > 0.10 s 算破线，与现有 90 % 规则同一形式。纯峰值判据另列为 `vPk` 列。

**各档都不放宽的条件：**
- 不中止：倾角 > 60° 即中止；
- 护圈间隙 ≥ 10 mm；
- 球不撞机；
- 网不过低；
- 无人机间距；
- 绳不松弛；
- 不出界。

**球–桨盘间隙：**各档都保持原来的 30 mm 判据，因为仿真没有下洗气流，2.7 g 的球靠近桨盘的风险被低估了。主表另有一列 `vBr`，是这项按该档桨盘数值放宽后的破线数。这是协调会话在正式批次之后提出的修正；用修正后的判据在 101–130 上重新套用选择规则，选出的 6 套设置不变，所以没有重跑。

**档 0 的判据：**直接用分析里的精确标志。摘要中的间隙四舍五入到 0.1 mm，例如 29.96 mm 显示为 30.0。

## 选择规则（看种子 1–30 之前定好，没有改选）
- **调参种子：**101–130，四组共 448 个球、120 次试验。
- **规则：**在该档破线试验 ≤ 6/120（5 %）的设置里，取接住最多的一个；接住数相同时取破线少的。
- **记录：**规则和备选见 `settings/selection.txt`。
- **新种子复核：**选定后在种子 1–30 跑了正式批次，又在全新种子 201–230 上把同样 8 套设置原样跑了一遍。没有改选，201–230 只是第二个无偏样本。

## 文件
- `settings/tier{1,2,3}_x{1,5}.json`：各档选定的设置。用法为 `--catch-limits`，五倍推力另加 `--thrust-scale 5`。
- `main_table.txt/.json`：主表，种子 1–30，档 × 推力 × 四组。
- `main_table_seeds201-230.*`：同一张表，种子 201–230。
- `tiers_caught.png`：安全线档位 → 接住率。共三个面板：难档 2 s、中档 1 s、四组合计。实线为种子 1–30，虚线为 201–230。
- `final/`：正式批次的原始汇总，含逐试验记录：电机各线超线时长、饱和、间隙、倾角、中止原因等。
- `robust/robust_table.txt/.json`：稳健性。种子 1–30 的难档 2 s 组，比较名义、网重 +20 %（20 → 24 g）、吊绳 ±10 mm 三种情况。吊绳 ±10 mm 指 0 号机 +10 mm、2 号机 −10 mm。
- `p2p/p2p_tiers.txt/.json`：点到点表，种子 101，距离 0.5–3 m，方向 0° 和 45°。
- `reach/reach_tiers.txt/.json`：单球可达图，每套设置 180 格。
- `screening/`：调参轮次的原始数据（`round_q*.json`）和变体定义（`q*.json`）；`screening_summary.txt` 是各档破线数的汇总。`round_q*.txt` 里的档 0 计数用的是取整后的间隙，以 `screening_summary.txt` 为准。
- `videos/x{1,5}_hard_seed19/compare_x{1,5}_hard_seed19.mp4`：同种子左右对比，左为档 0，右为档 2。两侧各有单独视频、5 张关键帧和试验日志，位于 `tier0/`、`tier2/`。
- `scripts/`：批量驱动和制表脚本，只调用 net_sim 的函数。
  - `drv2.py`：分档判据 `tier_viol`。
  - `screen2.py`：调参。
  - `final_r.py`：正式批次、稳健性、点到点、可达图。
  - `compile2.py`：制表和作图。

## 复现命令（在 net_drone_gravity 下）
单次试验：
```
python net_sim.py --mode catch --tier hard --seed 19 --catch-limits out_official/catch_speed_relaxed/settings/tier2_x1.json
python net_sim.py --mode catch --tier hard --seed 19 --thrust-scale 5 --catch-limits out_official/catch_speed_relaxed/settings/tier2_x5.json
```

种子 1–30，一组（摘要含 `safety.motor90.lines` 和饱和时长，可按档复算）：
```
python net_sim.py --mode catch --catch-study single --tier hard --catch-limits out_official/catch_speed_relaxed/settings/tier1_x1.json --out <目录>
python net_sim.py --mode catch --catch-study single --tier medium --interval 1 --thrust-scale 5 --catch-limits out_official/catch_speed_relaxed/settings/tier3_x5.json --out <目录>
python net_sim.py --mode catch --catch-study single --launch salvo --catch-limits out_official/catch_speed_relaxed/settings/tier2_x1.json --out <目录>
```

稳健性单次试验：在设置文件里加 `"p_over": {"m_net": 0.024}`，或加 `"p_over": {"string_each": [0.21, 0.2, 0.19, 0.2]}`。

批量重跑和制表（脚本把结果写回本目录，会覆盖现有文件）：
```
python out_official/catch_speed_relaxed/scripts/final_r.py final 1 30
python out_official/catch_speed_relaxed/scripts/final_r.py final 201 230
python out_official/catch_speed_relaxed/scripts/final_r.py robust
python out_official/catch_speed_relaxed/scripts/final_r.py p2p
python out_official/catch_speed_relaxed/scripts/final_r.py reach
python out_official/catch_speed_relaxed/scripts/compile2.py
python out_official/catch_speed_relaxed/scripts/compile2.py final_seeds201-230 _seeds201-230
python out_official/catch_speed_relaxed/scripts/screen2.py q2 out_official/catch_speed_relaxed/screening/q2.json 101 130
```

## net_sim.py 的改动（默认行为不变，回归逐位一致）
- **`safety.motor90` 的三个新统计键（只做分析）：**
  - `lines`：超过 90/95/98 % 各线的累计时长和最长连续时长；
  - `saturated_total_s`；
  - `saturated_longest_s`。
- **`Params.string_each`：**类属性，默认 None，表示每根吊绳分别取长度；已加入快照键。
