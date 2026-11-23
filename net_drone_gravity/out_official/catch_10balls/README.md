# 连续 10 个球，五倍推力（catch_10balls）

> **注意（2026-11-23）：**本目录存档的结果都是修复前（boxfix 关）跑的。规划器出界和高度下限的修复 `boxfix` 现在默认开启。
> - **按原命令重跑：**触发过兜底的个别试验结果会不同。
> - **逐位复现存档：**在设置文件或 `--catch-limits` 里加 `"boxfix": false`。
> - **修复前后对比：**见 `../catch_boxfix/`。

## 球数固定为 10 和种子的对应关系
- **球数：**`--n-balls 10 10`（命令行），或在 `--catch-limits` 文件里写 `"n_balls": [10, 10]`。
- **`count_draw`（新开关，默认 None）：**设为 `[3, 5]` 时，先按默认的 3–5 个球抽一次球数再丢掉，然后用 `n_balls` 的固定值。
  - 原因：不加这个开关时，固定 10 个球后所有随机抽取都会错位，种子 1–30 的球在中档和难档都与原来完全不同（逐一核对过，0/129 相同）。
  - 加了以后，每个种子的前 n 个球（n 为该种子原来的 3–5 个）仰角、方位、速度和发射时刻都与原来相同。
  - 只有发射管位置不同：管排按球数居中，10 个球时管排宽 0.54 m，所以同一个球左右偏移最多 0.27 m。
- **10 个球的设置文件：**`settings/x5_tier{0,1}_10balls.json` 已包含上面两项，以及下面的 `pile_catch` 和 `h_dn_table`。

## 落球试验（`drop/`）
- **方法：**编队悬停，规划器关闭，每 3 s 从网心上方 1 m 静止释放 1 个球，共 10 个；每个状态取下一个球释放前 1 s 的平均。
- **结果：**`drop/drop10.json`，含 5 倍和 1 倍（1 倍只作参考）。

| 网里球数 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 无人机到网心下沉增量（mm） | 0 | 51.2 | 71.2 | 83.6 | 90.8 | 96.9 | 102.2 | 106.5 | 109.7 | 113.4 | 115.8 |
| 网垂度（m） | 0.318 | 0.370 | 0.391 | 0.405 | 0.412 | 0.419 | 0.425 | 0.429 | 0.433 | 0.437 | 0.439 |
| 平均倾角（°） | 17.1 | 19.4 | 21.5 | 23.5 | 25.4 | 27.1 | 28.7 | 30.1 | 31.6 | 32.9 | 34.1 |
| 悬停电机平均占用，5 倍 | 0.114 | 0.117 | 0.120 | 0.124 | 0.128 | 0.131 | 0.135 | 0.139 | 0.143 | 0.147 | 0.151 |
| 最低网结离地（m） | 0.794 | 0.732 | 0.718 | 0.708 | 0.702 | 0.700 | 0.697 | 0.691 | 0.687 | 0.683 | 0.683 |

- **悬停电机占用，1 倍参考：**从 0.569 增加到 0.756。
- **间隙：**桨盘间隙全程 35.1 mm，护圈间隙 37.8–41.0 mm。
- **下沉表：**前 5 项与原表 `CATCH_DH_TABLE` 相差 ≤0.6 mm，所以 0–5 沿用原值，6–10 用实测值。完整的 11 项写在 10 个球的设置文件里（`h_dn_table`），规划器瞄准高度用它。默认表没有改，以免改变默认摘要。
- **重量前馈：**按网里实际球数线性增加（`ff_balls`），原来就适用于任意球数。
- **球叠放：**第 10 个球落在其他球上面，没有碰到网。原判据只认网接触，会把它算成"没接住"。新开关 `pile_catch`（默认关）：靠在网中球上的球（球–球接触链、随网一起运动、最后 1 s 内 ≥95% 时间如此）也算接住，摘要里标 `caught_on_pile`。开了这个开关后，落球试验 10/10。

## 接球高度下限 0.7 m 和"网不过低"
- 落球试验里，10 个球时最低网结比网心只低约 0 mm，网心离地 0.68 m。下限 0.7 m 指网心目标高度，距 0.30 m 线余量约 0.4 m。
- 调参种子 101–115 上 120 次试验，只有 1 次低到 0.291 m。所以下限和待命高度都没改。
- 种子 1–30 上有 1 次网低到 0.059 m，不是因为球多，而是规划器的一个旧问题，见报告和 `investigate/`。

## 其他检查
- **仿真时长：**最后一个球之后 14 s，随球数自动延长。
- **"接住"判据：**最后 1 s 与网接触 ≥95%、相对速度 <0.1 m/s；另加 `pile_catch`。
- **渲染：**`render_scene.py` 的球颜色扩到 10 种，原有 7 种不变。图例 10 行；状态里新增被砸出、被甩出、弹出、叠放接住。
- **掉出网的球（`loss` 字段，只做分析）：**每个摸过网但最后没接住的球分三类：
  - 被砸出（knocked_out）：曾在网里，另一个球首次碰网后 0.6 s 内离开；
  - 被甩出（thrown_out）：曾在网里，离开时没有新球到达；
  - 弹出（bounced_out）：从未在网里停住。

## 测试
- **推力：**`--thrust-scale 5`。
- **设置：**两套原样使用，未调参。
  - 档 0：`catch_speed/settings/catch_speed_x5.json` → `settings/x5_tier0_10balls.json`；
  - 档 1：`catch_speed_relaxed/settings/tier1_x5.json` → `settings/x5_tier1_10balls.json`。
- **分组：**中档和难档，顺序发射，间隔 2 s 和 1 s，种子 1–30，每次 10 个球。
- **先导：**种子 101–115，`runs/pilot_101-115.json`、`summary_pilot_101-115.txt`。先导里没有发现必须为 10 个球改的量。

## 文件
- `summary_main_1-30.txt/.json`：主表。每组列出：
  - 接住数/发射数，全接试验数，≥8 个的试验数，每次接住数的分布；
  - 按发射顺序和按到达时网里球数的接住率；
  - 漏球分类、掉出网的球、叠放接住；
  - 该档安全线的破线数和各安全项；
  - 前 3–5 个球与原 3–5 个球批次的对比。
- `catch_by_order_main_1-30.png`：上面两幅是按发射顺序的接住率（左 2 s，右 1 s），下面两幅是按到达时网里已有球数的接住率。
- `runs/main_1-30.json`：逐试验、逐球记录。
- `videos/tier1_medium2s_seed1/`：档 1，中档 2 s，种子 1，10/10。该组中位数就是 10，所以这一段同时就是全接的视频。
- `videos/tier1_hard2s_seed8/`：档 1，难档 2 s，种子 8，7/10（该组中位数）。
- `investigate/tier0_hard2_seed30/`：档 0，难档 2 s，种子 30。编队冲出场地、网降到离地 6 cm 的那次，含日志和视频。
- `scripts/`：
  - `drop10.py`：落球试验；
  - `run10.py`：批量；
  - `compile10.py`：制表和作图。
  - `run10.py` 用到 `../catch_speed_relaxed/scripts/drv2.py`（分档判据）。

## 复现命令（在 net_drone_gravity 下）
单次试验：
```
python net_sim.py --mode catch --tier medium --seed 1 --thrust-scale 5 --catch-limits out_official/catch_10balls/settings/x5_tier1_10balls.json
python net_sim.py --mode catch --tier hard --seed 8 --thrust-scale 5 --catch-limits out_official/catch_10balls/settings/x5_tier1_10balls.json
python net_sim.py --mode catch --tier hard --seed 30 --thrust-scale 5 --catch-limits out_official/catch_10balls/settings/x5_tier0_10balls.json
```

批量和制表（会覆盖本目录中的结果）：
```
python out_official/catch_10balls/scripts/drop10.py 5 1
python out_official/catch_10balls/scripts/run10.py main_1-30 1 30
python out_official/catch_10balls/scripts/compile10.py main_1-30
```

间隔 1 s 的一组 30 个种子：
```
python net_sim.py --mode catch --catch-study single --tier hard --interval 1 --thrust-scale 5 --catch-limits out_official/catch_10balls/settings/x5_tier1_10balls.json --out <目录>
```

## net_sim.py 和 render_scene.py 的改动（默认行为不变，回归逐位一致）
- **`net_sim.py`：**
  - `CATCH_DEFAULT` 新增 `count_draw=None`、`pile_catch=False`；
  - 新增 `catch_pile_support`、`catch_ball_losses`，每个掉出网的球多一个 `loss` 字段，只做分析；
  - `CATCH_DH_TABLE` 加注释，数值不变。
- **`render_scene.py`：**颜色扩到 10 种；新增掉出类型和叠放接住的状态文字；"网里球数"把叠放接住的球也算进去。
