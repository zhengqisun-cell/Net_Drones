# 规划器出界修复（catch_boxfix）

> **自 2026-11-23 起 `boxfix` 默认开启。**
> - **关闭方法：**在设置文件或 `--catch-limits` 里写 `"boxfix": false`，例如 `--catch-limits '{"boxfix": false}'`。关闭后与修复前的代码逐位一致。
> - **字典写法照旧可用：**例如 `{"boxfix": {"floor_tol_m": 0.0}}`，表示开启并改掉其中几项。
> - **旧目录的存档：**catch_speed、catch_speed_relaxed、catch_10balls 等目录存档的结果都是修复前（boxfix 关）跑的。按原命令重跑时，触发过兜底的个别试验结果会不同；要逐位复现存档，需加 `"boxfix": false`。
> - **设为默认时的核对：**见 `default_on_check.txt`；已知个例见文末。

## 问题
新路径会出场地（编队盒子）时，规划器的兜底是"先停到前方一点"。原实现有两个缺陷：
1. **停点带竖直速度：**停点 = 当前参考 + 速度 × 刹车距离，包含竖直分量，编队在下降时会被送得更低。
2. **停点每次重算：**每次重规划（20 Hz）都从仍在移动的参考重新算停点，停点不断前移，编队停不下来，可以冲出场地几米。

此外，路径检查有 5 cm 容差，而且完全不看高度下限。读代码时还发现了第三个问题：
3. **参考可远低于高度下限，且不经过兜底：**齐射、难档里刚接住第 1 个球、正在下降时重规划，新路径（带向下的速度）会下探到网心高度下限以下 0.1–0.42 m。例如一倍档 0 齐射种子 27，网最低到 0.278 m，破了 0.30 m 线。

## 开关 `boxfix`（2026-11-23 起默认开；`CATCH_BOXFIX` 是各项参数）
关闭：`--catch-limits '{"boxfix": false}'`，或在设置文件里写 `"boxfix": false`。只有 false、null 或 0 才关闭；字典（包括空字典）表示开启，并覆盖其中各项，例如 `{"boxfix": {"floor_tol_m": 0.0}}`。本目录的修复前后对比是在默认改变之前跑的：`before` 用的是 `"boxfix": false`（当时是默认值），`after` 用的是 `"boxfix": true`。

| 项 | 默认 | 作用 |
|---|---|---|
| `horizontal_stop` | true | 停点只取水平方向；高度保持当前高度，但不低于高度下限 |
| `latch` | true | 一旦开始刹车，就保持这个刹车计划直到停稳，期间不重规划 |
| `keep_plan` | true | 刹车路径本身也会出场地时（离墙太近、速度太快），保留当前计划；当前计划在制定时已检查过不出场地，下一次重规划再试 |
| `tol_m` | 1e-4 | 路径出场地检查的容差：0.1 mm，原来是 5 cm。正好为 0 时，目标被夹在场地面上，舍入误差会让每条新路径都被判出界，编队卡死 |
| `floor` | true | 新路径也检查高度下限：网心不低于 `z_net_range[0]`（0.7 m）、按网里球数折算的编队中心高度；参考当前高度或上一个目标更低时取更低者 |
| `floor_tol_m` | 0.10 | 只拒绝下探超过 0.10 m 的路径，被拒绝时保留当前计划到下一次重规划，不刹车 |

**为什么 `floor_tol_m` 不取 0：**带向下速度的重规划总会下探几厘米，取 0 会频繁保留旧计划，丢球。例如 catch 中档种子 5：一倍 4/5 → 3/5，五倍 5/5 → 3/5；种子 1–30 上各组少 1–4 个球。取 0.10 m 时这些都与修复前一致，参考最多比下限低 9.7 cm，网最低仍在 0.49 m 以上。

**曾试过、已放弃的做法：**下限违规也触发刹车并锁定。中档 2 s 每组多出 2–10 次刹车，10 个球时少接 4–7 个，已放弃。

## 新的直接检查（每次试验的摘要 `reference`，开关开或关都有）
- `out_of_box_max_m`：规划器的编队中心参考点超出场地（编队盒子）的最大距离。
- `below_floor_max_m`：低于当时高度下限的最大深度。
- 兜底次数：
  - `n_brake`：刹车兜底调用；
  - `n_brake_latched`：其中被锁定的；
  - `n_brake_kept_plan`：刹车也出界、改为保留计划的；
  - `n_floor_kept_plan`：因下限保留计划的。

## 文件
- `compare_boxfix.txt/.json`：修复前后的逐组对比。
  - 10 个球、五倍的 8 组，种子 1–30；
  - 3–5 个球的标准四组：五倍档 0、档 1，一倍档 0，种子 1–30；
  - 逐试验有变化的种子；
  - "两次都没触发任何兜底的试验中有几次变化"（全部为 0）。
- `default_on_check.txt`：2026-11-23 设为默认时的核对表。
- `compare_pilot.txt/.json`：先导（种子 101–115），修复前取 `../catch_10balls/runs/pilot_101-115.json`。
- `runs/`：逐试验记录（`before` 为开关关闭的重跑，与之前存档的结果逐个比对一致）。
  - `main10_1-30.json`；
  - `std_1-30.json`；
  - `pilot_101-115.json`（先导，只开修复）。
- `settings/`：开修复的设置文件，原设置文件未改。2026-11-23 默认改为开启后，这些文件与对应的原设置文件等效，原文件不带 `boxfix` 键，现在也默认开修复。
  - `x5_tier{0,1}_10balls_boxfix.json`；
  - `x5_tier{0,1}_boxfix.json`、`x1_tier0_boxfix.json`（3–5 个球）。
  - `x5_tier0_10balls_boxfix_off.json`：原 10 个球档 0 设置加 `"boxfix": false`，用于逐位复现修复前的存档。
- `videos/tier0_hard2s_seed30/compare_tier0_hard2s_seed30.mp4`：档 0、难档 2 s、种子 30 的前后对比，左为修复前，右为修复后。`before/` 和 `after/` 下有单独的视频和关键帧；`after/trials` 下有日志。
- `scripts/`：
  - `runbf.py`：批量；
  - `compbf.py`：对比表；
  - `compose.py`：并排视频。

## 复现命令（在 net_drone_gravity 下）
```
python net_sim.py --mode catch --tier hard --seed 30 --thrust-scale 5 --catch-limits out_official/catch_boxfix/settings/x5_tier0_10balls_boxfix.json
python net_sim.py --mode catch --tier hard --seed 30 --thrust-scale 5 --catch-limits out_official/catch_10balls/settings/x5_tier0_10balls.json
python net_sim.py --mode catch --tier hard --seed 30 --thrust-scale 5 --catch-limits out_official/catch_boxfix/settings/x5_tier0_10balls_boxfix_off.json
python out_official/catch_boxfix/scripts/runbf.py pilot
python out_official/catch_boxfix/scripts/runbf.py main10
python out_official/catch_boxfix/scripts/runbf.py std
python out_official/catch_boxfix/scripts/compbf.py
```
`runbf.py main10 after`（或 `std after`）只重跑开修复的一半，关修复的结果保留。

## net_sim.py 的改动
- **新增：**`CATCH_DEFAULT["boxfix"]`（2026-11-22 加入时为 False，2026-11-23 起为 True）、`CATCH_BOXFIX`；`boxfix` 已加入起飞后快照的键。
- **CatchPlan：**`_go` 里加了修复分支；新增兜底计数和下限记录（`zlo_hist`、`goal_z`），已加入快照状态。
- **分析：**新增 `catch_reference_check`，摘要新增 `reference` 键。
- **回归：**开关关闭时与修复前的代码逐位一致：1 m hold --ff、mission seed 1 = 45.81 s、5× mission seed 1 = 46.12 s、catch 中档 seed 5 = 4/5、5× seed 5 = 5/5。开关打开时，catch seed 5 的两个基准也逐位一致（hold 和 mission 不经过这段代码）。2026-11-23 改为默认开以后又核对了一遍，结果见 `default_on_check.txt`。

上面第 2 条命令（原设置文件）现在默认开修复，得修复后的结果：5/10，不出界。第 3 条加 `"boxfix": false`，得修复前的结果：2/10，冲出场地 3.35 m。

## 已知个例
- **先导种子 114（档 0、难档 2 s、五倍、10 个球）：**开修复后，一次"低于下限、保留计划"改变了之后的路径，26.96 s 时倾角达到 60.0°，试验中止，接住 5 → 2。先导这一组（种子 101–115）的总接住为 86 → 84（发射 150 → 145），8 组合计 791 → 791。见 `compare_pilot.txt`。
- **偏心接球种子 111（offcentre r=0.2、一倍、难档 2 s，即 catch_speed 第 1 轮筛选里的 B_r0.2 组合）：**"低于下限、保留计划"触发 7 次后，球 2 撞机。
  - **新增破线：**`clearance` 和 `ball_rotor`。
  - **对比修复关时：**

    | | 修复关 | 默认（修复开） |
    |---|---|---|
    | 桨盘最小间隙 | 34.0 mm | 26.6 mm |
    | 护圈最小间隙 | 29.9 mm | 14.4 mm |
    | 最大倾角 | 39.9° | 56.9° |
    | 球 2 | 甩出（fell_out） | 撞机（hit_drone） |

  - **接住数：**都是 1/4。种子 101–115 这一组的总接住为 23 → 24。
  - **适用范围：**offcentre 默认不开，也不在任何推荐设置里。
- **移动接球种子 26（catch_height 的 x5_moving_full、五倍、难档 2 s）：**修复关时破桨盘线（29.8 mm，倾角 52.1°）；默认开修复后为 31.4 mm、倾角 42.8°，不再破线。这一组（种子 1–30）的总接住为 89 → 89。
