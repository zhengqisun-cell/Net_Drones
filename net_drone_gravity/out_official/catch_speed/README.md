# 转移提速（catch_speed）文件索引与复现命令

> **注意（2026-11-23）：**本目录存档的结果都是修复前（boxfix 关）跑的。规划器出界和高度下限的修复 `boxfix` 现在默认开启。
> - **按原命令重跑：**触发过兜底的个别试验结果会不同。
> - **逐位复现存档：**在设置文件或 `--catch-limits` 里加 `"boxfix": false`。
> - **修复前后对比：**见 `../catch_boxfix/`。

## 文件
- `settings/catch_speed_x1.json`, `settings/catch_speed_x5.json`：推荐设置（5 倍要加 `--thrust-scale 5`）
- `settings/catch_speed_x1_aggressive.json`, `settings/catch_speed_x5_aggressive.json`：激进设置（破线，不推荐）
- `compare_final.json/.txt/.png`：正式对比，种子 1–30，每组 30 次，含分子、分母、被排除的球、安全项
- `final/`：正式批次的原始汇总（final = 首选和激进，final2 = S 曲线单项，final3 = 1 倍推荐）
- `ablation.json/.txt/.png`：15 种子筛选（种子 101–115），每项单独和组合；原始数据在 `screening/round_*.json`
- `p2p/p2p_before_after.json/.txt/.png`：点到点前后对比；`p2p/p2p_shape*.json`：轨迹形状和加加速度；
  `p2p/p2p_accel_map.json` 与 `accel_direction_clearance.png`：加速度 × 方向 → 最小间隙（C1）；
  `p2p/p2p_spacing_hook.json`：转移时改间距（C2）和挂点下移 20 mm（E）
- `offcentre_drop/drop_offset.json/.txt`：偏心落球试验（B）。偏心接球开关 `offcentre` 的筛选结果在 `screening/round_r1.json`（B_r0.2、B_r0.3）。
  boxfix 默认开之后，这些存档结果要加 `"boxfix": false` 才能逐位复现。新默认下，一倍、难档 2 s、种子 111（B_r0.2）新出现了一次破线（clearance + ball_rotor，桨盘 26.6 mm），见 `../catch_boxfix/README.md` 的"已知个例"。
- `reach_x1/reach/`, `reach_x5/reach/`：推荐设置的单球可达图（原设置见 `../catch/reach`、`../thrust5/catch/reach`）
- `videos/x1_hard_seed8/compare_x1_hard_seed8.mp4`, `videos/x5_hard_seed21/compare_x5_hard_seed21.mp4`：
  同种子前后对比（左原设置，右新设置）；每侧 `base/` `new/` 下有单独的视频、5 张关键帧和试验日志

## 命令（在 net_drone_gravity 下）
```
python net_sim.py --mode catch --tier hard --seed 8 --catch-limits out_official/catch_speed/settings/catch_speed_x1.json
python net_sim.py --mode catch --tier hard --seed 21 --thrust-scale 5 --catch-limits out_official/catch_speed/settings/catch_speed_x5.json
python net_sim.py --mode catch --catch-study single --tier hard --catch-limits out_official/catch_speed/settings/catch_speed_x1.json --out <目录>
python net_sim.py --mode catch --catch-study single --tier medium --interval 1 --thrust-scale 5 --catch-limits out_official/catch_speed/settings/catch_speed_x5.json --out <目录>
python net_sim.py --mode catch --catch-study single --launch salvo --catch-limits out_official/catch_speed/settings/catch_speed_x1.json --out <目录>
python net_sim.py --mode catch --catch-study reach --catch-limits out_official/catch_speed/settings/catch_speed_x1.json --out <目录>
python net_sim.py --mode catch --seed 101 --catch-limits "{\"p2p\": {\"d\": [2.0, 0.0, 0.0]}, \"traj\": {\"jerk\": 30.0}}" --out <目录>
```
`--catch-study single` 默认种子 1–30。点到点基准表用种子 101，距离 0.5/1/1.5/2/3 m，方向 0° 和 45°。
