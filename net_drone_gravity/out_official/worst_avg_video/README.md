# 同一条件下最差和平均的两段视频

保存于 2026-11-23。条件与 `out_official/best_video/` 相同：5 倍推力，0 档安全线，难档，间隔 2 s，连发 10 个球；默认规划器（`netdyn` 关闭，`boxfix` 开启）。

## 怎么挑的
同一条件共 60 次试验（种子 1–30 和 201–230，结果在 `out_official/catch_netdyn/runs/final_*/x5_T0_10__base__hard2.json`）：

| 接住数 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|
| 次数 | 2 | 6 | 13 | 13 | 15 | 7 | 3 | 1 |

平均 6.2 个，中位数 6 个。

- **最差：种子 221，3/10。**接住 3 个的有种子 215 和 221；221 还有一个球打到无人机（破了球碰机体这条安全线），所以取它。
- **平均：种子 6，6/10。**接住 6 个的有 13 次，取其中没破任何安全线、种子号最小的一次。

## 视频
| | 文件 | 接住 | 漏掉的原因 | 安全线 |
|---|---|---|---|---|
| 最差 | `x5_tier0_hard2s_seed221_worst/catch_hard_sequence_seed221_video.mp4` | 3/10（第 1、2、10 个） | 6 个来不及，1 个打到无人机（第 6 个） | 破 1 条：球碰无人机 |
| 平均 | `x5_tier0_hard2s_seed6_average/catch_hard_sequence_seed6_video.mp4` | 6/10 | 4 个来不及（第 4、6、7、8 个） | 都没破 |

两次重跑的结果都与批量里的一致。两次都没有中止、没有出界。

## 复现
把 `--seed` 和输出目录换成对应的值：
```
python net_sim.py --mode catch --tier hard --interval 2 --seed 221 --thrust-scale 5 --catch-limits out_official/catch_10balls/settings/x5_tier0_10balls.json --out out_official/worst_avg_video/x5_tier0_hard2s_seed221_worst
python render_scene.py --log out_official/worst_avg_video/x5_tier0_hard2s_seed221_worst/trials/catch_hard_sequence_seed221_log.npz --video --out out_official/worst_avg_video/x5_tier0_hard2s_seed221_worst
```

## 注意
5 倍推力是假设电机推力放大而质量不变，偏乐观；网的参数是假设值；没有模拟旋翼下洗气流。
