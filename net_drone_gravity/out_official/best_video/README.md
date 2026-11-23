# 最好的一段接球视频（不含网模型规划）

保存于 2026-11-23。使用当前默认规划器：`netdyn` 关闭，`boxfix` 默认开启。

## 视频
`x5_tier0_hard2s_seed206/catch_hard_sequence_seed206_video.mp4`

- **条件：**5 倍推力，0 档安全线（最严），难档，间隔 2 s，连发 10 个球，种子 206。
- **结果：**10/10 全部接住，各项安全线都没破，没有触发刹停。
  - 旋翼盘最小间距 33.1 mm，护圈 28.0 mm；
  - 最大倾角 48.65°，电机最大用量 22%；
  - 网最低点 0.60 m，离边界最近 0.87 m。
- **为什么选它：**种子 1–30 和 201–230 的 10 球终测里，难档只有这一次全接；它在 0 档和 1 档安全线下都是 10/10，这里存的是更严的 0 档。

同目录还有这次试验的日志（`trials/`）和 5 张关键帧。

## 复现
```
python net_sim.py --mode catch --tier hard --interval 2 --seed 206 --thrust-scale 5 --catch-limits out_official/catch_10balls/settings/x5_tier0_10balls.json --out out_official/best_video/x5_tier0_hard2s_seed206
python render_scene.py --log out_official/best_video/x5_tier0_hard2s_seed206/trials/catch_hard_sequence_seed206_log.npz --video --out out_official/best_video/x5_tier0_hard2s_seed206
```

## 注意
- 这是最好的一次，不是典型水平：同一组（0 档、难档、间隔 2 s）平均每次接住 5.8 个（种子 201–230，本视频所在的一批）和 6.6 个（种子 1–30）。
- 5 倍推力是假设电机推力放大而质量不变，偏乐观；网的参数是假设值；没有模拟旋翼下洗气流。
