#!/bin/sh
# Final batches (resumable: finished chunks are skipped). Run from net_drone_gravity.
P=out_official/catch_netdyn/scripts/nd_runs.py
for S in "1 30 final_1-30" "201 230 final_201-230"; do
  set -- $S
  python $P $3 $1 $2 x1_T0,x5_T1 base,ff_ah+,nd
  python $P $3 $1 $2 x5_T0 base,ff,nd
  python $P $3 $1 $2 x5_T0_10 base,nd
  python $P $3 $1 $2 x5_T1_10 base,ff_ah+,nd
done
