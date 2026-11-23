#!/bin/sh
# Robustness (D): baseline and the selected settings, hard tier 2 s, seeds 1-30, net mass +20 % / strings +-10 mm
# (the planner's model unchanged). Resumable. Run from net_drone_gravity.
P=out_official/catch_netdyn/scripts/nd_runs.py
for R in 'robust_mnet {"m_net":0.024}' 'robust_strings {"string_each":[0.21,0.2,0.19,0.2]}'; do
  T=${R%% *}; O=${R#* }
  python $P $T 1 30 x1_T0,x5_T1,x5_T1_10 base,ff_ah+ groups=hard_seq2,hard2 "p_over=$O"
  python $P $T 1 30 x5_T0 base,ff groups=hard_seq2 "p_over=$O"
  python $P $T 1 30 x5_T0_10 base,nd groups=hard2 "p_over=$O"
done
