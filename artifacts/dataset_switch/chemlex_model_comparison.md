# Chemlex amide: all PU methods vs the HistGB champion (TEST ROC-AUC, 5 seeds)

Ranked by transductive/raw_ecfp. Delta = method - champion in that cell.

| model | transductive/raw_ecfp | transductive/corr08 | inductive/raw_ecfp | inductive/corr08 | Δ vs champ |
|---|---|---|---|---|---|
| prop_ips | 0.795±0.005 | 0.730±0.003 | 0.784±0.002 | 0.774±0.001 | +0.024 |
| elkan | 0.779±0.006 | 0.724±0.004 | 0.750±0.006 | 0.729±0.007 | +0.008 |
| bagging | 0.774±0.002 | 0.727±0.002 | 0.770±0.001 | 0.746±0.001 | +0.004 |
| sup_histgb_l2 (champion) | 0.771±0.003 | 0.726±0.005 | 0.767±0.002 | 0.745±0.001 | +0.000 |
| sar_em | 0.747±0.005 | 0.727±0.005 | 0.761±0.001 | 0.753±0.002 | -0.024 |
| spy | 0.741±0.002 | 0.716±0.002 | 0.713±0.005 | 0.702±0.009 | -0.030 |
| iw_pu | 0.644±0.001 | 0.640±0.001 | 0.679±0.002 | 0.672±0.002 | -0.127 |
| nnpu | 0.644±0.001 | 0.640±0.001 | 0.679±0.002 | 0.672±0.002 | -0.127 |
