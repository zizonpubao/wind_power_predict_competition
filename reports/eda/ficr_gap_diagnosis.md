# FICR Gap Diagnosis — tuned LightGBM run `20260720_161850_lgbm_tuned`

**Author**: evaluator agent | **Date**: 2026-07-21 | **Run diagnosed**: `experiments/20260720_161850_lgbm_tuned`

## 0. Method

CLAUDE.md §5 flags FICR as a step-function metric (nMAE tiers: ≤6% → 4원/kWh, 6–8% →
3원/kWh, >8% → 0원). This run scores 1-NMAE=0.857 (decent average error) but FICR=0.294
(much lower) — this report investigates why, using genuine out-of-fold (OOF) predictions
rather than the in-sample final-refit model.

**OOF reconstruction**: for each of the 3 groups, refit `GroupLGBMModel` per CV fold using
the tuned `best_params` from `experiments/20260720_161850_lgbm_tuned/tuning_results.json`
(no `n_estimators` override — default 2000 + `early_stopping_rounds=50`, exactly matching
`tune_hyperparams._cv_score`), over the same 5-fold `BlockTimeSeriesSplit` used in training.
Each fold's held-out validation predictions were concatenated into one OOF series per group.

**Sanity check**: the reconstructed per-fold `score`/`1-NMAE`/`FICR` matched
`experiments/20260720_161850_lgbm_tuned/metrics.json`'s `fold_metrics` **exactly** (diff =
0.0000 on every fold/group, expected since `random_state=42` is fixed and the CV/params are
identical) — confirms the OOF reconstruction faithfully reproduces the reported CV, and the
predictions used below are genuinely held-out, not in-sample.

**Coverage caveat**: `BlockTimeSeriesSplit(n_splits=5)` cuts the sorted forecast blocks into
6 chunks; chunk 0 is only ever used as training data (never validated), so the OOF series
covers roughly the later 5/6 of each group's time-ordered rows, not the very earliest blocks.
OOF row counts: group_1 21,810 (of 26,200), group_2 21,811 (of 26,201), group_3 14,610 (of
17,538).

---

## 1. Tier distribution among eligible hours (actual ≥ 10% capacity)

| Group | n eligible / n OOF | ≤6% (4원) | 6–8% (3원) | >8% (0원, no settlement) |
|---|---|---|---|---|
| kpx_group_1 | 12,916 / 21,810 (59.2%) | 27.66% | 8.88% | **63.46%** |
| kpx_group_2 | 12,879 / 21,811 (59.0%) | 29.02% | 9.43% | **61.54%** |
| kpx_group_3 | 7,559 / 14,610 (51.7%) | 26.74% | 8.08% | **65.18%** |

Roughly **62–65% of every group's eligible hours earn zero settlement**. Only ~27–29% reach
the top tier. This confirms the FICR gap is not a few outliers — it's the majority case.

## 2. How far into the >8% tier are the misses? (percentile breakdown of nmae_h, in %)

| Group | p10 | p25 | p50 (median) | p75 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|---|
| kpx_group_1 | 9.5 | 11.9 | **17.0** | 24.5 | 33.5 | 39.5 | 52.7 | 83.5 |
| kpx_group_2 | 9.4 | 11.9 | **16.9** | 24.9 | 34.3 | 41.0 | 58.2 | 77.7 |
| kpx_group_3 | 9.7 | 12.6 | **18.9** | 27.2 | 36.2 | 41.4 | 53.1 | 83.9 |

Bucketed as a share of the >8% population:

| Group | "close" 8–10% | "mid" 10–15% | "far" 15–25% | "wild" >25% |
|---|---|---|---|---|
| kpx_group_1 | 13.4% | 27.6% | 35.0% | 24.0% |
| kpx_group_2 | 13.7% | 27.9% | 33.8% | 24.7% |
| kpx_group_3 | 11.7% | 23.7% | 33.8% | 30.9% |

**Key finding**: the median hour in the no-settlement tier misses by 17–19 percentage
points, not a hair over the 8% line. Only ~12–14% of the >8%-tier hours (≈7.6–8.5% of *all*
eligible hours — see `pct_eligible_hours_in_8_10pct_band` below) are "near-miss" (8–10%
band); the other 86–88% would need much bigger error reductions to cross the threshold.
**This is a smooth-loss-vs-tiered-metric gap that mostly needs genuinely lower error, not a
small nudge near the boundary** — though the near-miss band is still worth harvesting (see §5
counterfactual).

Counterfactual: if every 8–10%-band hour were pushed into the ≤6% tier (best case), FICR
would rise by +0.077 (g1), +0.083 (g2), +0.061 (g3); a more modest push into the 6–8% tier
(3원) only would still add +0.058 / +0.062 / +0.046. That's a real but limited upside (~0.05–0.08
FICR points) relative to the ~0.55–0.65 gap between current FICR and 1.0.

## 3. Why do large errors happen? Worst 5% of eligible hours by nmae_h

Hypotheses checked: cut-out-adjacent wind variance, low-wind ramp events, seasonality,
lead_hour. Compared the worst 5% (n=646 / 644 / 378) against the rest.

**Ruled out / weak**:
- **Near cut-out (wind speed >18 m/s, approaching the ~22–22.5 m/s cut-out)**: 0% of worst-5%
  hours in all 3 groups — LDAPS wind speeds in this dataset essentially never approach
  cut-out, so this is not a contributor.
- **Wind-speed uncertainty (rolling std)**: `corr(nmae_h, ldaps_10m_speed_idw_roll_std_6h)` is
  only 0.04 (g1), 0.04 (g2), 0.11 (g3) — weak. High-variance/uncertain forecast periods are
  **not** a strong driver of large errors.
- **Wind speed itself**: correlation with nmae_h is weak/inconsistent (0.03, -0.02, 0.21).
- **Ramp zone (3–8 m/s, steep cubic part of the power curve)**: ~71–80% of both worst-5% and
  the general eligible population sit in this zone (it's just where most operating hours
  are), and mean |nmae_h| in-zone vs out-of-zone is similar (14.1% vs 13.4% g1; 14.1% vs
  12.8% g2; 14.1% vs 17.9% g3 — g3 is actually *worse* outside the ramp zone). So the
  power-curve-nonlinearity-amplifies-error hypothesis is not well supported either.
- **lead_hour**: only a mild shift (worst-5% mean 12.4–13.1h vs rest 12.2–12.3h) — consistent
  with CLAUDE.md §4's prior finding of no strong lead-hour degradation.

**Confirmed / strong**:
- **Systematic under-prediction bias, worst at high generation.** Across *all* eligible hours
  (not just the worst 5%), the model underpredicts (`pred < actual`) 62–66% of the time, with
  a consistently negative mean signed error: **−1,246 kWh (g1), −721 kWh (g2), −1,207 kWh
  (g3)** on capacities of 21,600/21,600/21,000 kWh. The worst-5% hours have far higher mean
  actual generation than the rest (g1: 13,848 vs 10,390 kWh; g3: 14,159 vs 9,686 kWh; g2 flat
  at ~11,200 both), and 75% of g1's and g3's worst-5% hours (57% of g2's) are underpredictions.
  **The model is systematically compressing predictions toward the mean and under-calling
  high-generation events** — a classic gradient-boosting regression-to-the-mean effect,
  likely reinforced by the tuned hyperparameters' regularization (reg_alpha up to 1.5,
  subsample down to 0.54–0.74, num_leaves as low as 12 for group_3) which was selected purely
  to maximize mean CV `competition_score`, with no explicit penalty for this asymmetry.
- **Mild seasonality**: worst-5% hours are overrepresented in December (13.7–15.6% vs
  15.0–16.0% rest — comparable) and **July** (10.7–18.3% vs 8.6–11.6% rest — clearly
  overrepresented in all 3 groups) and November for group_3 (17.5% vs 11.1%). Matches
  CLAUDE.md §4's known winter/summer seasonality in the label–SCADA gap.
- **Mild fold-recency effect**: worst-5% errors skew toward earlier CV folds (less training
  data) for group_1 (fold 2+3 = 57% of worst-5% vs uniform 40%) and group_3 (fold 1+2 = 54%
  vs uniform 40%), weaker for group_2. Consistent with models trained on less history being
  somewhat worse — doesn't directly apply to the production full-refit model (trained on all
  rows) but suggests more training history would help.

## 4. Feature importance (gain-based, from the 3 tuned `model_<group>.joblib`)

All 3 groups use the same 141-column feature set. Top signal in every group: wind
u/v-components (multiple heights/grids), `ldaps_10m_speed_mean`/`_idw`, `gfs_dswrf`
(shortwave radiation — likely acting as a time-of-day/atmospheric-stability proxy), and
`ldaps_blh` (boundary layer height). LDAPS features dominate over GFS at the very top for
group_2/3, consistent with CLAUDE.md §4's note that LDAPS (1.5km) correlates with generation
more strongly than GFS (0.25°).

**Top 10 by gain, kpx_group_1**: `gfs_80m_u_idw` (27.3%), `gfs_100m_u_idw` (17.6%),
`ldaps_10m_speed_mean` (9.9%), `gfs_10m_u_idw` (8.8%), `ldaps_10m_u_idw` (7.2%),
`gfs_dswrf_nearest` (2.9%), `ldaps_10m_speed_idw_roll_mean_6h` (2.6%), `gfs_dswrf_idw`
(2.4%), `ldaps_blh_idw` (1.2%), `ldaps_t2m_nearest` (0.9%). Top 4 features alone = 63.6% of
total gain.

**Top 10 by gain, kpx_group_2**: `ldaps_10m_u_nearest` (16.6%), `ldaps_10m_speed_idw_roll_mean_3h`
(14.8%), `ldaps_10m_u_idw` (12.2%), `gfs_100m_u_idw` (6.8%), `ldaps_10m_power_curve_idw`
(4.3%), `ldaps_10m_speed_idw` (4.2%), `gfs_80m_u_idw` (3.5%), `ldaps_10m_power_curve_idw_roll_mean_3h`
(3.5%), `gfs_dswrf_nearest` (2.8%), `gfs_dswrf_idw` (1.7%).

**Top 10 by gain, kpx_group_3**: `ldaps_10m_speed_idw_roll_mean_3h` (21.9%),
`ldaps_10m_power_curve_idw_roll_mean_3h` (12.7%), `ldaps_10m_speed_idw` (12.5%),
`ldaps_10m_power_curve_idw` (9.2%), `ldaps_10m_u_nearest` (8.8%), `ldaps_10m_speed_idw_roll_mean_6h`
(3.9%), `gfs_100m_u_idw` (3.8%), `gfs_10m_v_nearest` (3.3%), `gfs_80m_u_idw` (2.8%),
`gfs_10m_v_idw` (1.8%).

**Long tail / pruning candidates**: features needed to reach 95% cumulative gain: **54/141
(g1)**, **73/141 (g2)**, **48/141 (g3)**. So roughly 47–66 near-zero-value features per group
(mostly the `*_power_curve_idw_roll_{mean,std}_{3h,6h,12h}` GFS-derived rolling features and
`gfs_80m/100m_power_curve_idw` raw columns) sit in a long tail contributing <0.05% gain each.
Only 2 features (group_3) have literally zero gain, but the practical tail is much larger
than that — these are strong candidates for pruning to reduce the ~141-features/~20-26k-rows
overfitting risk noted in `src/models/lgbm_model.py`'s own docstring.

Full per-feature CSVs (gain, split count, gain_pct, sorted) were generated for all 3 groups
during this analysis but are intermediate working files, not checked into the repo.

## 5. Prioritized recommendations

1. **Diagnose and correct the systematic under-prediction bias first — highest-leverage,
   most concrete finding.** 62–66% of eligible hours are underpredicted, with mean signed
   error consistently negative and worst at high-generation hours. This alone likely explains
   more of the FICR gap than the tier-boundary mechanics. Try: OOF-residual-based bias
   correction/calibration (e.g. isotonic regression of pred→actual, or a simple multiplicative
   correction conditioned on predicted-generation bucket) as a cheap post-processing fix;
   separately, re-examine whether the Optuna search's pure `competition_score` objective is
   pushing hyperparameters (heavy `reg_alpha`/low `subsample`/low `num_leaves` for group_3)
   toward mean-shrinkage as a side effect, and consider constraining regularization ranges or
   adding a bias term to the tuning objective.
2. **Try an asymmetric/custom training loss that penalizes crossing the 6%/8% nMAE
   thresholds directly** (e.g. a loss with a steep penalty jump past 8%, or quantile/pinball
   loss favoring slight over- rather than under-prediction) — directly targets the metric this
   competition actually pays for, rather than optimizing smooth MAE/L1 as a proxy.
3. **Feature pruning to roughly the top 50–75 gain features per group** (matching the 95%
   cumulative-gain cutoffs found in §4), then re-tune. Given the ~140:20k feature:row ratio,
   this is likely to reduce overfitting/variance more than it costs in signal — cheap to try
   and directly testable via the same CV harness already in `tune_hyperparams.py`.
4. **Don't chase the wind-variance/near-cut-out/ramp-zone hypotheses further** — all three
   were checked directly against OOF data and showed weak or no effect. Time is better spent
   on the bias-correction and loss-function angles above.
5. **Harvest the near-miss (8–10% nMAE) band as a secondary, lower-effort win**: only
   7.6–8.5% of eligible hours sit there, worth +0.05 to +0.08 FICR if fully fixed — real but
   small next to the ~0.55–0.65 point gap to a perfect score. Don't over-invest here relative
   to item 1.
6. **Follow up on the July/December seasonal skew** in worst-error hours (already flagged
   generally in CLAUDE.md §4's label-SCADA seasonality finding) — worth a short eda-analyst
   pass to check whether month/season-interaction features would help, before spending model
   time on it.
