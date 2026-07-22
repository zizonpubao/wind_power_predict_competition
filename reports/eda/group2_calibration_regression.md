# Why isotonic OOF calibration hurts kpx_group_2 — root-cause investigation

**Author**: eda-analyst | **Date**: 2026-07-22 | **Run analyzed**: `experiments/20260721_205339_lgbm_tuned`
(retuned on this machine; not the original `20260720_161850_lgbm_tuned` referenced in
`ficr_gap_diagnosis.md`/the calibration commit, which no longer exists locally)

## 0. Method

Re-ran `src.training.evaluate_calibration` (read-only — no models/artifacts touched, no
`--save-calibrators`) against the current tuned run, then wrote a standalone diagnostic script
(reusing `tune_hyperparams._oof_predict` and `PredictionCalibrator` unchanged) that:
1. Reconstructs genuine OOF `(pred, actual)` pairs per group (same 5-fold `BlockTimeSeriesSplit`
   the tuning/evaluation pipeline already uses).
2. Fits the pooled ("production") isotonic curve and all 5 leave-one-fold-out ("cross-fit")
   curves per group, and compares them.
3. Buckets bias two ways — by **predicted**-value decile (what the isotonic curve actually
   trains on) and by **actual**-value quintile restricted to eligible hours (`actual >= 10%
   capacity`, what the competition score actually cares about) — since these can disagree.

Intermediate OOF parquets and the summary CSV are scratch files, not committed; the only new
repo artifact is `reports/eda/group2_calibration_curve_comparison.png` (per-fold correction
curves, one panel per group).

## 1. Reproduced the regression

The reported CV score delta reproduces on this run (small numeric differences vs. the original
commit's numbers are expected — the tuned hyperparameters differ slightly since this run was
retuned on this machine — but the **pattern is identical**):

| Group | score (raw→calib) | Δscore | mean signed err, eligible hrs (raw→calib) | tier ≤6% (raw→calib) |
|---|---|---|---|---|
| kpx_group_1 | 0.5736 → 0.5865 | **+0.0129** | −1236.1 → −806.2 (+429.9) | 27.74% → 27.90% |
| kpx_group_2 | 0.6012 → 0.5915 | **−0.0097** | −717.0 → **−813.5 (−96.6)** | 29.02% → **27.67%** |
| kpx_group_3 | 0.5525 → 0.5546 | **+0.0020** | −1197.3 → −794.6 (+402.6) | 26.86% → 25.64%* |
| OVERALL | 0.5758 → 0.5775 | +0.0018 | | |

(*group_3's tier≤6% share drops too even though its score improves overall — the fine-grained
tier bucketing is noisy at n≈7.5k; score/FICR are the reliable summary here.)

Group_2 is the only group where calibration pushes the eligible-hour mean signed error **further
from zero** (−717 → −813.5), not toward it — this is the mechanism, not just a side effect: the
1-NMAE and FICR both worsen for group_2 (Δ1-NMAE −0.0032, ΔFICR −0.0161) and every settlement
tier moves the wrong way (≤6% share drops 1.36pp, >8% share rises 1.33pp).

## 2. Why: two consistent, non-overlapping lines of evidence

### 2.1 Group_2's raw bias is genuinely smaller **and pointing a different way in the region the isotonic curve is fit on**

Bucketing OOF rows by **predicted**-value decile (the isotonic curve's actual training axis),
signed error = `pred_mean − actual_mean`:

| Pred decile (≈top range) | kpx_group_1 | kpx_group_2 | kpx_group_3 |
|---|---|---|---|
| 6 | −576.4 | −174.4 | −25.4 |
| 7 | −745.4 | **−54.5** | −406.9 |
| 8 | −484.0 | **+240.3** | −1142.8 |
| 9 (highest) | −609.9 | **+331.8** | −547.6 |

Group_1 and group_3 under-predict (need a large **positive/upward** isotonic correction) at
every high-pred decile. Group_2 is the outlier: by decile 8–9 it's already **over-predicting**
(+240, +332) — the exact opposite sign. This one difference is the whole story: the same
isotonic-fit machinery, applied uniformly, learns "add ~500–700 kWh in the high-pred region" for
group_1/3 (which is exactly what's needed) but learns "**subtract** a few hundred kWh in the
high-pred region" for group_2.

But the competition score is evaluated conditional on **actual** generation, not predicted
generation. Bucketing the same OOF rows by actual-value quintile among *eligible* hours only
(`actual >= 10% capacity`) tells the opposite-looking story — group_2 *also* under-predicts at
high actual generation, just less severely than group_1/3:

| Actual quintile (eligible) | raw err g1 | calib err g1 | raw err g2 | calib err g2 | raw err g3 | calib err g3 |
|---|---|---|---|---|---|---|
| Q3 (high) | −2261.8 | **−1707.8** (better) | −2052.4 | **−2169.7** (worse) | −2213.9 | **−1575.7** (better) |
| Q4 (highest) | −4345.0 | **−3775.6** (better) | −3050.0 | **−3297.6** (worse) | −5332.4 | **−4684.8** (better) |

This is not a contradiction — it's the crux of the problem. Isotonic regression correctly learns
`E[actual | pred]`, which for group_2's top *pred* decile really is "actual tends to be a bit
lower than this prediction" (decile 8/9 above). But the eligible-hour scoring metric conditions
on **actual**, not pred, and among genuinely-high-actual-generation hours group_2 still
under-predicts (Q3/Q4 raw err still negative, just smaller than g1/g3's). Because group_1/3's
pred-conditional and actual-conditional views happen to agree in direction (both say "add" at
the high end), calibration helps both simultaneously. Group_2's two views **disagree in sign**
at the very top, so the (statistically valid, but pred-conditional) isotonic correction pushes
predictions the wrong way for the specific high-actual-generation hours the score cares about.

**Ruled out**: this is not a noisier pred↔actual relationship for group_2 — `corr(pred, actual)`
in the top predicted decile is actually *better* for group_2 (0.274) than group_1 (0.297) or
group_3 (0.227), and `std(actual | top-10%-pred)` is *smaller* for group_2 (3,422) than group_1
(4,032) or group_3 (4,382). It is also not a data-volume/history difference — group_2's OOF
sample (n=21,811) is essentially identical in size to group_1's (n=21,810); group_3's shorter
history (n=14,610, no 2022 labels) is irrelevant to this specific group_2-vs-others comparison.

### 2.2 Fold-to-fold isotonic curve instability is highest for group_2 relative to its own signal

Comparing each group's 5 leave-one-fold-out isotonic curves against the pooled curve, at a
common grid of predicted values (see
[`group2_calibration_curve_comparison.png`](./group2_calibration_curve_comparison.png)):

| Group | mean\|pooled correction\| (kWh) | fold-curve std (kWh, "noise") | noise / signal ratio |
|---|---|---|---|
| kpx_group_1 | 500.3 | 172.4 | **0.34** |
| kpx_group_2 | 251.2 | 193.5 | **0.77** |
| kpx_group_3 | 487.6 | 232.0 | **0.48** |

Group_2 needs the smallest correction of the three (251 kWh mean absolute correction — roughly
half of group_1/3's) but the fold-to-fold disagreement about *where exactly* that correction
should sit is almost as large as the correction itself (ratio 0.77 vs. 0.34–0.48). In other
words: group_2's isotonic curve is fitting a genuinely weak, noisy signal rather than a robust
systematic bias, so a curve fit on 4 folds generalizes far less reliably to the 5th held-out
fold than it does for group_1/3 — consistent with the leave-one-fold-out cross-fit evaluation
finding a net-negative effect specifically for this group.

### 2.3 A plausible mechanistic driver: group_2's independently-tuned hyperparameters land in a much less regularized regime

From `experiments/20260721_205339_lgbm_tuned/config.yaml`'s `tuned_params_per_group` (each
group's Optuna search is fully independent, no shared params):

| Group | reg_alpha | reg_lambda | subsample | num_leaves |
|---|---|---|---|---|
| kpx_group_1 | 1.275 | 0.0065 | 0.664 | 49 |
| kpx_group_2 | **0.001** | 0.0020 | **0.922** | 53 |
| kpx_group_3 | 1.487 | 0.0169 | 0.567 | 24 |

Group_2's search landed on essentially **no L1 regularization** (reg_alpha 0.001 vs. 1.28–1.49
for the others) and **almost no subsampling** (0.922 vs. 0.57–0.66) — i.e., markedly less
mean-shrinkage than group_1/3's tuned models. This is consistent with `ficr_gap_diagnosis.md`'s
hypothesis that heavy regularization reinforces the regression-to-the-mean under-prediction
pattern: group_2's own tuned model already sits in a regime with much less of that compression
artifact, leaving correspondingly less genuine bias for a post-hoc calibrator to fix — and what
residual "signal" exists (§2.1/§2.2) is comparatively noisy and, per §2.1, points the wrong way
in the specific high-actual-generation zone the score weights most.

## 3. Side finding (unrelated to group_2 specifically, but blocks any fix here)

`src.training.tune_hyperparams.tune_group()` computes a production `PredictionCalibrator` per
group (line ~293) but **`main()` never calls `.save()` / `joblib.dump()` on it** — only
`evaluate_calibration.py --save-calibrators` does that, and only when invoked separately.
Confirmed: `experiments/20260721_205339_lgbm_tuned/` contains no `calibrator_*.joblib` files.
**Calibration is therefore not currently active in this run's submission at all**, despite the
CLAUDE.md/commit-message description of it being "wired into production." Any fix to the
group_2 issue (§4 below) needs to land alongside actually persisting calibrators from
`tune_hyperparams.py` per group, or the whole feature stays inert.

## 4. Recommended next actions

1. **Make calibration a per-group decision, not a single global on/off switch.** The current
   code only checks the *overall* mean-of-3-groups delta before saving any calibrator. Change
   `evaluate_calibration.py` (and whatever `tune_hyperparams.py` fix restores persistence) to
   save `calibrator_<group>.joblib` **only for groups whose own cross-fit delta is positive**
   (group_1, group_3 here) and skip it for group_2 given the consistent negative delta shown
   above. This is a small, low-risk, well-evidenced change.
2. **Don't try to force group_2 into the same calibration approach with a smaller learning rate
   fix** (e.g. shrinking the correction by some blend factor) as a first step — §2.1 shows the
   direction of the pred-conditional signal is actually opposite to what the actual-conditional
   (eligible-hour) evaluation needs at the high end, so a smaller dose of the same wrong-signed
   correction still hurts, just less. If group_2's calibration is revisited later, prefer fitting
   the isotonic curve restricted to eligible-hour rows only (conditioning closer to what's
   scored) rather than all OOF rows, and re-validate with the same leave-one-fold-out harness
   before trusting it.
3. **Fix the tune_hyperparams.py persistence gap (§3) regardless** — right now no calibrator is
   active in the latest run's submission for any group, silently. This should be corrected in
   the same change as item 1 so group_1/group_3 actually get the validated +0.013/+0.002 benefit.
4. **Longer term**: since group_2's tuned hyperparameters landed in a qualitatively different
   regularization regime than group_1/3 purely from independent per-group Optuna search (not by
   design), consider whether `ficr_gap_diagnosis.md`'s suggestion of constraining the
   regularization search range or adding an explicit bias term to the tuning objective would
   make the 3 groups' post-tuning bias characteristics more homogeneous — which would make a
   single shared calibration policy safer to apply uniformly in the future, rather than needing
   permanent per-group exceptions.

## Artifacts

- [`group2_calibration_curve_comparison.png`](./group2_calibration_curve_comparison.png) — per-group
  panels showing the pooled (production) isotonic correction curve vs. all 5 leave-one-fold-out
  curves, as a function of raw predicted value.
