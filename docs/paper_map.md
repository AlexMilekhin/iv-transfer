# Where each result comes from

All files are written to `<root>/results/paper/` by `python -m ivtransfer paper` (and `trading`, `figures`). Every number quoted in
the text is also in `paper_numbers.csv` (columns: section, item, market, period, value, how) and, for Section 8,
`paper_numbers_trading.csv`. Periods: `2023-25` = test years 2023, 2024 and 2025 pooled; `2026` = the holdout (January to mid-October).
Scores: `primary` = forecast at t against the target starting at the next grid date, on well-measured name-years; `same-day` =
target starting at t, all names.

| Result | Output file | Code |
|---|---|---|
| Table 1. Markets and samples | `table1_counts.csv` | `tables.table1` |
| Table 2. Universe rules | rules in `markets/*.py`; universe files `universe/<m>.parquet` | `markets/*.py`, `universe.build` |
| Table 3. Model inputs | `config.ALL17`, `config.LEAN12` + seasonal input | `state.samples`, `train.season_input` |
| Table 4. Variants | | `train.train_targets`, `evaluate.Evaluation.combine` |
| Table 5. Walk-forward folds | | `train.masks` |
| Table 6. Gain over the local model (Panels A and B; Panel C = `local A %`) | `table6_gains.csv` | `evaluate.Evaluation.tables` |
| Table 7. Model chosen on the validation year | `table7_chosen.csv` | `evaluate.Evaluation.tables` |
| Table 8. Networks and simple benchmarks | `q5_skill_wide.csv`, `q5_baselines_skill.csv`; intervals `q5_bootstrap.csv` | `tables.Robustness.benchmarks` |
| Table 9. Permutation importance of the term slope | `table9_importance.csv` | `tables.permutation_importance` |
| Table 10. Single seeds | `table10_single_seeds.csv`, `table10_summary.csv` | `tables.single_seeds` |
| Table A1. Changes tested on validation years | not reproduced: the tuning runs are not part of this repository | |
| Table A2. Validation skill, initial and final networks | `appendix_a_validation.csv` | `tables.validation_skills`, `tables.initial_network_items` |
| Table C1. Block-bootstrap intervals | `bootstrap_gains.csv` (and `bootstrap_market_differences.csv`) | `tables.bootstrap` |
| Table C2. 2026 comparisons, Benjamini-Hochberg | `c2_2026_bh.csv` (all Diebold-Mariano statistics: `c1_diebold_mariano.csv`) | `evaluate.Evaluation.tables` |
| Table C3. Score decomposition | `c3_score_decomposition.csv` | `tables.score_decomposition` |
| Figure 1. Network | `docs/figure1.png` | `model.MoE` (3,464 parameters) |
| Figure 2. Gains with intervals | `figure2.png` | `figures.figure2` |
| Section 4.1 roll check | `roll_check.csv`, `states/<m>/state_stats.json` | `tables.roll_check`, `state.build` |
| Section 4.2 largest 1% of targets | `top1_share.csv` | `tables.top1_share` |
| Section 4.6 / 6.3 well-measured names | `well_measured.csv` | `tables.data_quality` |
| Section 5.1 US model at home | `us_home.csv` | `evaluate.Evaluation.tables` |
| Section 6.2 training curves | `training_curves.csv` | `tables.training_curves` |
| Section 6.3 retention | `retention.csv` | `tables.retention` |
| Section 6.4 output scales | `output_scales.csv` | `tables.output_scales` |
| Section 7.1 / 7.5 initial network, calibration | `initial_calibration.csv` | `tables.initial_network_items` |
| Section 7.2 gradient boosting | `table6_gains.csv` (`GB gain`), `gb_trees_k.csv` | `evaluate.gb_fold` |
| Section 7.3 leave-one-seed-out | `loo_best_variant.csv`, `loo_by_variant.csv` | `tables.robustness_items` |
| Section 7.6 accuracy over time | `skill_2026_by_month.csv`, `target_drift.csv`, `q6_skill_by_year.csv` | `tables.diagnostics_2026`, `tables.Robustness.by_year` |
| Section 7.7 expiry listings | `paper_numbers.csv` (section 7.7) | `tables.listing_check` |
| Section 8. Economic value | `trading_v1_2023-25.csv`, `trading_v1_2026.csv`, `paper_numbers_trading.csv` | `trading.evaluate` |
| Robustness: strict liquidity, equal weights, one name left out, surface fits | `q1_strict_subsets.csv`, `q2_equal_weight.csv`, `q3_leave_one_name_out.csv`, `q4_surface_fits.csv` | `tables.Robustness` |
| Reproduction gate | `gate_final.csv`, `gate_initial.csv` | `evaluate.Evaluation.forecasts_market` |
