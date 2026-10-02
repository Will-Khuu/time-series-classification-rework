# Verification report

**Overall: PASS.** Regenerate with `python scripts/verify.py`.

- Generated: 2026-10-02 17:37 UTC
- Code fingerprint: `322effb3884ec4b2` (SHA-256 of every `.py` file in `src`, `tests`, `data`, `scripts`)
- Checked on top of commit `c965612`; the fingerprint identifies the exact code
- Python 3.12.14 on Darwin arm64
- numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1, pytest 9.1.1

## Test suite: PASS

76 passed, 0 failed, 0 skipped.

| Test file | Passed | Failed | Skipped |
|---|---|---|---|
| `tests/test_evaluation.py` | 15 | 0 | 0 |
| `tests/test_features.py` | 46 | 0 | 0 |
| `tests/test_loading.py` | 11 | 0 | 0 |
| `tests/test_models.py` | 4 | 0 | 0 |

## Dataset integrity: PASS

`data/download.py` file and row counts: all match (88 files across 7 folders).

## Duplicate recordings: PASS

`data/check_independence.py` found 8 duplicate pairs covering 7 copied files.
They match the `COPIES` constant in `src/arwsn/loading.py`.

## Reproduction of the course version: PASS

Reran the course version's method on its 69/19 split (`results/metrics.json`, part `reproduce`).

| Task | Per-l CV scores vs course version (20 each) | Test result | Course version test result |
|---|---|---|---|
| binary | max difference 0.0000 | 19/19 | 1.000 |
| multiclass | max difference 0.0005 | 17/19 | 0.895 |

The course version printed multiclass scores to 3 decimals, so differences up to 0.0005 are rounding.

## README numbers match the results: PASS

| Number computed from `results/metrics.json` | In README |
|---|---|
| 74/81 (91.4%), 95% CI 83.2% to 95.8% | yes |
| 79/81 (97.5%), 95% CI 91.4% to 99.3% | yes |
| 77/81 (95.1%), 95% CI 88.0% to 98.1% | yes |
| 17/19 (89.5%), 95% CI 68.6% to 97.1% | yes |
| 19/19 (100.0%), 95% CI 83.2% to 100.0% | yes |
| 18.5% | yes |
| 84.0% | yes |

## Mutation checks: PASS

| Deliberate bug | File | Caught by |
|---|---|---|
| forget one duplicate file | `src/arwsn/loading.py` | `test_counts` and 1 more |
| parse commas only, losing the space-separated file | `src/arwsn/loading.py` | `test_counts` and 9 more |
| change the notebook split sizes | `src/arwsn/loading.py` | `test_legacy_split_reproduces_notebook` |
| order features segment-first, as the notebook's names wrongly assumed | `src/arwsn/features.py` | `test_values_match_notebook[2]` and 20 more |
| use population std instead of sample std | `src/arwsn/features.py` | `test_values_match_notebook[1]` and 20 more |
| compute the wrong first quartile | `src/arwsn/features.py` | `test_values_match_notebook[1]` and 19 more |
| let held-out recordings into selection and fitting (the leak nested CV prevents) | `src/arwsn/evaluation.py` | `test_naive_max_is_optimistic_on_noise` and 1 more |
| break the Wilson interval formula | `src/arwsn/evaluation.py` | `test_wilson_known_values[19-19-0.832-1.0]` and 5 more |
| break ties toward the largest l | `src/arwsn/evaluation.py` | `test_ties_go_to_smallest_l` |
| ignore groups, so copies can split across folds | `src/arwsn/evaluation.py` | `test_groups_keep_copies_in_one_fold` and 2 more |
| match groups to recordings by position instead of by id | `src/arwsn/evaluation.py` | `test_groups_matched_by_id_not_position` |
