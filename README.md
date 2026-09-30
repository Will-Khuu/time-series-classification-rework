# Activity recognition from wireless sensor data

This project classifies human activities (bending, cycling, lying, sitting, standing,
walking) from the signal strength between three wireless sensors worn on the body. It uses
the [UCI AReM dataset](https://archive.ics.uci.edu/dataset/366/activity+recognition+system+based+on+multisensor+data+fusion+arem).

It began as two course assignments. This repo reworks them into a tested Python package and
re-checks their results. Coursework pipelines often report numbers that don't hold up, and
this one is no exception.

> **Work in progress.** The data layer is done. Features, evaluation, and results come next.
> This README will report measured accuracy, with baselines and confidence intervals, once
> those steps land.

## Findings so far

- **The dataset has 81 recordings, not 88.** Seven files are exact or near-exact copies of
  other files, six in `lying` and one in `cycling`. `data/check_independence.py` finds them
  by comparing every pair of files.
- **The original test set leaked.** Under the assignments' train and test split, 4 of the 19
  test recordings have an exact copy in the training set. Their test accuracy is partly
  memorization.
- **Every recording comes from one person.** No result here can claim to generalize to new
  people.

[`data/README.md`](data/README.md) has the details, including the file format problems the
loader handles.

## Run it

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt -e .
.venv/bin/python data/download.py
.venv/bin/python -m pytest
```

`data/download.py` fetches the dataset from UCI and verifies its checksum.

## Layout

- `data/` holds the download script, the duplicate check, and the dataset notes.
- `src/arwsn/loading.py` parses all recordings into one table and drops the copies.
- `src/arwsn/features.py` turns each recording into one row of segment statistics.
- `tests/` checks the loader and features against the original notebook code on every file.
- `scripts/verify.py` runs every check and writes [`reports/verification.md`](reports/verification.md).
