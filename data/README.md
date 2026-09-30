# AReM dataset

Activity Recognition system based on Multisensor data fusion, from the
[UCI Machine Learning Repository, dataset 366](https://archive.ics.uci.edu/dataset/366/activity+recognition+system+based+on+multisensor+data+fusion+arem).
Licensed CC BY 4.0.

> Palumbo, F., Gallicchio, C., Pucci, R., & Micheli, A. (2016). Activity Recognition system
> based on Multisensor data fusion (AReM) [Dataset]. UCI Machine Learning Repository.
> https://doi.org/10.24432/C5SS33

## Get the data

```bash
python data/download.py
```

The script downloads the UCI zip, checks its SHA-256, extracts it to `data/AReM/`, and
verifies the file and row counts below. It needs only the standard library. A verified
copy is left alone on rerun. `data/AReM/` is not committed.

## What was measured

One actor wore three IRIS wireless nodes, on the chest and on each ankle. The nodes
measured the received signal strength (RSS) of packets exchanged between each pair at
20 Hz. Every 250 ms, the five samples in that window become one row: the mean and the
variance of RSS for each node pair.

| Column | Meaning |
|---|---|
| `time` | Milliseconds since the recording started, in 250 ms steps |
| `avg_rss12`, `var_rss12` | Chest to right ankle |
| `avg_rss13`, `var_rss13` | Chest to left ankle |
| `avg_rss23`, `var_rss23` | Right ankle to left ankle |

The UCI page calls the second statistic "variance" in one place and "standard deviation"
in another. This project does not depend on which one it is.

## Layout

The folder name is the label. Each `datasetN.csv` is one 120-second recording, and `N` is
the repetition number.

| Folder | Files | Unique recordings |
|---|---|---|
| `bending1` | 7 | 7 |
| `bending2` | 6 | 6 |
| `cycling` | 15 | 14 |
| `lying` | 15 | 9 |
| `sitting` | 15 | 15 |
| `standing` | 15 | 15 |
| `walking` | 15 | 15 |
| **Total** | **88** | **81** |

`bending1` and `bending2` are two different bending postures. `bendingType.pdf` in the
extracted data shows both.

## Known problems

**Seven files duplicate other files.** `python data/check_independence.py` compares every
pair of the 88 files and prints each pair that shares more than half its rows:

| File | Copy of |
|---|---|
| `cycling/dataset15` | `cycling/dataset1`, identical |
| `lying/dataset15` | `lying/dataset1`, identical |
| `lying/dataset10` | `lying/dataset1`, identical except one row (`time` 1500) |
| `lying/dataset11` | `lying/dataset3`, identical |
| `lying/dataset12` | `lying/dataset5`, identical |
| `lying/dataset13` | `lying/dataset7`, identical |
| `lying/dataset14` | `lying/dataset9`, identical |

Among the other pairs, the most rows any two files share is 31 of 480. That happens
when still postures repeat values. Pairs sharing all or all but one row are copies, not
repetitions. A train and test split that puts two copies on opposite sides leaks the test
recording into training. The loader in `src/arwsn/loading.py` handles this.

**All recordings come from one person.** The UCI description says the network was "worn
by an actor". Accuracy on this dataset measures how well a model separates one person's
activities. It says nothing about new people.

**Format irregularities.** The loader handles each of these:

- Every file starts with five `#` comment lines. The fifth holds the column names.
- `bending2/dataset4.csv` separates fields with spaces and ends each line with a space.
  Every other file uses commas.
- The last row of `cycling/dataset9.csv` and `cycling/dataset14.csv` ends with a stray
  comma. A naive `read_csv` turns it into an eighth, empty column.
- Line endings mix CRLF and LF, and some rows end with trailing whitespace.
- `cycling/dataset9.csv` and `cycling/dataset14.csv` have no final newline.
- `sitting/dataset8.csv` has 479 rows, not 480. The row for `time` 13500 is missing.
