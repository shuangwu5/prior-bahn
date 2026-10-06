# Prior Bahn: reliable train routes with TabPFN-3.5

Plan a train trip in Germany and see how likely each transfer works. The app predicts the
delay of every train on the candidate routes with TabPFN-3.5, from what was known at the
moment you ask, and ranks the routes by when they get you there.

Built for the Prior Labs TabPFN-3.5 Hackathon, on Deutsche Bahn data from September 2026.

> **Disclaimer.** This repository was written largely with AI assistance. It was made only for
> a hackathon, to show a possible use case of TabPFN. The code has not been reviewed in depth.
> Do not rely on it to plan real trips.

https://github.com/user-attachments/assets/74879025-5650-472d-a24a-15401ea88af8

## The question

"My train is 8 minutes late now. Will I make my connection, or should I take another route?"

DB's own apps know the current delay, but they do not say how likely a 6-minute transfer is.
This project answers that for every transfer of every route, and checks the answer against
what actually happened.

## How it works

1. **Routes.** A router built from the dataset's own timetable finds up to 10 train-only
   routes from your station, leaving at or after "now" (`priorbahn/router`).
2. **Context.** For each search, the app collects up to 5,000 past rows for TabPFN
   (`priorbahn/model/context.py`). It uses only what was known at "now":
   - the same trains: the stops they passed today before "now", and their whole rides on
     the last 14 days
   - other trains at the same stations, around the same time, on the last 7 days, plus the
     last 60 minutes before "now" (this shows a disruption that is going on right now)
   - a random sample of the same train types from all stations
3. **Last known delay.** If a train is already running, its delay at the last stop before
   "now" is a feature, and TabPFN predicts how much the delay changes from there
   (`priorbahn/model/last_known.py`). Past days are replayed at the same clock time, so the
   context rows look like the question. A train that has not started predicts the delay
   itself.
4. **One call per search.** TabPFN fits on the context and predicts the arrival delay at
   every stop where you leave a train, as three numbers: median (q50), 80% and 95%.
5. **Transfers.** Each transfer gets a chance that it works (`priorbahn/risk.py`): how
   likely the incoming train arrives at least 2 minutes before the connecting train leaves.
   The connecting train's typical delay comes from its last known delay, or from the same
   train on past days (`carry_forward`, no extra TabPFN call). Canceled trains are not part
   of the chance. Four levels cut the chance at 97, 80 and 50%, so that green means a
   transfer that almost never fails.

## Results

Evaluated on 198 real searches in the validation week (September 17 to 23) and run once,
without tuning, on 191 searches in the test week (September 24 to 30). "Now" is the
departure time of each search. Scores are the pinball loss of the arrival delay (lower is
better):

| Test week (1,118 arrivals) | All | Train not started | Train already running |
|---|---|---|---|
| Same delays as all trains (`global`) | 3.39 | 3.63 | 3.10 |
| This train at this station, past days (`train_station`) | 2.48 | 2.72 | 2.17 |
| The delay stays the same (`carry_forward`) | 2.27 | 2.72 | 1.71 |
| TabPFN-3.5 Fast | 2.23 | 2.77 | 1.56 |
| XGBoost, one model per day (`xgboost`) | 2.17 | 2.70 | 1.52 |
| **TabPFN-3.5** | **2.13** | **2.65** | **1.49** |

All methods use the same history: the 14 days before the search. For a train that has not
started, `carry_forward` has no delay to carry and uses `train_station`. XGBoost is trained
once per day on that history (about 2.4 million rows, at most 200,000 per train type, about
3 minutes) with the same features as TabPFN, including the train's last known delay.
TabPFN gets 5,000 rows per search and no training. Its rows also include trains of the
search day known before "now", which XGBoost does not see.

- **Trains already running:** TabPFN beats "the delay stays the same" by 0.21 (95% range
  0.01 to 0.44), the case that matters most for "will I make my connection?". It also did
  in the validation week (1.55 against 2.00).
- **Against XGBoost:** about level. In the validation week TabPFN was better on running
  trains (1.55 against 1.74, 95% range of the difference −0.39 to −0.02), in the test week
  the difference is small and not clear (−0.03, range −0.17 to +0.13). XGBoost's quantiles
  are the best calibrated: 51%, 79% and 96% of test arrivals stayed under its 50%, 80% and
  95% levels.
- **Trains not started yet:** no clear difference to the train's own history (test −0.07,
  95% range −0.32 to +0.18; validation +0.08).

How often transfers actually held (2 minutes to change trains, both trains running), by
the level TabPFN gave them:

| Level | Chance | Share (val / test) | Held, validation | Held, test |
|---|---|---|---|---|
| Almost sure | 97% or more | 46% / 45% | 99% | 99% |
| Likely | 80 to 97% | 18% / 19% | 88% | 91% |
| Uncertain | 50 to 80% | 18% / 20% | 66% | 58% |
| Unlikely | below 50% | 18% / 16% | 39% | 39% |

Brier score of the chance (the mean squared difference between the chance and the outcome;
lower is better), validation / test: TabPFN **0.110 / 0.099**, XGBoost 0.111 / 0.105,
`carry_forward` 0.114 / 0.112, `train_station` 0.122 / 0.125. In the test week TabPFN is
clearly better than `carry_forward` and `train_station`. Against XGBoost there is no clear
difference in either week. Details and all experiments: `data/eval/README.md` (made by the
evaluation).

## Run it

Needs [uv](https://docs.astral.sh/uv/) and about 1 GB of disk for the data.

```bash
uv sync --frozen

# the September 2026 file of the dataset (CC BY 4.0)
uv run --no-sync hf download piebro/deutsche-bahn-data --repo-type dataset \
  --include "monthly_processed_data/data-2026-09.parquet" --local-dir data

# build the stops table (data/processed/stops.parquet, about 2 minutes)
uv run --no-sync python -m priorbahn.data.prep

# the app
uv run --no-sync streamlit run app/app.py
```

The app uses the Prior Labs API (costs credits). Put your key in `.env` as
`TABPFN_TOKEN=...`. To run TabPFN on your machine instead, start the app with
`uv run --no-sync streamlit run app/app.py -- --backend local`. The first local search
downloads the model weights.

In the app, pick a day between September 15 and 30, two stations and "now". Each
route shows its planned times with the typical (median) arrival delay, a timeline with a
colored dot per transfer, and the stops with their planned times. In the stop list, a curve
shows the predicted arrival delay at each transfer and at the destination, with the median,
80% and 95% marked. Before a transfer it is red where the train is too late to change.
"Show what actually happened" reveals the real delays, marks them on the curves, and shows
whether each transfer held.

### Evaluation

The evaluation uses the API (add `--local` to run it on your machine).

```bash
uv run --no-sync python -m priorbahn.eval.requests validation          # sample and route searches
uv run --no-sync python -m priorbahn.eval.run_requests validation carry_forward --models arr
uv run --no-sync python -m priorbahn.eval.run_requests validation tabpfn_14d_5k_last_known \
  --models arr --workers 4
uv run --no-sync python -m priorbahn.eval.run_requests validation xgboost --models arr  # ~25 min
uv run --no-sync python -m priorbahn.eval.run_requests validation report
uv run --no-sync python -m priorbahn.eval.transfers validation         # transfer chances
```

Use `test` instead of `validation` for the test week. The searches in the results were drawn
before the data was cut to September, so drawing them again gives a slightly different
sample.

### Tests

```bash
uv run --no-sync pytest
```

The browser test (`tests/test_app_ui.py`, Playwright with your installed Chrome) runs only
when the app is running at `APP_URL` (default `http://localhost:8502`).

## Repository

| Path | What |
|---|---|
| `priorbahn/data/prep.py` | builds the stops table from the raw files |
| `priorbahn/router/` | route search on the timetable of one day |
| `priorbahn/model/` | context, last known delay, features, TabPFN call |
| `priorbahn/risk.py` | transfer levels and route ranking |
| `priorbahn/eval/` | baselines, metrics, evaluation on sampled searches |
| `app/` | the Streamlit app |

## Limitations

- Trains are treated as independent, and the connecting train's delay is one number, not a
  range. On a bad network day, the delays of both trains are linked, so the chances are less
  reliable then.
- TabPFN's 80% and 95% levels are a bit optimistic: 74% and 91% of arrivals stayed under
  them. The transfer levels still held about as often as they claim.
- Canceled trains are not part of the chance. In the evaluation, 6 to 14% of the transfers
  had a canceled train.
- A missed transfer is not turned into a later arrival. Routes with weak transfers show
  their arrival "if all transfers work".
- The router is this project's own, built from the dataset's timetable. It does not use DB's
  public API, so its routes differ from DB Navigator and can include routes that are not the
  best ones.
- The data covers one month (September 2026), and the test week is one week.

## Related work

[Bahn-Vorhersage](https://bahnvorhersage.de) ([source code](https://gitlab.com/bahnvorhersage/bahnvorhersage))
has worked on predicting train delays and transfers for years. It uses XGBoost with a different dataset with more
features than this project, for example the coordinates of each station. If you are interested
in this topic, have a look at it too.

## Data

[piebro/deutsche-bahn-data](https://huggingface.co/datasets/piebro/deutsche-bahn-data),
CC BY 4.0.
