# Prior Bahn: reliable train routes with TabPFN-3.5

Plan a train trip in Germany and see how likely each transfer works. The app predicts the
delay of every train on the candidate routes with TabPFN-3.5, from what was known at the
moment you ask, and ranks the routes by when they get you there.

Built for the Prior Labs TabPFN-3.5 Hackathon, on Deutsche Bahn data from September 2026.

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
   every stop where you leave a train, as three numbers: expected (q50), 80% and 95%.
5. **Transfers.** Each transfer gets one of four levels (`priorbahn/risk.py`): does it still
   leave 2 minutes to change trains if the incoming train is as late as its predicted 95%,
   80% or 50% level? The connecting train is taken as on time (only 0.14% of departures
   leave early).

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

How often transfers actually held (2 minutes to change trains), by the level TabPFN gave
them:

| Level | Share of transfers | Held, validation | Held, test |
|---|---|---|---|
| Very likely | 43% | 91% | 93% |
| Likely | 13% | 80% | 84% |
| Uncertain | 20% | 69% | 62% |
| Unlikely | 24% | 38% | 41% |

TabPFN puts more transfers into "very likely" than the baselines (43 to 45% against 38 to
39%), and they hold a little more often (91 to 93% against 89 to 91%). Details and all experiments: `docs/plan.md` and
`data/eval/README.md` (made by the evaluation).

## Run it

Needs [uv](https://docs.astral.sh/uv/) and about 1 GB of disk for the data.

```bash
uv sync

# the September 2026 file of the dataset (CC BY 4.0)
uv run --no-sync hf download piebro/deutsche-bahn-data --repo-type dataset \
  --include "monthly_processed_data/data-2026-09.parquet" --local-dir data

# build the stops table (data/processed/stops.parquet, about 2 minutes)
uv run --no-sync python -m priorbahn.data.prep

# the app
uv run --no-sync streamlit run app/app.py
```

The app runs TabPFN on your machine (`LOCAL = True` in `app/app.py`). The first
search downloads the model weights. One search takes about 50 seconds on a Mac. To use the
Prior Labs API instead, set `LOCAL = False` and put your key in `.env` as
`PRIORLABS_API_KEY=...`.

In the app, pick a day between September 15 and 30, two stations and "now". Each
route shows its planned times with the expected (median) arrival delay, a timeline with a
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
uv run --no-sync python -m priorbahn.eval.transfers validation         # transfer levels
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
| `docs/` | plans, data notes, lessons from the bahnvorhersage project |

## Limitations

- Trains are treated as independent. On a bad network day, delays of the incoming and the
  connecting train are linked, so the levels are less reliable then.
- TabPFN's 80% and 95% levels are a bit optimistic: 74% and 91% of arrivals stayed under
  them. The held rates above already include this.
- A missed transfer is not turned into a later arrival. Routes with weak transfers show
  their arrival "if all transfers work".
- The router only offers trains planned at or after "now", so it misses a late train you
  could still catch. It only transfers within one station.
- The data covers one month (September 2026), and the test week is one week.

## Data

[piebro/deutsche-bahn-data](https://huggingface.co/datasets/piebro/deutsche-bahn-data),
CC BY 4.0.
