# Plan

Submission deadline: October 6, 2026 (to be confirmed on the hackathon page).
Deliverable: a runnable repo, plus an optional demo video.

# Product
The user picks a start station, a destination and one time, "now" (decided October 4). "Now" is
also the earliest departure: all candidate routes leave at or after it. Only what happened before
"now" may be used.
We return up to 10 train-only routes. For each route we show:
- how likely each transfer works, as one of four levels (very likely, likely, uncertain,
  unlikely)
- the arrival at the destination reached with 80% and 95% certainty (for example "by 14:53
  (80%), by 14:56 (95%)")

Routes are ranked by the arrival reached with 80% certainty, then by fewer transfers (decided
October 5).

Headline question: "My train is 8 min late now. Will I make my connection, or should I take
another route?" Trains already running at "now" are predicted from their observed delay. Trains
not running yet (later legs, or trips planned in advance) are predicted from the schedule.

# Decisions
- Routing: a router built from the dataset's own timetable. No external routing API.
- Scope: trains only. Bus and rail-replacement rows are dropped.
- What the model predicts (decided October 4, replaces "live state is step two"): the change in delay, like bahnvorhersage's `delay_diff`. One model for all legs (`docs/data-prep-plan.md`, section 10).
  - Train already running at "now": the input is its last known delay (the delay at the last stop where we know it before "now"), and the model predicts how much that changes by the stop we care about. We have no DB forecasts, so the last known delay stands in for them. This is the headline case.
  - Train not started yet (later legs, or trips planned ahead): its last known delay is empty, and the model predicts the delay itself from the timetable and history. Needed in almost every request.
- Models: TabPFN-3.5. The evaluation used the Prior Labs API (key in `.env`, not committed). The app runs TabPFN on this machine by default (`LOCAL` in `app/app.py`, decided October 5 to save API credits), with TabPFN 3.5 or 3.5 Fast to choose in the form.
- Model setup (decided October 5): `tabpfn_14d_5k_last_known` for all legs, whether the train is running or not. Context of 5,000 rows: 7 days back, 14 days for the same train. Features: timetable, `days_ago` and the last known delay. Development and evaluation focus on arrivals: arrival and departure delays at one stop are within 1 min of each other at 93% of stops.
- Demo: a Streamlit app. An agent wrapper is a stretch goal.

# Data
Source: `monthly_processed_data/` from https://huggingface.co/datasets/piebro/deutsche-bahn-data (CC BY 4.0). We use September 2026 only (decided October 5): TabPFN and the baselines look back at most 14 days, so the earliest validation day needs September 3.

Facts from the September file (14.8M stop events, 5,284 stations):
- `train_line_ride_id` repeats across days. The per-day run key is `id` without its last part (`<ride id>-<run start YYMMDDHHMM>-<stop number>`).
- Times are local German time without a timezone.
- Train numbers are only unique together with the train type.
- 38 station names have more than one EVA number (München Hbf has 4), so stations are merged by name.
- Non-train types to drop: Bus, bus, SEV, Bsv, BSv (about 306k rows).
- Delays: median 1 min, mean 3.4, p90 9, p99 36. 35% are zero. 3.6% of stops are canceled.

# Components
1. Data prep
   - Rebuild runs, drop non-train rows, merge stations by name.
   - Stops table is schedule-only (no lagged history columns). TabPFN gets history through the shared context rows. Lagged history features (for example the delay of the same train at the same station over the previous 28 days) are optional: add them only if they clearly beat the schedule-only version on the validation week (see `docs/data-prep-plan.md`, section 7). Only days before the row's own day may be used.
2. Router
   - Pair consecutive stops of each run into hops (about 670k per day).
   - Connection Scan over the hops sorted by departure, with a minimum transfer time of 5 minutes within a station.
   - Alternatives: rerun for later departures, keep routes that are not worse on both arrival time and number of transfers.
3. Delay model
   - For each leg, predict the arrival-delay distribution at the alighting stop and the departure-delay distribution of the connecting train at the transfer station.
   - Two models, two calls per request: one for arrival delay, one for departure delay. Cancellation is left out of the first version.
   - Legs of started and not-started trains share one context, so a request still makes two calls. Context rows of earlier days get their last known delay by replaying that day at the same clock time as "now" (`docs/data-prep-plan.md`, section 10). Separate contexts for the two kinds are a variant to test on the validation week.
   - Context: one shared context per request, not per leg. Prediction time is set by the context size, not by the number of rows predicted, so one call answers all legs of all candidate routes.
   - The context is built from the stations and trains on the candidate routes (at most 5 routes), from the request day before "now" and the 7 days before it (decided October 5), in three parts:
     1. Same train (done). First the stops the train passed on the request day before "now", latest stop first. These come from the query's own runs and are the strongest signal. Then the whole rides of the same train on the earlier days, most recent day first (planned October 5; before: only at the stations of the query rows).
     2. Other trains at those stations (done). First the rows of the 60 minutes before "now" on the request day, which show a disruption that is going on right now. Then rows of the earlier days in the 60 minutes before each query row's planned time (planned October 5: on all 7 days; before: only on the same weekday). Both are split evenly over the train types of the request, like part 3. Rows a type cannot fill go to the other trains at the station (not S-Bahn, when the request has no S-Bahn train).
     3. A general sample from all stations that fills the rest of the context, split evenly over the train types of the request (done). A uniform sample would be about half S-Bahn (46% of all rows), whatever the request is about. The stops table itself is not subsampled: the router and the evaluation need all runs.
   - Rows of the request day (done, `known_at` in `priorbahn/model/context.py`): an event counts only if its actual time is before "now". A planned time before "now" is not enough: a late train may not have left yet. Later events of the same row are set to empty. These rows need the `days_ago` feature (0 for the request day), otherwise TabPFN cannot tell them from older rows.
   - Size (decided October 5): up to 5k context rows, with 14 days for the same train (`priorbahn/model/request.py`). Parts 1 and 2 take everything they find, part 3 fills the rest. If parts 1 and 2 alone have more rows than that, part 1 goes first. 10k rows with 7 days was tested and was worse and slower. The first tests used 2k rows split 40% / 40% / 20%.
   - Measured on a Mac with local weights (10 query rows): about 5 s per call at 1k context rows, 17 s at 3k, 95 s at 10k. API timing is not measured yet.
4. Route risk (done October 5, `priorbahn/risk.py`)
   - Four levels per transfer: does it still leave 2 min to change trains when the incoming train is as late as its predicted q95 (very likely), q80 (likely) or q50 (uncertain)? Otherwise unlikely. The connecting train is taken as on time (only 0.14% of departures are early). Only the arrival model is needed.
   - A transfer held if the connecting train actually left at least 2 min after the incoming train arrived (a cancellation is a miss). With TabPFN, the four levels held 91% / 80% / 69% / 38% of the time in the validation week and 93% / 84% / 62% / 41% in the test week. TabPFN puts 43 to 45% of transfers into "very likely", the baselines (14 days of history) 38 to 39%.
   - Routes are ranked by the arrival reached with 80% certainty, then by fewer transfers. The levels are shown, but do not change the order.
   - TabPFN's q80 and q95 are a bit too low (74% and 91% of arrivals under them). They are not corrected; the measured held rates above already include this.
   - Earlier version (replaced): a yes/no "safe" rule with 5 min at q95. It was too strict: two thirds of transfers were "at risk", and 73% of those still held.
   - Not used (earlier proposal): compute a single transfer analytically from the two distributions (assuming independence), sum over all a of P(arr = a) * P(dep >= a - buffer), with buffer = planned gap - minimum transfer time. If trains never depart early, this is the same as P(arr <= buffer) + sum over a > buffer only of P(arr = a) * P(dep >= a - buffer). Summing over all a after P(arr <= buffer) counts the cases a <= buffer twice and can give more than 1. No sampling noise. Sampling stays for whole routes. Needs untruncated distributions (bahnvorhersage caps at +30 min and so underrates short buffers).
   - Not built: a penalty for a missed connection (the next departure on the same line). The arrival times are shown "if all transfers work".
5. Evaluation
   - Sampling unit: a request (start, destination, time), not a random leg. Sample a few hundred real requests from the validation and test weeks. Each request gets its own shared context, built the same way as in the app.
   - Per leg: absolute error and distribution quality (pinball loss), on the legs of the sampled requests.
   - Per connection: Brier score and a calibration plot, on historical transfer pairs.
   - Per route: predicted probability of an on-time arrival against what actually happened on that day.
   - Baselines: historical frequency per train and station, and a gradient-boosted model. For trains already seen also "the delay stays the same" (MAE 1.27 at 10 min ahead, 2.51 at 30 min on the validation week). This is the baseline to beat for seen trains, not the 2.7 of the not-seen case.
   - Ask each request at a few times (for example 60, 15 and 0 min before the first departure). Report not-seen legs separately, and seen legs by horizon (up to 15 min, 15 to 60, over 60).
   - Compare the TabPFN-3.5 variants (Fast, Plus, Thinking).
   - Proposed additions (from the bahnvorhersage review, `docs/bahnvorhersage-lessons.md`):
     - Per leg: CRPS on the full predicted distribution, next to pinball loss.
     - The historical-frequency baseline is a distribution: empirical CDF per (train, station), falling back to (train type, station), then global. Scored with the same metrics.
     - Calibration per leg: average predicted CDF against the observed one, and how often the actual delay is at or below the predicted p80 and p95.
     - Breakdowns: long-distance against regional, and by hour.
     - Gradient-boosted baseline: fit once on the whole context pool (bahnvorhersage refits per batch and keeps only the last one).
6. Demo and submission
   - Streamlit app (done October 5, `app/`): replays a past day. Pick stations, date and "now", see the ranked routes as cards with a timeline, colored transfer levels and a stop list like the DB app, then reveal what actually happened with a switch on the page. Checked in a real browser with Playwright (`tests/test_app_ui.py`).
   - README and video.

# Splits
| Set | Days | Used for |
|---|---|---|
| Context pool | September 1 to 16 (August 1 until October 5) | History the model sees, training data for the gradient-boosted baseline |
| Validation | September 17 to 23 | Choosing context size, retrieval rule, features, model variant |
| Test | September 24 to 30 | Final numbers, run once |

Rules:
- When predicting day D, the context may include every day before D.
- Split by run, not by row. A run belongs to the day it starts.
- History features are lagged for context rows and query rows alike.
- All three evaluation levels use the same validation and test days.
- Evaluate on a sample of a few hundred requests (stratified by train type and hour) to limit API cost.

# Status and handoff (end of October 5)

The deadline is October 6, 2026. The product works end to end on `main` (no remote, nothing
pushed). What is left is the demo video and the submission.

## Done

| Part | Where | Notes |
|---|---|---|
| Data prep, September only | `priorbahn/data/prep.py` → `data/processed/stops.parquet` | Data-quality fixes (section 4 of `data-prep-plan.md`). Context pool Sep 1–16, validation Sep 17–23, test Sep 24–30. |
| Router | `priorbahn/router/core.py`, `find_journeys` in `priorbahn/eval/requests.py` | Up to 10 routes in the app, 3 in the evaluation. 5 min minimum transfer when planning. |
| Context known at "now" | `priorbahn/model/context.py` | Parts 1–3 as in component 3. Final size: 5,000 rows, 14 days for the same train, 7 for the rest. |
| Last known delay | `priorbahn/model/last_known.py` | Features plus "predict the change". Context rows replay their day at the same clock time. |
| TabPFN call | `priorbahn/model/predict.py`, `priorbahn/model/request.py` | `version` "v3.5" or "v3.5-fast", `local` or API. `request.py` holds the final setup. |
| Transfer levels and ranking | `priorbahn/risk.py` | Four levels with 2 min to change trains. Ranking: 80% arrival, then fewer transfers. |
| Evaluation | `priorbahn/eval/run_requests.py`, `priorbahn/eval/transfers.py`, `priorbahn/eval/baselines.py` | 198 validation and 191 test searches, arrivals only. Baselines `global`, `train_station`, `carry_forward`, all with 14 days of history. 95% ranges by resampling searches. |
| App | `app/app.py`, `app/render.py` | Cards with timeline, colored transfer dots, DB-style stop list, in-page "Show what actually happened" switch, model choice, prediction time. `LOCAL = True` (API credits are nearly used up). |
| Tests | `tests/` | 26 unit tests. `tests/test_app_ui.py` checks the app in Chrome with Playwright, only when the app runs at `APP_URL`. |
| Docs | `README.md`, this file, `data-prep-plan.md` | README has results, setup, limitations. The experiment log is `data/eval/README.md` (not in git, experiments 1–10). |

Results, arrival pinball (lower is better), all methods with the 14 days before the request day:

| Week | Method | All | Not started | Running |
|---|---|---|---|---|
| Validation (1,194 arrivals) | `train_station` | 2.41 | 2.72 | 1.99 |
| | `carry_forward` | 2.42 | 2.72 | 2.00 |
| | TabPFN | 2.28 | 2.79 | 1.55 |
| Test (1,118 arrivals, run once) | `train_station` | 2.48 | 2.72 | 2.17 |
| | `carry_forward` | 2.27 | 2.72 | 1.71 |
| | TabPFN | 2.13 | 2.65 | 1.49 |

On running trains TabPFN beats "the delay stays the same" in both weeks (test: −0.21, 95%
range −0.44 to −0.01). On trains not started there is no clear difference to `train_station`
(test −0.07, range −0.32 to +0.18; validation +0.08). Transfer levels held 91–93% / 80–84% /
62–69% / 38–41% (very likely / likely / uncertain / unlikely). The early per-leg evaluation
on 300 sampled runs (`priorbahn/eval/run.py`, experiments 1–3) is superseded.

## Left, in order

1. **Demo video.** Not started. Suggested story: Heidelberg Hbf → Lübeck Hbf, September 24,
   08:00. Show the ranked routes, the colored transfers (Hamburg Hbf is "uncertain", the
   ECE 8 is predicted +15, 80%: +35), open "Stops and times", then tick "Show what actually
   happened" (the ECE 8 arrived +31, the transfer held). Then the README results table.
2. **Submit.** Read the judging criteria and the submission form on
   https://platform.priorlabs.ai/hackathon-3.5 first (it could not be fetched automatically).
   Pushing to a public repo needs a remote, which does not exist yet: ask the user.
3. **Optional, if time is left** (none started):
   - The router misses late trains planned before "now" that you could still catch (example:
     ICE 1205 on September 20, planned to leave Berlin Hbf at 13:56, 18 min late).
   - A missed transfer is not turned into a later arrival (penalty: the next departure on the
     same line). The cards say "if all transfers work".
   - The departure model is not evaluated for the final setup (arrivals only; arrival and
     departure delays at one stop are within 1 min at 93% of stops).
   - TabPFN 3.5 against 3.5 Fast: only a 2-search timing test (Fast about 10% slower, too few
     to judge). The app shows the prediction time of each search for a comparison by hand.
   - TabPFN's q80 and q95 are a bit optimistic (74% and 91% of arrivals under them).
   - Gradient-boosted baseline: written, too slow to run.
   - Proposals in `docs/bahnvorhersage-lessons.md` not adopted (CRPS, geography features).

## How to pick it up

- Setup and commands: `README.md` (download September, `python -m priorbahn.data.prep`, app,
  evaluation, tests).
- Not in git: `data/` (raw file, stops table, evaluation caches and scores) and `.env` (the
  Prior Labs key, `PRIORLABS_API_KEY`). TabPFN predictions of the evaluation are cached per
  search in `data/eval/<split>/tabpfn_cache/`, so re-scoring needs no API calls.
- API credits are nearly used up: use the local model (`--local` in the evaluation, `LOCAL`
  in the app). One app search takes about 50 s locally on a Mac, about 15–20 s via the API.
- Streamlit keeps imported modules (`app/render.py`, `priorbahn/...`) after a code change.
  Restart the server after changing them, not just "Rerun".
- `st.html` cleans the HTML with DOMPurify, but `<style>`, `<details>` and `<input>` survive.
  The reveal switch is CSS only (`:has(#rr-reveal:checked)`), so opened stop lists stay open.
- Working conventions of the user: uv only (`uv run --no-sync ...`), ruff via pre-commit,
  feature branch for bigger work, Conventional Commits in plain, simple English, commit model
  or context changes only after their evaluation runs finished, stay at plan level when
  brainstorming, never publish anything online (mockups as local files), explain in short,
  simple sentences.

# Risks
- Weak signal for trains not seen yet: the per-train-per-station median gives 2.7 min absolute error against 3.0 for the global median. The result has to stand on calibrated probabilities, not point accuracy.
- Strong baseline for seen trains: "the delay stays the same" is hard to beat when the stop is close. The room for TabPFN is at 15 min and more ahead, where the delay changes more (mean change +0.5 min at 16 min, +1.1 at 30 min).
- The last observed delay is a weaker input than DB's own forecast, which also knows schedule buffers and disruptions.
- The schedule is tight: the change-in-delay model, route risk and the app all land on October 5 and 6.
- Latency: one search takes about 15 to 20 s through the API, about 50 s with the local model on a Mac.
- Legs are treated as independent, which is wrong on bad network days. Stated as a limitation.
- The router only plans on days in the dataset and only transfers within one station.
- One test week may be unrepresentative.

# Open points
See "Left, in order" above.
