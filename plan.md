# Plan

Submission deadline: October 6, 2026 (to be confirmed on the hackathon page).
Deliverable: a runnable repo, plus an optional demo video.

# Product
The user picks a start station, a destination and a departure time.
We return several train-only routes. For each route we show:
- the probability that every connection holds
- the arrival-time distribution at the destination (for example "80% by 14:32, 95% by 15:40")

Routes are ranked by reliable arrival, not by planned arrival.

# Decisions
- Routing: a router built from the dataset's own timetable. No external routing API.
- Scope: trains only. Bus and rail-replacement rows are dropped.
- Horizon: planning case first (schedule and history only). Live state (delays already observed that day) is step two if time allows.
- Models: TabPFN-3.5 through the Prior Labs API. The key is read from `.env`, which is not committed.
- Demo: a Streamlit app. An agent wrapper is a stretch goal.

# Data
Source: `monthly_processed_data/` from https://huggingface.co/datasets/piebro/deutsche-bahn-data (CC BY 4.0). We use August and September 2026.

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
   - Build lagged history features (for example the delay of the same train at the same station over the previous 28 days). Only days before the row's own day may be used.
2. Router
   - Pair consecutive stops of each run into hops (about 670k per day).
   - Connection Scan over the hops sorted by departure, with a minimum transfer time of 5 minutes within a station.
   - Alternatives: rerun for later departures, keep routes that are not worse on both arrival time and number of transfers.
3. Delay model
   - For each leg, predict the arrival-delay distribution at the alighting stop and the departure-delay distribution of the connecting train at the transfer station.
   - Context: a few thousand history rows retrieved per query (same train and station, similar trains at that station, same weekday and hour).
4. Route risk
   - Sample the leg distributions to get the probability of each transfer and the arrival distribution.
   - A missed connection costs a fixed penalty (the next departure on the same line).
5. Evaluation
   - Per leg: absolute error and distribution quality (pinball loss).
   - Per connection: Brier score and a calibration plot, on historical transfer pairs.
   - Per route: predicted probability of an on-time arrival against what actually happened on that day.
   - Baselines: historical frequency per train and station, and a gradient-boosted model.
   - Compare the TabPFN-3.5 variants (Fast, Plus, Thinking).
6. Demo and submission
   - Streamlit app that replays a past day: pick stations and time, see ranked routes with risk, then reveal what actually happened.
   - README and video.

# Splits
| Set | Days | Used for |
|---|---|---|
| Context pool | August 1 to September 16 | History the model sees, training data for the gradient-boosted baseline |
| Validation | September 17 to 23 | Choosing context size, retrieval rule, features, model variant |
| Test | September 24 to 30 | Final numbers, run once |

Rules:
- When predicting day D, the context may include every day before D.
- Split by run, not by row. A run belongs to the day it starts.
- History features are lagged for context rows and query rows alike.
- All three evaluation levels use the same validation and test days.
- Evaluate on a sample (a few thousand legs stratified by train type, a few hundred routes) to limit API cost.

# Schedule
- October 3: data prep, router, baselines.
- October 4: delay model and per-leg evaluation.
- October 5: route risk, connection evaluation, Streamlit app.
- October 6: variant comparison, README, video, submit.

# Risks
- Weak signal in the planning case: the per-train-per-station median gives 2.7 min absolute error against 3.0 for the global median. The result has to stand on calibrated probabilities, not point accuracy.
- Latency and API cost: several predictions per route.
- Legs are treated as independent, which is wrong on bad network days. Stated as a limitation.
- The router only plans on days in the dataset and only transfers within one station.
- One test week may be unrepresentative.

# Open points
- Read the judging criteria on the hackathon page (it could not be fetched automatically).
- Check API pricing with `estimate_cost()` before running the evaluation.
