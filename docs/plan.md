# Plan

Submission deadline: October 6, 2026 (to be confirmed on the hackathon page).
Deliverable: a runnable repo, plus an optional demo video.

# Product
The user picks a start station, a destination and one time, "now" (decided October 4). "Now" is
also the earliest departure: all candidate routes leave at or after it. Only what happened before
"now" may be used.
We return several train-only routes. For each route we show:
- the probability that every connection holds
- the arrival-time distribution at the destination (for example "80% by 14:32, 95% by 15:40")

Routes are ranked by reliable arrival, not by planned arrival.

Headline question: "My train is 8 min late now. Will I make my connection, or should I take
another route?" Trains already running at "now" are predicted from their observed delay. Trains
not running yet (later legs, or trips planned in advance) are predicted from the schedule.

# Decisions
- Routing: a router built from the dataset's own timetable. No external routing API.
- Scope: trains only. Bus and rail-replacement rows are dropped.
- What the model predicts (decided October 4, replaces "live state is step two"): the change in delay, like bahnvorhersage's `delay_diff`. One model for all legs (`docs/data-prep-plan.md`, section 10).
  - Train already running at "now": the input is the delay where we last saw it, and the model predicts how much that changes by the stop we care about. We have no DB forecasts, so this sighting stands in for them. This is the headline case.
  - Train not started yet (later legs, or trips planned ahead): marked "not seen yet", and the model predicts the delay itself from the timetable and history. Needed in almost every request.
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
   - Stops table is schedule-only (no lagged history columns). TabPFN gets history through the shared context rows. Lagged history features (for example the delay of the same train at the same station over the previous 28 days) are optional: add them only if they clearly beat the schedule-only version on the validation week (see `docs/data-prep-plan.md`, section 7). Only days before the row's own day may be used.
2. Router
   - Pair consecutive stops of each run into hops (about 670k per day).
   - Connection Scan over the hops sorted by departure, with a minimum transfer time of 5 minutes within a station.
   - Alternatives: rerun for later departures, keep routes that are not worse on both arrival time and number of transfers.
3. Delay model
   - For each leg, predict the arrival-delay distribution at the alighting stop and the departure-delay distribution of the connecting train at the transfer station.
   - Two models, two calls per request: one for arrival delay, one for departure delay. Cancellation is left out of the first version.
   - Seen and not-seen legs share one context, so a request still makes two calls. The context holds rows from earlier days of both kinds: runs with a sighting at a similar horizon, and runs with no sighting. Separate contexts for the two kinds are a variant to test on the validation week.
   - Context: one shared context per request, not per leg. Prediction time is set by the context size, not by the number of rows predicted, so one call answers all legs of all candidate routes.
   - The context is built from the stations and trains on the candidate routes (at most 5 routes), in three parts:
     1. Same train (done). First the stops the train passed on the request day before "now", latest stop first. These come from the query's own runs and are the strongest signal. Then the same train at the same station on earlier days, most recent day first.
     2. Other trains at those stations (done). First the rows of the 60 minutes before "now" on the request day, which show a disruption that is going on right now. Then rows of earlier days, on the same weekday, in the 60 minutes before each query row's planned time. Both are split evenly over the train types of the request, like part 3. Rows a type cannot fill go to the other trains at the station (not S-Bahn, when the request has no S-Bahn train).
     3. A small general sample from all stations, split evenly over the train types of the request (done). A uniform sample would be about half S-Bahn (46% of all rows), whatever the request is about. The stops table itself is not subsampled: the router and the evaluation need all runs.
   - Rows of the request day (done, `known_at` in `dbdelay/model/context.py`): an event counts only if its actual time is before "now". A planned time before "now" is not enough: a late train may not have left yet. Later events of the same row are set to empty. These rows need the `days_ago` feature (0 for the request day), otherwise TabPFN cannot tell them from older rows.
   - Starting point: about 2k context rows, split roughly 40% / 40% / 20% across the three groups. Both the size and the split are settings to tune on the validation week.
   - Measured on a Mac with local weights (10 query rows): about 5 s per call at 1k context rows, 17 s at 3k, 95 s at 10k. API timing is not measured yet.
4. Route risk
   - Sample the leg distributions to get the probability of each transfer and the arrival distribution.
   - Proposed: compute a single transfer analytically from the two distributions (assuming independence), sum over all a of P(arr = a) * P(dep >= a - buffer), with buffer = planned gap - minimum transfer time. If trains never depart early, this is the same as P(arr <= buffer) + sum over a > buffer only of P(arr = a) * P(dep >= a - buffer). Summing over all a after P(arr <= buffer) counts the cases a <= buffer twice and can give more than 1. No sampling noise. Sampling stays for whole routes. Needs untruncated distributions (bahnvorhersage caps at +30 min and so underrates short buffers).
   - A missed connection costs a fixed penalty (the next departure on the same line).
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
   - Streamlit app that replays a past day: pick stations, departure time and "now", see the observed delays of running trains and the ranked routes with risk, then reveal what actually happened.
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
- Evaluate on a sample of a few hundred requests (stratified by train type and hour) to limit API cost.

# Status (October 4)
Done:
- data prep (`dbdelay/data/prep.py`) with the data-quality fixes
- router and Streamlit UI (`dbdelay/router`, `app/`)
- shared-context builder, feature lists with optional `days_ago`, TabPFN predict step (`dbdelay/model`), smoke test
- per-leg evaluation of not-seen legs (`dbdelay/eval/`, results in `data/eval/README.md`): 300 sampled validation runs, baselines `global` and `train_station`, TabPFN variants. Result: `train_station` beats TabPFN overall (arrival pinball 1.19 against 1.28). TabPFN wins only on S-Bahn. These scores use the stops table from before the data-quality fixes.

Not done: the gradient-boosted baseline (written, too slow to run), "the delay stays the same" baseline, parts 1 and 2 of the context with rows of the request day, change-in-delay model (sightings), route risk, delay model in the app.

# Schedule
- October 3: data prep, router. Done.
- October 4: data-quality fixes and historic train types. Baselines `global` and `train_station`, per-leg evaluation of not-seen legs. Done.
- October 5: context parts 1 and 2 with rows of the request day and the type split, evaluation with "now" set to each request's departure time, "delay stays the same" baseline. Then route risk with the analytic transfer probability, connection evaluation.
- October 6: Streamlit app with "now", variant comparison, README, video, submit.

# Risks
- Weak signal for trains not seen yet: the per-train-per-station median gives 2.7 min absolute error against 3.0 for the global median. The result has to stand on calibrated probabilities, not point accuracy.
- Strong baseline for seen trains: "the delay stays the same" is hard to beat when the stop is close. The room for TabPFN is at 15 min and more ahead, where the delay changes more (mean change +0.5 min at 16 min, +1.1 at 30 min).
- The last observed delay is a weaker input than DB's own forecast, which also knows schedule buffers and disruptions.
- The schedule is tight: sightings, route risk and the app all land on October 5 and 6.
- Latency and API cost: several predictions per route.
- Legs are treated as independent, which is wrong on bad network days. Stated as a limitation.
- The router only plans on days in the dataset and only transfers within one station.
- One test week may be unrepresentative.

# Open points
- Read the judging criteria on the hackathon page (it could not be fetched automatically).
- Check API pricing with `estimate_cost()` before running the evaluation.
- Decide which of the remaining proposals from `docs/bahnvorhersage-lessons.md` to adopt (evaluation additions above, geography features in `docs/data-prep-plan.md`, section 6).
- The router only uses trains planned at or after "now". A late train planned before "now" may still be catchable (example: ICE 1205 on September 20, planned to leave Berlin Hbf at 13:56, 18 min late). Decide whether the router should offer such trains.
