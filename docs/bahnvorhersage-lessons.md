# Lessons from bahnvorhersage

Review of https://gitlab.com/bahnvorhersage/bahnvorhersage (read on October 4, 2026): the
`ml_models/` folder, `public_config.py` and `predictor_webserver/`. The router that calls their
predictor was not read. Numbers about our data were measured on `data/processed/stops.parquet`.

Lessons 1 and 3 (date errors, train types) are done in `prep.py` (`data-prep-plan.md`, section 4).
The other proposals are in `plan.md` and `data-prep-plan.md`, marked as proposed.

## 1. Their setup

- One XGBoost classifier for arrivals and departures together (flag `is_arrival`). The target is
  shifted into 34 one-minute classes from -3 to +30, so the output is a full distribution.
- Main target: `delay_diff`, the final delay minus DB's current live forecast (`delay_prognosed`).
- Planning mode in the same model: every stop is also added once with `delay_prognosed = 0`,
  `dwell_time_prognosed = dwell_time_schedule` and `minutes_to_prognosed_time = 1440`. In these
  rows the target is the plain final delay, which is our planning target.
- Retrained every night. The live forecast is not in the model, it comes with each request.

### Model inputs (17 columns)

| Input | Notes |
|---|---|
| `number` | Train number as an integer, not a category |
| `lat`, `lon` | Station coordinates. No station ID is used |
| `stop_sequence`, `distance_traveled` | Position in the trip, metres since the first stop |
| `bearing` | Direction in degrees from the first stop to the destination |
| `dwell_time_schedule` | Planned dwell time in minutes |
| `minute_of_day`, `weekday` | Planned time as one column, ISO weekday |
| `is_regional`, `is_arrival` | Derived from a list of long-distance types, event flag |
| `operator`, `category`, `line` | Categories |
| `delay_prognosed`, `dwell_time_prognosed`, `minutes_to_prognosed_time` | Live inputs (0, schedule and 1440 in planning mode) |

### Data preparation

- Drops non-stopping and operational stops, cancelled stops, and a list of historic or
  non-passenger categories (steam and museum trains, `DB`, `P`, `D`, `-`).
- Drops trains that arrive days early and negative dwell times.
- Drops snapshots taken after the event already happened (the live version of our leak rule).
- Subsamples training trips by type, by whole trip: 80% of S-Bahn trips removed (the code comment
  says 60%), 50% of RB, 40% of RE, 20% of other regional. Long-distance trips are all kept. This only
  shrinks training data. S-Bahn is still predicted at request time.
- Clips the target to -3 to +30 and drops rows outside.
- Unknown categories at prediction time become null (`cast(..., strict=False)`).

### Evaluation

- SED: the sum over classes of |predicted CDF - observed step CDF|. With one-minute classes this is
  a discrete CRPS.
- Baselines: global mean, per category, per stop, both as point forecasts and as empirical
  distributions scored with SED.
- Point forecasts from the distribution compared three ways: mean, median, most likely class.
- Marginal calibration: average predicted CDF against average observed CDF, and the average
  predicted histogram against the actual one.
- Breakdowns by forecast horizon, long-distance against regional, ICE only, time of day in
  20-minute bins.
- Drift: one fixed model scored on rolling weekly windows for a year.

### Transfer probability

`SinglePredictor.calculate_transfer_scores` computes P(transfer works) analytically from the two
discrete distributions, assuming the arriving and the departing delay are independent:

P(arr <= buffer) + sum over a > buffer of P(arr = a) * P(dep >= a - buffer),
with buffer = planned gap - minimum transfer time.

The sum covers only a > buffer: the cases a <= buffer are in the first term already. If trains
never depart early, the same value is the sum over all a of P(arr = a) * P(dep >= a - buffer),
without the first term.

Because their distribution stops at +30, a large arrival delay can never be caught, so transfers
with a short buffer come out too pessimistic.

## 2. Takeaways for us

### Data preparation

1. **Null impossible delays.** 15 delays in our table are about ±24h. Two kinds, neither safely
   fixable:
   - planned date one day late: RB 13918 (Aug 30, 5 rows at -1437). The same run also has a planned
     leg of 1444 min Serrig to Saarburg on four days.
   - actual time one day late just after midnight: erx 21043, erx 21088, SBH 34402 (7 rows,
     +1351 to +1417).

   Also 7 planned legs of about 24h (`run_planned_min` 1429 to 1485: RB 13918, RB 7889536,
   RB 7889562, S 38560). The other legs over 300 min are real night trains (NJ, DZ, UEX, D), so a
   plain upper bound on `run_planned_min` is wrong. 274 arrivals and 84 departures are more than
   60 min early. These are not checked yet.
2. **Do not clip the target.** We need the tail for p95, route risk and transfers.
3. **Drop historic and odd train types:** `DB` (10,832 rows), `PRE` (1,876), `MBB` (1,305), `P`,
   `SDG`, `DPN`, `ÖBA`, `KTB`, `UEF` and similar. About 17k rows in total. Keep `UEX`: these are
   holiday night trains with passengers (272 rows, 8 legs over 300 min, see lesson 1).
4. **One `minute_of_day` per event** instead of separate hour and minute columns.
5. **Geography:** station `lat` and `lon`, `distance_traveled` and `bearing`. They generalize to
   stations or trains that are missing from a request's context, and `bearing` gives the direction
   of travel that we lose by leaving out `final_destination`. Needs a station table that maps EVA
   numbers to coordinates.
6. **`is_regional` flag** (and maybe operator) as a coarse fallback for rare train types.
7. **Unknown categories:** decide on purpose. Test null against a category with no context rows.
8. **If we ever subsample, sample whole runs, stratified by type,** never single rows.

### Evaluation

9. **CRPS** on the full predicted distribution, next to pinball loss.
10. **Distribution baselines:** empirical CDF per (train, station), falling back to
    (type, station), then global, scored with the same metrics.
11. **Calibration:** average predicted CDF against observed, plus the coverage of p80 and p95.
12. **Breakdowns:** long-distance against regional, and by hour.
13. **Analytic transfer probability** from the two distributions, scored with the Brier score.
    No sampling noise.

### Mistakes to avoid

14. **Batch training that keeps only the last batch.** Their loop calls `fit` once per weekly batch
    without `xgb_model`, so each call starts a new model. The final model has seen one week, and
    that week ends 7 days before the cutoff. Our gradient-boosted baseline is fit once on the whole
    context pool.
15. **Tuning on the test week.** Their Optuna study scores trials on the week they report. We tune
    only on the validation week and run the test week once.

## 3. Planning with the current delay

"My train is 8 min late now, will I make my connection, or should I take another route?" is still
planning. It only adds what is known at query time t.

How much the last observed delay carries, measured on the validation week (predict `arr_delay` as
the departure delay k stops earlier):

| Observed | Median time ahead | MAE, carry forward | MAE, predict 0 | Correlation | Mean change |
|---|---|---|---|---|---|
| 1 stop back | 3 min | 0.62 | 3.35 | 0.97 | -0.09 |
| 3 stops back | 10 min | 1.27 | 3.54 | 0.89 | +0.24 |
| 5 stops back | 16 min | 1.74 | 3.70 | 0.82 | +0.53 |
| 10 stops back | 30 min | 2.51 | 3.95 | 0.69 | +1.11 |

The best planning baseline (per train and station median) has MAE 2.7. Our dataset has no DB
forecasts, so the observed delay at an earlier stop stands in for `delay_prognosed`. The proposed
data design is in `data-prep-plan.md`, section 10.

## 4. Not relevant for us

- Their live-only inputs as they stand (`dwell_time_prognosed`, DB's forecast itself).
- Subsampling S-Bahn: our context is per request, so a category is never under-represented in its
  own request.
- Their memory dtype casts (we already use Int16).
