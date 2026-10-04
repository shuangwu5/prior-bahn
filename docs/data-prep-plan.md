# Data preparation plan

The `stops` table is built by `dbdelay/data/prep.py` (output `data/processed/stops.parquet`). `hops` and `transfers` (section 8) are not built yet. Raw files are in `data/monthly_processed_data/`.

Provenance of the facts below:
- **Dataset card** (HF `piebro/deutsche-bahn-data`): column names and types, and the descriptions
  "Delay in minutes" and "Actual arrival or departure time". The card does not say which event a
  row's delay refers to.
- **Measured** on the September file (14.77M rows): null rates and the delay behaviour in section 3.
  The August file (14.52M rows) has the identical schema, but the section 3 checks were only run
  on September.
- **Guessed from the column name**, not documented: `is_additional_stop`, `is_replacement_train`,
  `replaced_train_number`. These three are not in the card.

## 1. What one row is

One row is one **stop event**: one train, at one station, on one run. A run is one trip of one
train on one day. All 21 columns are in the table below.

## 2. Raw columns

| Column | Type | Meaning | Null rate |
|---|---|---|---|
| `station_name` | str | Station name | 0.06% |
| `xml_station_name` | str | Station name as written in the source XML (spacing differs, e.g. `Fürth(Bay)Hbf`) | 0 |
| `eva` | str | Station ID (EVA number). One name can have several EVAs (München Hbf has 4) | 0 |
| `train_number` | str | Train number. Only unique together with `train_type` | ~0 |
| `line_number` | str | Line label such as `S12` or `RB25`. Mostly empty for long-distance trains | 2% |
| `final_destination_station` | str | Last station of the run, as shown to passengers | 7.7% |
| `delay_in_min` | int | Delay of this row's event (see section 3) | 0 |
| `time` | datetime | Actual time of this row's event (planned plus delay) | 0 |
| `arrival_is_canceled` | bool | The arrival at this stop was canceled | 0 |
| `departure_is_canceled` | bool | The departure from this stop was canceled | 0 |
| `train_type` | str | ICE, IC, RE, RB, S, Bus, and so on | 0 |
| `is_additional_stop` | bool | Unscheduled extra stop | 0 |
| `is_replacement_train` | bool | Train that replaces another one | 0 |
| `replaced_train_number` | str | Number of the replaced train | 99.5% |
| `train_line_ride_id` | str | ID of the ride. Repeats across days | 0 |
| `train_line_station_num` | int | Stop number within the run (1 is the first stop) | 0 |
| `arrival_planned_time` | datetime | Scheduled arrival | 7.7% (first stop) |
| `arrival_change_time` | datetime | Actual arrival | 7.7% |
| `departure_planned_time` | datetime | Scheduled departure | 7.7% (last stop) |
| `departure_change_time` | datetime | Actual departure | 7.7% |
| `id` | str | `<train_line_ride_id>-<run start YYMMDDHHMM>-<stop number>`. Unique per row | 0 |

Times are local German time without a timezone.

## 3. What the target really is (important finding)

`delay_in_min` is not "arrival delay". It is the delay of whichever event the row is about:

| Stop position | `delay_in_min` equals | Share |
|---|---|---|
| First stop (no arrival) | departure change minus departure planned | 100% |
| Middle stop | **departure** delay | 100% (12.49M of 12.49M rows) |
| Last stop (no departure) | arrival delay | 100% |

At middle stops the arrival delay differs from the departure delay in about 47% of rows (a train
can gain or lose time while standing). The plan needs both, so we **recompute** them from the
four time columns instead of using `delay_in_min`:

- `arr_delay = arrival_change_time - arrival_planned_time` (minutes)
- `dep_delay = departure_change_time - departure_planned_time` (minutes)

`time` is the actual departure time when there is one, else the actual arrival time. It is
therefore "planned plus delay". Never use it as a feature (it contains the answer). Use the
planned times instead.

Canceled rows are not usable as a delay target. The `delay_in_min` of a canceled row is noise
(mean 5.7, max 1389, and 68% are exactly 0).

## 4. Rows to drop

| Rule | Why | Size |
|---|---|---|
| `train_type` matches `bus`, `Bus`, `SEV`, `Bsv`, `BSv` (case-insensitive) | Train-only scope | about 306k per month |
| `is_replacement_train` is true | Replacement services have no stable history to learn from. Dropped (decided). | 0.56% |
| `is_additional_stop` is true | Not in the planned timetable, so the planning case never knows about them | 0.17% |
| Both planned times null | Nothing to plan or predict | 5 rows |
| Rows without `station_name` | Keep, and label with the EVA code (done in `prep.py`) | 0.06% |

Canceled rows are **not dropped**. They stay in the table with a cancel label, because
"will this stop be canceled" is part of the risk. They are excluded only from the delay-regression
target.

### Planned for October 4 (from the bahnvorhersage review, not in `prep.py` yet)

Details and counts in `bahnvorhersage-lessons.md`, section 2.

| Rule | Why | Size |
|---|---|---|
| Drop historic and non-passenger types: `DB`, `PRE`, `MBB`, `P`, `SDG`, `UEX`, `DPN`, `ÖBA`, `KTB`, `UEF` and similar | Museum, steam and special trains, not useful for planning | about 17k rows |
| Set `arr_delay` / `dep_delay` to null when the absolute value is over 1000 min | Date errors of one day, in the planned time (RB 13918) or the actual time just after midnight (erx 21043, erx 21088, SBH 34402). Not safely fixable | 15 delays |
| Set `run_planned_min` to null when it is over 1300 min | Planned date one day late. Legs over 300 min are otherwise real night trains (NJ, DZ, UEX), so no lower bound | 7 legs |
| Check arrivals more than 60 min early | Not yet known whether they are errors | 274 arrivals, 84 departures |

The target is **not clipped**. The tail is needed for p95, route risk and transfers.

## 5. Columns to drop

| Column | Why |
|---|---|
| `time` | Leaks the target. Not a planning-time feature. |
| `delay_in_min` | Ambiguous (section 3). Replaced by `arr_delay` and `dep_delay`. |
| `xml_station_name` | Duplicate of `station_name` |
| `replaced_train_number` | 99.5% null, and replacement rows are dropped |
| `is_replacement_train`, `is_additional_stop` | Constant after the filters above |
| `train_line_ride_id` | Repeats across days. Replaced by `run_id` (below) |
| `line_number` | Optional. Only 2% null but 392 distinct values, and it is redundant with train type plus number for long-distance trains. Keep as a low-priority categorical. |

## 6. Columns to add or transform

| New column | How | Why |
|---|---|---|
| `run_id` | `id` without its last part (`<ride id>-<run start>`) | Real per-day run key |
| `run_day` | Date from the run-start part of `id` | The split unit. A run belongs to the day it starts (1.4% of rows have a `time` on the next day). |
| `train_key` | `train_type` + `train_number` | Train number alone is not unique |
| `station` | `station_name`, else the EVA code. Stations are merged by name. | 38 names have several EVAs |
| `final_destination` | `final_destination_station`, with the empty values filled by `station` | The source leaves it empty exactly at the last stop of a run, where the destination is that station. Note: on other stops the destination uses the source's spelling (`Frankfurt(Main)Süd`), which differs from `station` (`Frankfurt (Main) Süd`) in about 26% of runs. |
| `arr_delay`, `dep_delay` | Section 3 | The two targets |
| `arr_canceled`, `dep_canceled` | Existing flags, renamed | Cancel label |
| `planned_arr`, `planned_dep` | Existing planned-time columns | Kept as the schedule |
| `dwell_planned_min` | `planned_dep - planned_arr` | Planned time standing at the stop |
| `run_planned_min` | Planned time from the previous stop to this one | Planned leg length |
| `arr_hour`, `arr_minute`, `dep_hour`, `dep_minute`, `weekday` | Hour and minute of `planned_arr` and of `planned_dep`, and the weekday | Calendar features. Each model uses the hour and minute of its own event. They are missing where the event does not exist (first and last stop). The old `hour` and `is_weekend` are removed. |
| `stop_num`, `n_stops`, `stop_frac` | `train_line_station_num`, the highest stop number of the run, and their ratio | Position in the run. Later stops tend to collect more delay. |
| `split` | `context`, `validation` or `test` from `run_day` (splits in `plan.md`) | Keeps the split rule in one place |
| `prev_station` | Station of the previous stop, null if the previous stop is missing (about 4% of runs have gaps) | Kept in the table, not used as a feature yet |
| `prev_stop_dep_delay` | **Not in `stops`.** The model uses the delay where the train was last seen before the question time instead, built per request (section 10) | The previous stop may not have happened yet when the question is asked |
| History features | Lagged statistics, see section 7 | Optional. Added only if the validation test shows a gain. |

### Feature lists (decided)

There are two models, so there are two feature lists. Each request builds one shared context (see
`plan.md`, component 3), so a context holds only the stations and trains of the candidate routes,
about 10 to 40 stations and not 5k. Because of that, stations and trains can go in as plain
categories, with no target encoding.

| Column | Arrival model | Departure model | Notes |
|---|---|---|---|
| `train_type` | yes | yes | All 110 values kept, no grouping |
| `station` | yes | yes | Category |
| `train_key` | yes | yes | Category. Marks rows of the same train |
| `line_number` | yes | yes | Category, low priority. May identify S-Bahn trains better than the train number (to check) |
| `weekday` | yes | yes | |
| `arr_hour`, `arr_minute` | yes | no | Hour and minute of the planned arrival |
| `dep_hour`, `dep_minute` | no | yes | Hour and minute of the planned departure |
| `run_planned_min` | yes | yes | Planned length of the leg into this stop |
| `dwell_planned_min` | no | yes | Planned standing time. Matters for departures only. 15% null (first and last stops) |
| `stop_num`, `n_stops`, `stop_frac` | yes | yes | Position in the run |

Left out for now: `prev_station`, `final_destination`, `is_weekend`, the raw planned times, and the
keys and labels (`run_id`, `run_day`, `split`, canceled flags). Both models also train only on rows
that are not canceled and have a non-null target.

### Proposed feature changes (from the bahnvorhersage review)

Each is tested on the validation week before it replaces a decided feature.

| Change | Why |
|---|---|
| `arr_minute_of_day` and `dep_minute_of_day` instead of the separate hour and minute columns | One continuous time column. The minute alone is close to noise |
| Station `lat`, `lon` | Generalizes to stations that are missing from a request's context. Needs a table from EVA number to coordinates |
| `distance_traveled` (km since the first stop) | Better position feature than `stop_frac`. Needs the coordinates |
| `bearing` (degrees from the first stop to the final destination) | Direction of travel as a number. The two directions of a line can behave differently |
| `is_regional` (from a list of long-distance types) | Coarse fallback for rare train types |
| A query category missing from the context: null, or a category with no context rows | Decide by test (open question in the TabPFN section below) |

### How the inputs are passed to TabPFN (checked in tabpfn 9.1.0)

- `fit(X, y)` takes a pandas DataFrame. A datetime column is refused with an error by default.
  The main runs use the explicit hour and minute columns and pass no datetime column.
- Optional experiment: pass `planned_arr` (arrival model) or `planned_dep` (departure model) with
  `inference_config={"TRANSFORM_DATES": True}`, and drop the explicit hour and minute columns of
  that model. TabPFN then builds year, day of year, seconds since the epoch, minute, second, and
  circular (sin and cos) month, day, hour and weekday. Checked on a toy frame with tabpfn 9.1.0.
  Year and second would be constant here, and the date parts only extrapolate, so I expect little
  gain. Not checked: whether the Prior Labs API client supports this setting.
- TabPFN guesses which columns are categories. An integer column with more than 30 distinct values
  is read as a number. So `train_type`, `station`, `train_key` and `line_number` must be given as
  pandas `category` dtype (or listed in `categorical_features_indices`). A column declared this way
  is a category at any size, so no integer codes are needed.
- Build each category list from the context rows and the query rows together, so both use the
  same categories. Behaviour for a query value that is not in the context is not checked yet. Test
  it early.
- Output: `predict(X, output_type="quantiles", quantiles=[0.5, 0.8, 0.95])` returns one array per
  quantile (the values must be Python floats). `output_type="full"` also returns the whole
  predicted distribution, which the route-risk step needs for sampling.
- The row and feature limits for v3.5 are only known after loading the model. Our context of about
  2k rows is far below any limit seen in the package.

## 7. History features (optional, to be tested)

Nothing yet shows that hand-built history columns are needed. TabPFN learns in context, so it may
get the signal from the retrieved context rows alone. The 28-day window in `plan.md` was an
example, not a tested choice.

Order of work:
1. Build `stops` with schedule-only columns (section 6, without history). Done. The first TabPFN
   experiments can start.
2. Compare on the validation week:
   - A: schedule-only columns, with context rows chosen by simple retrieval (same train and
     station, same station and hour).
   - B: A plus lagged history columns (for example mean, p90 and cancel rate of the same train at
     the same station, plus the observation count).
3. Add history only if B clearly beats A. Treat the window length (7, 14, 28 days) and the choice
   of statistics as parameters to tune, not as decisions.

If history is added, the one rule stays: the window ends the day before the row's own day, for
context rows and query rows alike. Otherwise the evaluation leaks.

## 8. Output tables

1. **`stops`**: one row per stop event after the filters, with the columns of section 6 (no history
   features yet). A single parquet file, sorted by `run_day`, `run_id`, `stop_num`.
2. **`hops`**: pairs of consecutive stops of one run (feeds the router and leg evaluation).
3. **`transfers`**: pairs of (arriving run, departing run) at the same station with a planned gap
   of at least 5 minutes and at most about 60 minutes (feeds connection evaluation).

Splits follow `plan.md`: context pool Aug 1 to Sep 16, validation Sep 17 to 23, test Sep 24 to 30,
split by `run_day`.

## 9. Open questions

1. ~~Keep replacement trains in the timetable~~ Decided: drop (done in `prep.py`).
2. ~~Merge stations by name or by EVA~~ Decided: name (done in `prep.py`).
3. ~~How to feed 5k stations to TabPFN~~ Decided: raw category, together with `train_key`. The
   shared per-request context holds only a few dozen stations. See "Feature lists" in section 6.
4. Runs that started on July 31 appear in the August file. ~~Keep them or cut?~~ Decided: cut by `run_day` to Aug 1 to Sep 30 (done in `prep.py`).
5. ~~Is a September-only check enough for the August schema?~~ Decided: yes. Schemas are identical
   and `prep.py` applies the same filters to both files. The section 3 checks stay September-only.
6. Which source for station coordinates (EVA number to `lat`, `lon`)? Needed for the geography
   features in section 6.
7. Are the arrivals more than 60 min early real or data errors?

## 10. Predicting the change in delay (decided October 4, not built yet)

The person asks at time t ("now"). Some trains on the candidate routes are already running, others
have not started yet. One model handles both. Our data has no DB forecasts, so the delay where we
last saw the train stands in for them (measured in `bahnvorhersage-lessons.md`, section 3).

### What the model predicts

The model predicts how much the delay changes between the last place we saw the train and the stop
we care about. Example: the train left stop A 5 min late, and we want stop C. If the model predicts
+2, the train arrives at C about 7 min late.

If we have not seen the train yet (it has not started), there is nothing to change from. The model
then predicts the delay itself, from the timetable and from how this train usually runs.

### Columns added per row

| Column | Meaning |
|---|---|
| `seen_delay` | Delay where we last saw the train before t. Empty if not seen yet |
| `seen_stop_num` | Stop where we last saw it. Empty if not seen yet |
| `horizon_min` | Scheduled travel time from the last sighting to the target stop. Empty if not seen yet |
| `stops_ahead` | Number of stops from the last sighting to the target stop. Empty if not seen yet |
| `delay_change` | The target: delay at the target stop minus `seen_delay`, or the delay itself if not seen yet |

"Seen" means an arrival or departure with an actual time before t. A departure can also be predicted
from the arrival at the same stop (the train has arrived but not left yet). The timetable columns of
the target stop come from `stops` unchanged.

### Rules

- `seen_delay` stays an input next to the target: big delays tend to shrink, small ones tend to
  grow. The delay itself is `seen_delay` plus the predicted change, so route risk and transfers
  work as before. The target is not clipped.
- "Not seen yet" is written as empty, never as 0. A delay of exactly 0 is the most common sighting
  (35%) and means "on time a few minutes ago", which is very different from "we know nothing".
- When the question is asked (2 hours or 1 day before departure) does not matter for a train we have
  not seen. We know the same thing in both cases, so there is no column for it.
- The rows are built per request in the context builder, not stored as one big table (all
  combinations would be hundreds of millions of rows):
  - rows to predict: the last sighting of each train before t, or empty.
  - context rows from earlier days: for each run we pick, one sighting with a horizon similar to
    the rows we predict, plus some rows with no sighting for the trains that have not started.
- Actual times are scheduled time plus delay, so `stops` needs no new columns. Canceled stops have
  no actual time and never count as a sighting.
- Leak rule: a sighting must be an earlier stop of the same run, with an actual time before t.
  Context rows come only from the context-pool days, as before.
- The section 4 date-error rules come first. A wrong `seen_delay` breaks both an input and the
  target.
- Missing stop numbers (about 4% of runs) do not matter here, since the horizon comes from times.
- To test on the validation week: one shared context for both kinds of rows (the default) against
  separate contexts for seen and not-seen rows.
