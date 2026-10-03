# Data preparation plan

Plan only, no code yet. Files are in `data/monthly_processed_data/`.

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
| `is_replacement_train` is true | Replacement services have no stable history to learn from. Keep only if the router needs them. Decision needed. | 0.56% |
| `is_additional_stop` is true | Not in the planned timetable, so the planning case never knows about them | 0.17% |
| Both planned times null | Nothing to plan or predict | 5 rows |
| Rows without `station_name` | Keep, and label with the EVA code (already done in `router_core.py`) | 0.06% |

Canceled rows are **not dropped**. They stay in the table with a cancel label, because
"will this stop be canceled" is part of the risk. They are excluded only from the delay-regression
target.

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
| `stop_idx`, `n_stops`, `stop_frac` | From `train_line_station_num` and the run length | Position in the run. Later stops tend to collect more delay. |
| `prev_stop_dep_delay` | **Not used.** It is live state (step two in the plan). | Would leak in the planning case |
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
1. First build `stops` with schedule-only columns (section 6, without history). This is enough to
   start the first TabPFN experiments.
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

1. **`stops`**: one row per stop event after the filters, with the columns of section 6 and the
   history features. Partitioned by `run_day`.
2. **`hops`**: pairs of consecutive stops of one run (feeds the router and leg evaluation).
3. **`transfers`**: pairs of (arriving run, departing run) at the same station with a planned gap
   of at least 5 minutes and at most about 60 minutes (feeds connection evaluation).

Splits follow `plan.md`: context pool Aug 1 to Sep 16, validation Sep 17 to 23, test Sep 24 to 30,
split by `run_day`.

## 9. Open questions

1. Keep replacement trains in the timetable (the router might use them) or drop them? Proposed: drop.
2. Merge stations by name, as planned, or by EVA? Proposed: name, as in `plan.md`.
3. ~~How to feed 5k stations to TabPFN~~ Decided: raw category, together with `train_key`. The
   shared per-request context holds only a few dozen stations. See "Feature lists" in section 6.
4. Runs that started on July 31 appear in the August file. Keep them (they belong to the context pool) or cut at `run_day >= Aug 1`? Proposed: cut by `run_day`.
5. Is a September-only check enough for the August schema, or do we want a quick equality check of
   schema and filters across both files? Schemas are already identical.
