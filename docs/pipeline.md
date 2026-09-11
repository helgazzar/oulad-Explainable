# pipeline.py

Turns the seven OULAD CSVs into a preprocessed train/test split that any model can fit on. Every model file calls `pipeline.run()` rather than the raw data directly, so feature changes happen here.

`helperFunctions.py` holds three small utilities the pipeline uses (`_resolve` to find a CSV, `_ordinal` to map bands to numbers, `_safe_div` to divide without inf/NaN).

## Running it

```
cd src
python pipeline.py
```

Prints the split shapes and the at-risk rate for train and test. Everything else imports it.

## Config

`PreprocessConfig` is a frozen dataclass; pass a modified copy to change a setting.

| Setting | Default | What it does |
|---|---|---|
| `data_dirs` | `../data` | Where the CSVs are searched for, in order |
| `cutoff_day` | 32 | Day of the presentation the prediction is made on. Nothing after this day is used |
| `include_withdrawn` | True | Whether `Withdrawn` counts as at-risk alongside `Fail` |
| `test_size` | 0.20 | Hold-out fraction, stratified on the target |
| `random_state` | 42 | Seed for the split and the imputer |
| `vle_chunksize` | 2,000,000 | Rows per chunk when streaming `studentVle.csv` |
| `keys` | module, presentation, student | The three columns that identify one student-in-a-presentation |

```python
from dataclasses import replace
cfg = replace(PreprocessConfig(), cutoff_day=60)
X_train, X_test, y_train, y_test, preprocessor, feature_columns = run(cfg)
```

## The leakage rule

`cutoff_day` is the one rule everything else follows. The model is meant to flag a student on day 32 using only what an instructor could see on day 32, so:

- clicks with `date > cutoff_day` are dropped before aggregation
- assessments with a due date after the cutoff are dropped, and so are submissions after it
- `date_unregistration` only counts if it is on or before the cutoff
- `final_result` is used to build `y` and is not a feature

## What each table becomes

Each CSV has a `load_*` function (just `read_csv`) and a `process_*` function that returns one row per student per presentation.

| Table | Features produced |
|---|---|
| `courses.csv` | `module_presentation_length` |
| `studentRegistration.csv` | `registration_lead_days`, `registered_late`, `unregistered_before_cutoff` |
| `studentInfo.csv` | `imd_band_ordinal`, `age_band_ordinal`, `highest_education_ordinal`, `disability_flag`, `gender_flag`, `num_of_prev_attempts`, `studied_credits`, `region` (kept as a string), `final_result` (target only) |
| `vle.csv` | lookup of `id_site` to `activity_type`, merged into the click data |
| `studentVle.csv` | `total_clicks`, `active_days`, `distinct_sites`, `avg_clicks_per_active_day`, `engagement_score`, one `clicks_<activity_type>` column per activity type |
| `assessments.csv` | cutoff-filtered list of due assessments; `n_expected_assessments` per presentation |
| `studentAssessment.csv` | `n_submitted`, `mean_score`, `weighted_score`, `avg_days_before_due`, `pct_banked`, `submission_rate` |

`build_feature_table()` left-merges all of these onto `studentInfo` so a student with no clicks or submissions still gets a row, then adds `progress_fraction` (cutoff day divided by presentation length). Click and submission counts that are missing after the merge are set to 0, because no rows means no activity, not unknown activity.

`studentVle.csv` is 10.6M rows, so it is read in chunks and filtered on the cutoff as it goes.

## Target

`build_target()` makes `at_risk`: 1 for `Fail` (and `Withdrawn` if `include_withdrawn`), 0 for `Pass` and `Distinction`. With withdrawn included the classes are close to balanced (about 47% at-risk).

## sklearn preprocessing

`build_preprocessing_pipeline()` returns one `ColumnTransformer`:

- numeric columns: `IterativeImputer` (MICE style, each missing value regressed on the other columns) then `StandardScaler`
- `region`: most-frequent impute then `OneHotEncoder`

`NUMERIC_FEATURES` lists the fixed numeric columns. The `clicks_<activity_type>` columns depend on which activity types show up before the cutoff.

`run()` fits the transformer on the train split only and applies it to test. Feature names after transformation come from `preprocessor.get_feature_names_out()`; `models.feature_names()` strips the `numeric__` / `categorical__` prefix.

## ex Note

- The `IterativeImputer` is the part most likely to change results between sklearn versions since it is still marked experimental.
- `unregistered_before_cutoff` is the strongest single feature in both models. Next steps to test are dropping this feature. 
