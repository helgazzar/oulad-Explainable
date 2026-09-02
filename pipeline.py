from __future__ import annotations
from helperFunctions import _resolve, _ordinal, _safe_div
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# pandas for data handling
import pandas as pd
# scikit-learn column-wise transformer orchestration
from sklearn.compose import ColumnTransformer
# IterativeImputer: MICE-style imputation
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
# pipeline chains imputation and scaling into one fitted object
from sklearn.pipeline import Pipeline
# encoders and scalers for the final feature matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# stratified split preserves class balance in train/test
from sklearn.model_selection import train_test_split


# Use to set default values for the Preprocessor class.
# passing a different values overwrites the defaults.
@dataclass(frozen=True)
class PreprocessConfig:

    data_dirs: tuple[Path, ...] = (Path(__file__).resolve().parent / "data",)

    # day of the module presentation at which prediction is made (leakage boundary)
    cutoff_day: int = 30

    # whether Withdrawn students count as the positive (at-risk) class
    include_withdrawn: bool = True

    # proportion of records held out for testing
    test_size: float = 0.20

    # static seed for reproducibility
    random_state: int = 42

    # rows per chunk when streaming the 10.6M-row studentVle.csv
    vle_chunksize: int = 2_000_000

    # the three columns that uniquely identify a student-in-a-presentation
    keys: tuple[str, ...] = ("code_module", "code_presentation", "id_student")


### courses processing ### 
# courses.csv holds one row per (code_module, code_presentation)
# giving the calendar length of that presentation. Used later to turn the fixed
# cutoff_day into a progress fraction that's comparable across presentations of
# different lengths.
def load_courses(cfg: PreprocessConfig) -> pd.DataFrame:
    """Read courses.csv."""
    return pd.read_csv(_resolve("courses.csv", cfg))

def process_courses(courses: pd.DataFrame) -> pd.DataFrame:
    """One row per (code_module, code_presentation) with the presentation length."""
    out = courses[["code_module", "code_presentation", "module_presentation_length"]].copy()
    out["module_presentation_length"] = out["module_presentation_length"].astype("float32")
    return out


### registration processing ###
# studentRegistration.csv records when a student joined
# and (if applicable) left a presentation. Turned into lead-time and
# withdrew-before-cutoff features rather than passed through as raw dates.
def load_student_registration(cfg: PreprocessConfig) -> pd.DataFrame:
    """Read studentRegistration.csv."""
    return pd.read_csv(_resolve("studentRegistration.csv", cfg))

def process_registration(reg: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    """Per-student registration timing features, keyed like everything else."""
    keys = list(cfg.keys)
    out = reg[keys + ["date_registration", "date_unregistration"]].copy()

    # registration_lead_days: positive means the student registered before
    # day 0 (the presentation's official start); negative means they joined late
    out["registration_lead_days"] = (-out["date_registration"]).astype("float32")

    # registered_late: joined on or after the presentation's start day
    out["registered_late"] = (out["date_registration"] >= 0).astype("float32")

    # unregistered_before_cutoff: NaN date_unregistration means the student never unregistered 
    # so it must not be treated as "unregistered on day 0"
    out["unregistered_before_cutoff"] = (
        out["date_unregistration"].notna() & (out["date_unregistration"] <= cfg.cutoff_day)
    ).astype("float32")

    return out.drop(columns=["date_registration", "date_unregistration"])


### demographics processing ###
# studentInfo.csv holds one row per student per presentation 
# with static demographics plus the final_result label.
def load_student_info(cfg: PreprocessConfig) -> pd.DataFrame:
    """Read studentInfo.csv."""
    return pd.read_csv(_resolve("studentInfo.csv", cfg))

# mapped OULAD banded demographic columns
_IMD_BAND_ORDER = {
    "0-10%": 0, "10-20": 1, "20-30%": 2, "30-40%": 3, "40-50%": 4,
    "50-60%": 5, "60-70%": 6, "70-80%": 7, "80-90%": 8, "90-100%": 9,
}
_AGE_BAND_ORDER = {"0-35": 0, "35-55": 1, "55<=": 2}
_EDUCATION_ORDER = {
    "No Formal quals": 0,
    "Lower Than A Level": 1,
    "A Level or Equivalent": 2,
    "HE Qualification": 3,
    "Post Graduate Qualification": 4,
}

def process_student_info(info: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    """Demographic features plus final_result."""
    keys = list(cfg.keys)
    out = info.copy()

    out["imd_band_ordinal"] = _ordinal(out["imd_band"], _IMD_BAND_ORDER)
    out["age_band_ordinal"] = _ordinal(out["age_band"], _AGE_BAND_ORDER)
    out["highest_education_ordinal"] = _ordinal(out["highest_education"], _EDUCATION_ORDER)
    # binary mapping
    out["disability_flag"] = _ordinal(out["disability"], {"N": 0, "Y": 1})
    out["gender_flag"] = _ordinal(out["gender"], {"F": 0, "M": 1})
    # no change
    out["num_of_prev_attempts"] = out["num_of_prev_attempts"].astype("float32")
    out["studied_credits"] = out["studied_credits"].astype("float32")

    # since region is nominal keep as a string for the
    # categorical branch of the sklearn pipeline
    return out[keys + [
        "region", "imd_band_ordinal", "age_band_ordinal", "highest_education_ordinal",
        "disability_flag", "gender_flag", "num_of_prev_attempts", "studied_credits",
        "final_result",
    ]]


### vle processing ###
# vle.csv is metadata about each VLE material/site (activity_type).
def load_vle(cfg: PreprocessConfig) -> pd.DataFrame:
    """Read vle.csv."""
    return pd.read_csv(_resolve("vle.csv", cfg))

def process_vle_metadata(vle: pd.DataFrame) -> pd.DataFrame:
    """Lightweight lookup: id_site -> (code_module, code_presentation, activity_type)."""
    return vle[["id_site", "code_module", "code_presentation", "activity_type"]].copy()


### studentVle processing ###
# chunks to keep memory bounded, restricted to date <= cutoff_day 
# set to one engagement-summary row per student per presentation.
def process_student_vle(cfg: PreprocessConfig, vle_meta: pd.DataFrame) -> pd.DataFrame:
    """Chunked read of clicks up to cfg.cutoff_day, aggregated once into engagement features."""
    keys = list(cfg.keys)
    path = _resolve("studentVle.csv", cfg)

    # chunksize only needs to bound memory while filtering the full 10.6M-row
    kept_parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, chunksize=cfg.vle_chunksize):

        # drop any click that happened after the cutoff day
        chunk = chunk[chunk["date"] <= cfg.cutoff_day]
        if not chunk.empty:
            kept_parts.append(chunk)

    if not kept_parts:

        # no clickstream activity before the cutoff for any student
        return pd.DataFrame(columns=keys + ["total_clicks", "active_days", "distinct_sites", "avg_clicks_per_active_day", "engagement_score"])

    # bring in activity_type
    clicks = pd.concat(kept_parts, ignore_index=True).merge(
        vle_meta, on=["id_site", "code_module", "code_presentation"], how="left"
    )

    sites = clicks.groupby(keys, as_index=False)["id_site"].nunique().rename(columns={"id_site": "distinct_sites"})
    activity = clicks.groupby(keys + ["activity_type"], as_index=False)["sum_click"].sum()

    totals = clicks.groupby(keys, as_index=False).agg(
        total_clicks=("sum_click", "sum"),
        active_days=("date", "nunique"),
    )
    totals["total_clicks"] = totals["total_clicks"].astype("float32")
    totals["active_days"] = totals["active_days"].astype("float32")
    totals["avg_clicks_per_active_day"] = _safe_div(totals["total_clicks"], totals["active_days"])

    # total engagement score: click volume weighted by how consistently the student showed up
    totals["engagement_score"] = (
        totals["total_clicks"] * (totals["active_days"] / max(cfg.cutoff_day, 1))
    ).astype("float32")

    # capture per-activity-type click counts
    activity_wide = (
        activity.pivot_table(index=keys, columns="activity_type", values="sum_click", fill_value=0)
        .add_prefix("clicks_")
        .reset_index()
    )

    return totals.merge(activity_wide, on=keys, how="left").merge(sites, on=keys, how="left")


### assessments processing ###
# assessments.csv is static metadata (type, due date, weight) about each assessment. 
def load_assessments(cfg: PreprocessConfig) -> pd.DataFrame:
    """Read assessments.csv."""
    return pd.read_csv(_resolve("assessments.csv", cfg))

# Restrict to assessments due on/before the cutoff so joining student submissions do not pull in future assessment.
def process_assessments(assessments: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    """Assessment metadata restricted to items already due by cfg.cutoff_day."""
    out = assessments.copy()
    out["weight"] = out["weight"].astype("float32")

    # NaN dates, which are mostly end-of-module exams, fail the comparison and drop out
    out = out[out["date"].notna() & (out["date"] <= cfg.cutoff_day)]
    return out[["code_module", "code_presentation", "id_assessment", "assessment_type", "date", "weight"]]

def _expected_assessment_counts(assessments_processed: pd.DataFrame) -> pd.DataFrame:
    """How many assessments were due by the cutoff, per (module, presentation) - the denominator for submission_rate."""
    return (
        assessments_processed.groupby(["code_module", "code_presentation"], as_index=False)["id_assessment"]
        .nunique()
        .rename(columns={"id_assessment": "n_expected_assessments"})
    )


### studentAssessment processing ###
# studentAssessment.csv holds individual
# submissions. Inner-joined to the cutoff-filtered assessments table (so only
# already-due assessments count), then filtered again on submission date, and
# set to one row per student per presentation.
def load_student_assessment(cfg: PreprocessConfig) -> pd.DataFrame:
    """Read studentAssessment.csv."""
    return pd.read_csv(_resolve("studentAssessment.csv", cfg))

def process_student_assessment(
    student_assessment: pd.DataFrame,
    assessments_processed: pd.DataFrame,
    cfg: PreprocessConfig,
) -> pd.DataFrame:
    """Per-student submission-behavior features: volume, scores, timing, and completion rate."""
    keys = list(cfg.keys)

    # inner join restricts submissions to assessments within the cutoff window
    merged = student_assessment.merge(assessments_processed, on="id_assessment", how="inner")
    # drop submissions made after the cutoff day too
    merged = merged[merged["date_submitted"] <= cfg.cutoff_day].copy()
    # submission timing: positive = submitted before the due date, negative = late
    merged["days_before_due"] = (merged["date"] - merged["date_submitted"]).astype("float32")
    # weight-adjusted score contribution
    merged["_weighted_score"] = merged["score"] * merged["weight"]

    # aggregate + merge to add the weighted-score sums
    agg = merged.groupby(keys, as_index=False).agg(
        n_submitted=("id_assessment", "count"),
        mean_score=("score", "mean"),
        avg_days_before_due=("days_before_due", "mean"),
        pct_banked=("is_banked", "mean"),
        _score_sum=("_weighted_score", "sum"),
        _weight_sum=("weight", "sum"),
    )
    agg["mean_score"] = agg["mean_score"].astype("float32")
    agg["avg_days_before_due"] = agg["avg_days_before_due"].astype("float32")
    agg["pct_banked"] = agg["pct_banked"].astype("float32")
    agg["weighted_score"] = _safe_div(agg["_score_sum"], agg["_weight_sum"])
    agg = agg.drop(columns=["_score_sum", "_weight_sum"])

    # submission_rate: how much of the work due so far was actually turned in
    n_expected = _expected_assessment_counts(assessments_processed)
    agg = agg.merge(n_expected, on=["code_module", "code_presentation"], how="left")
    agg["submission_rate"] = _safe_div(agg["n_submitted"].astype("float32"), agg["n_expected_assessments"].astype("float32"))

    return agg.drop(columns=["n_expected_assessments"])


### Assemle features ###
# merge every (module, presentation, student) table onto studentInfo
def build_feature_table(cfg: PreprocessConfig) -> pd.DataFrame:
    """Load, process, and merge all seven OULAD tables into one feature-per-row frame."""
    keys = list(cfg.keys)

    courses = process_courses(load_courses(cfg))
    registration = process_registration(load_student_registration(cfg), cfg)
    info = process_student_info(load_student_info(cfg), cfg)
    vle_meta = process_vle_metadata(load_vle(cfg))
    clickstream = process_student_vle(cfg, vle_meta)
    assessments_processed = process_assessments(load_assessments(cfg), cfg)
    submissions = process_student_assessment(load_student_assessment(cfg), assessments_processed, cfg)

    # studentInfo - left-merging everything else onto it allows a
    # student with no clicks/submissions before the cutoff still gets a row
    features = info.merge(courses, on=["code_module", "code_presentation"], how="left")
    features = features.merge(registration, on=keys, how="left")
    features = features.merge(clickstream, on=keys, how="left")
    features = features.merge(submissions, on=keys, how="left")

    # progress_fraction: how far into the presentation the cutoff sits, so a
    # fixed cutoff_day is comparable across presentations of different lengths
    features["progress_fraction"] = _safe_div(
        pd.Series(cfg.cutoff_day, index=features.index).astype("float32"),
        features["module_presentation_length"],
    )

    # process rows with no recorded clicks/submissions before the cutoff as 0.0
    click_cols = [c for c in clickstream.columns if c not in keys]
    zero_fill_cols = click_cols + ["n_submitted", "pct_banked", "submission_rate"]
    for col in zero_fill_cols:
        if col in features.columns:
            features[col] = features[col].fillna(0.0)

    return features


### Target constructor ### 
# final_result produces y
def build_target(features: pd.DataFrame, cfg: PreprocessConfig) -> pd.Series:
    """Binarize final_result into an at_risk label (Fail, plus Withdrawn if configured)."""
    at_risk_labels = {"Fail"}
    if cfg.include_withdrawn:
        at_risk_labels.add("Withdrawn")
    return features["final_result"].isin(at_risk_labels).astype("int8").rename("at_risk")


### sklearn preprocessing ### 
# numeric imputation + scaling and categorical imputation +
# one-hot encoding, orchestrated by a single ColumnTransformer. 

# fixed numeric features present regardless of which activity types appear before the cutoff
NUMERIC_FEATURES: tuple[str, ...] = (
    "imd_band_ordinal", "age_band_ordinal", "highest_education_ordinal",
    "disability_flag", "gender_flag", "num_of_prev_attempts", "studied_credits",
    "module_presentation_length", "progress_fraction",
    "registration_lead_days", "registered_late", "unregistered_before_cutoff",
    "total_clicks", "active_days", "distinct_sites", "avg_clicks_per_active_day", "engagement_score",
    "n_submitted", "mean_score", "weighted_score", "avg_days_before_due", "pct_banked", "submission_rate",
)
CATEGORICAL_FEATURES: tuple[str, ...] = ("region",)

def build_preprocessing_pipeline(
    numeric_features: Iterable[str] = NUMERIC_FEATURES,
    categorical_features: Iterable[str] = CATEGORICAL_FEATURES,
    random_state: int = 42,
) -> ColumnTransformer:
    """MICE-style iterative impute + scale numeric columns; mode-impute + one-hot categorical columns."""
    numeric_pipeline = Pipeline(steps=[
        ("impute", IterativeImputer(random_state=random_state)),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, list(numeric_features)),
        ("categorical", categorical_pipeline, list(categorical_features)),
    ])


### Orchestrator ###
# tie feature assembly, target construction, the train/test split,
# and the sklearn preprocessing pipeline together into a single function.
def run(cfg: PreprocessConfig = PreprocessConfig()):
    """Build features, split, and fit/apply the preprocessing pipeline. Returns
    (X_train, X_test, y_train, y_test, fitted_preprocessor, feature_columns)."""
    features = build_feature_table(cfg)
    y = build_target(features, cfg)

    # per-activity-type click columns 
    click_cols = sorted(c for c in features.columns if c.startswith("clicks_"))
    numeric_features = list(NUMERIC_FEATURES) + click_cols

    feature_columns = numeric_features + list(CATEGORICAL_FEATURES)
    X = features[feature_columns]

    # stratified split preserves the at-risk class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y,
    )

    # build pipeline
    preprocessor = build_preprocessing_pipeline(numeric_features, CATEGORICAL_FEATURES, random_state=cfg.random_state)
    # fit the training split
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    return X_train_t, X_test_t, y_train, y_test, preprocessor, feature_columns


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, preprocessor, feature_columns = run()
    print(f"train: {X_train.shape}, test: {X_test.shape}")
    print(f"at-risk rate - train: {y_train.mean():.3f}, test: {y_test.mean():.3f}")
