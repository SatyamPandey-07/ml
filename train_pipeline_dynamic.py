"""
Dynamic retrainable credit scoring pipeline
"""

import os
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, precision_recall_curve
)
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

# --- Optional libraries ---
try:
    from lightgbm import LGBMClassifier
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# --- Paths / config ---
DATA_IN = "beneficiary_credit_features_with_targets.csv"
OUT_DIR = "credit_scoring_pipeline_outputs"
os.makedirs(OUT_DIR, exist_ok=True)
METRICS_LOG = os.path.join(OUT_DIR, "training_metrics_log.csv")

# --- Keyword patterns ---
CONSUMPTION_KW = ["energy", "electricity", "recharge", "mobile", "bill", "utility", "consumption", "usage"]
REPAYMENT_KW = ["repay", "repayment", "overdue", "delinquent", "paid", "installment", "punctuality", "delay"]
LOAN_KW = ["loan", "disbursal", "principal", "balance", "outstanding", "utilization"]
REPEAT_KW = ["repeat", "num_past_loans", "num_loans"]

# --- Utility helpers ---
def _detect_columns(df):
    consumption_cols = [c for c in df.columns if any(k in c.lower() for k in CONSUMPTION_KW)]
    repayment_cols = [c for c in df.columns if any(k in c.lower() for k in REPAYMENT_KW)]
    loan_cols = [c for c in df.columns if any(k in c.lower() for k in LOAN_KW)]
    repeat_cols = [c for c in df.columns if any(k in c.lower() for k in REPEAT_KW)]
    return consumption_cols, repayment_cols, loan_cols, repeat_cols

def _engineer_features(df, consumption_cols, repayment_cols, loan_cols, repeat_cols):
    df_fe = df.copy()
    if repayment_cols:
        df_fe["repayment_mean"] = df_fe[repayment_cols].mean(axis=1, numeric_only=True)
        df_fe["repayment_median"] = df_fe[repayment_cols].median(axis=1, numeric_only=True)
        df_fe["repayment_std"] = df_fe[repayment_cols].std(axis=1, numeric_only=True).fillna(0)
    else:
        df_fe["repayment_mean"], df_fe["repayment_median"], df_fe["repayment_std"] = 0.8, 0.8, 0.0

    if consumption_cols:
        df_fe["consumption_mean"] = df_fe[consumption_cols].mean(axis=1, numeric_only=True)
        df_fe["consumption_median"] = df_fe[consumption_cols].median(axis=1, numeric_only=True)
    else:
        df_fe["consumption_mean"], df_fe["consumption_median"] = np.nan, np.nan

    if "outstanding_balance" in df_fe.columns and "loan_amount" in df_fe.columns:
        df_fe["utilization"] = df_fe["outstanding_balance"] / (df_fe["loan_amount"] + 1e-8)
    elif loan_cols:
        df_fe["utilization"] = df_fe[loan_cols].max(axis=1)
    else:
        df_fe["utilization"] = np.nan

    df_fe["repeat_loan_count"] = df_fe[repeat_cols].sum(axis=1, numeric_only=True) if repeat_cols else df_fe.get("num_past_loans", 0)
    for c in ["repayment_mean", "repayment_median", "consumption_mean", "consumption_median", "utilization"]:
        df_fe[c] = df_fe[c].fillna(df_fe[c].median())
    df_fe["repayment_std"] = df_fe["repayment_std"].fillna(0)
    df_fe["repeat_loan_count"] = df_fe["repeat_loan_count"].fillna(0)
    return df_fe

def _select_features(df_fe):
    engineered = [
        "repayment_mean", "repayment_median", "repayment_std",
        "consumption_mean", "consumption_median", "utilization", "repeat_loan_count"
    ]
    categorical_low_card = [c for c in df_fe.select_dtypes(exclude=[np.number]).columns if df_fe[c].nunique() < 30]
    return engineered + [c for c in categorical_low_card if c not in engineered]

def _f2_score(p, r):
    return (5 * p * r) / (4 * p + r) if (p + r) != 0 else 0.0

def _tune_threshold_by_f2(y_true, y_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    best_threshold, best_f2 = 0.5, -1
    for i in range(len(thresholds)):
        f2 = _f2_score(precisions[i], recalls[i])
        if f2 > best_f2:
            best_f2, best_threshold = f2, thresholds[i]
    return best_threshold, best_f2

# --- Core training pipeline ---
def train_pipeline(df):
    consumption_cols, repayment_cols, loan_cols, repeat_cols = _detect_columns(df)
    df_fe = _engineer_features(df, consumption_cols, repayment_cols, loan_cols, repeat_cols)
    features = _select_features(df_fe)
    X, y = df_fe[features].copy(), df_fe["default_flag"].copy()
    if y.dtype == object:
        y = y.map(lambda v: 1 if str(v).strip().lower() in ("1", "yes", "true", "y") else 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in numeric_cols]
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    preprocessor = ColumnTransformer([("num", num_pipe, numeric_cols), ("cat", cat_pipe, cat_cols)])

    classifiers = {}
    if LGB_AVAILABLE:
        classifiers["lgbm"] = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=42)
    else:
        classifiers["rf"] = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    if XGB_AVAILABLE:
        classifiers["xgb"] = XGBClassifier(n_estimators=200, learning_rate=0.05, random_state=42, use_label_encoder=False, eval_metric="logloss")

    candidate_best, candidate_cvrecall = {}, {}
    for name, clf in classifiers.items():
        imb_pipe = ImbPipeline([("preprocessor", preprocessor), ("smote", SMOTE(random_state=42)), ("clf", clf)])
        grid = {
            "lgbm": {"clf__n_estimators": [100, 200], "clf__num_leaves": [15, 31]},
            "xgb": {"clf__n_estimators": [100, 200], "clf__max_depth": [4, 6]},
            "rf": {"clf__n_estimators": [100, 200], "clf__max_depth": [6, None]}
        }[name]
        gs = GridSearchCV(imb_pipe, grid, scoring="recall", cv=3, n_jobs=-1)
        gs.fit(X_train, y_train)
        candidate_best[name], candidate_cvrecall[name] = gs.best_estimator_, gs.best_score_

    best_name = max(candidate_cvrecall, key=candidate_cvrecall.get)
    best_pipeline = candidate_best[best_name]
    best_pipeline.fit(X_train, y_train)

    preproc = best_pipeline.named_steps["preprocessor"]
    clf_step = best_pipeline.named_steps["clf"]
    try:
        X_cal_train, X_cal_val, y_cal_train, y_cal_val = train_test_split(X_train, y_train, test_size=0.15, stratify=y_train, random_state=42)
        X_cal_val_trans = preproc.transform(X_cal_val)
        calibrated = CalibratedClassifierCV(base_estimator=clf_step, method="sigmoid", cv="prefit")
        calibrated.fit(X_cal_val_trans, y_cal_val)
        final_pipeline = Pipeline([("preprocessor", preproc), ("clf", calibrated)])
    except Exception:
        final_pipeline = best_pipeline

    y_proba_test = final_pipeline.predict_proba(X_test)[:, 1]
    y_pred_default = (y_proba_test >= 0.5).astype(int)
    default_report = classification_report(y_test, y_pred_default, digits=3, output_dict=True)
    best_threshold, best_f2 = _tune_threshold_by_f2(y_test, y_proba_test)
    y_pred_tuned = (y_proba_test >= best_threshold).astype(int)
    tuned_report = classification_report(y_test, y_pred_tuned, digits=3, output_dict=True)

    metrics = {
        "model": best_name,
        "cv_recall": float(candidate_cvrecall[best_name]),
        "tuned_threshold": float(best_threshold),
        "tuned_f2": float(best_f2),
        "tuned_precision_class1": float(tuned_report["1"]["precision"]),
        "tuned_recall_class1": float(tuned_report["1"]["recall"]),
        "tuned_f1_class1": float(tuned_report["1"]["f1-score"]),
        "train_time_utc": datetime.utcnow().isoformat() + "Z"
    }

    version = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    model_path = os.path.join(OUT_DIR, f"best_balanced_model_v{version}.pkl")
    joblib.dump({"pipeline": final_pipeline, "features": features, "threshold": best_threshold, "metrics": metrics}, model_path)

    print(f"\n✅ Model trained & saved at: {model_path}")
    print(json.dumps(metrics, indent=2))
    return {"pipeline": final_pipeline, "features": features, "threshold": best_threshold, "metrics": metrics, "artifact": model_path}

# --- Public functions ---
def retrain_from_csv():
    if not os.path.exists(DATA_IN):
        raise FileNotFoundError(f"{DATA_IN} not found.")
    df = pd.read_csv(DATA_IN)
    return train_pipeline(df)

def update_with_new_data(new_df, save_csv=True):
    if not isinstance(new_df, pd.DataFrame):
        raise ValueError("new_df must be a pandas DataFrame")
    if os.path.exists(DATA_IN):
        orig = pd.read_csv(DATA_IN)
        combined = pd.concat([orig, new_df], ignore_index=True)
    else:
        combined = new_df.copy()
    if save_csv:
        combined.to_csv(DATA_IN, index=False)
        print(f"Appended {len(new_df)} rows to {DATA_IN} (total {len(combined)}).")
    else:
        print(f"Retraining in-memory on {len(combined)} rows.")
    return train_pipeline(combined)

# --- Script entrypoint ---
if __name__ == "__main__":
    print("Running retrain_from_csv() ...")
    r = retrain_from_csv()
    print("Done. Latest metrics:", r["metrics"])