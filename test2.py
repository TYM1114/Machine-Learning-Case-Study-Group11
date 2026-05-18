from __future__ import annotations

import re
import warnings
from pathlib import Path

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent / "home-credit-default-risk"
RANDOM_STATE = 42
N_SPLITS = 5
DEBUG = False

def load_csv(name: str, nrows: int | None = None) -> pd.DataFrame:
    if DEBUG and nrows is None:
        nrows = 1000
    path = DATA_DIR / name
    if not path.exists():
        print(f"Warning: {path} not found.")
        return pd.DataFrame()
    print(f"Loading {path.name} ...")
    df = pd.read_csv(path, nrows=nrows)
    for key in ("SK_ID_CURR", "SK_ID_PREV", "SK_ID_BUREAU"):
        if key in df.columns:
            df[key] = df[key].astype("int64")
    return df

def clean_column_name(name: str) -> str:
    name = str(name)
    name = re.sub(r"[^0-9a-zA-Z_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name.upper()

def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col]) and col not in ["TARGET", "SK_ID_CURR"]]
    if not cat_cols:
        return df
    return pd.get_dummies(df, columns=cat_cols, dummy_na=True)

def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        col_type = df[col].dtype
        if col.startswith("SK_ID_"):
            df[col] = df[col].astype("int64")
        elif pd.api.types.is_integer_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_float_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif col_type == bool:
            df[col] = df[col].astype(np.int8)
    return df

def aggregate_table(
    df: pd.DataFrame,
    group_key: str,
    prefix: str,
    numeric_aggs: list[str] | None = None,
    categorical_aggs: list[str] | None = None,
) -> pd.DataFrame:
    if df.empty: return pd.DataFrame()
    numeric_aggs = numeric_aggs or ["mean", "max", "min", "sum", "var"]
    categorical_aggs = categorical_aggs or ["mean"]

    num_cols = [col for col in df.columns if col != group_key and pd.api.types.is_numeric_dtype(df[col]) and not set(df[col].dropna().unique()).issubset({0, 1})]
    cat_cols = [col for col in df.columns if col != group_key and (not pd.api.types.is_numeric_dtype(df[col]) or set(df[col].dropna().unique()).issubset({0, 1}))]
    
    if any(not pd.api.types.is_numeric_dtype(df[col]) for col in cat_cols):
        df = one_hot_encode(df)
        cat_cols = [c for c in df.columns if any(c.startswith(oc) for oc in cat_cols) and c != group_key]

    agg_dict = {col: numeric_aggs for col in num_cols}
    for col in cat_cols: agg_dict[col] = categorical_aggs

    grouped = df.groupby(group_key).agg(agg_dict)
    grouped.columns = [f"{prefix}_{clean_column_name(col)}_{stat.upper()}" for col, stat in grouped.columns]
    grouped.reset_index(inplace=True)
    return reduce_memory(grouped)

def add_application_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["DAYS_EMPLOYED"].replace(365243, np.nan, inplace=True)
    df["DAYS_LAST_PHONE_CHANGE"].replace(0, np.nan, inplace=True)

    # 2nd Place Strategy: Interest Rates and Ratios
    df["APP_ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["APP_CREDIT_ANNUITY_RATIO"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
    df["APP_GOODS_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / df["AMT_CREDIT"]
    df["APP_EMPLOYED_BIRTH_RATIO"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    
    # External Sources (1st Place Key)
    df["APP_EXT_SOURCES_PROD"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
    df["APP_EXT_SOURCES_MEAN"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    df["APP_EXT_SOURCES_NAN_COUNT"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].isnull().sum(axis=1)
    
    return df

def neighbors_target_mean(train_df: pd.DataFrame, test_df: pd.DataFrame, n_neighbors: int = 500) -> tuple[pd.Series, pd.Series]:
    cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "APP_CREDIT_ANNUITY_RATIO"]
    data = pd.concat([train_df[cols + ["TARGET"]], test_df[cols]], axis=0)
    data_imputed = data[cols].fillna(data[cols].median())
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_imputed)
    
    train_scaled, test_scaled = data_scaled[:len(train_df)], data_scaled[len(train_df):]
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=-1)
    
    oof_neighbors = np.zeros(len(train_df))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, val_idx in skf.split(train_df, train_df["TARGET"]):
        knn.fit(train_scaled[tr_idx], train_df["TARGET"].iloc[tr_idx])
        oof_neighbors[val_idx] = knn.predict(train_scaled[val_idx])
    
    knn.fit(train_scaled, train_df["TARGET"])
    test_neighbors = knn.predict(test_scaled)
    return pd.Series(oof_neighbors, index=train_df.index), pd.Series(test_neighbors, index=test_df.index)

def bureau_features() -> pd.DataFrame:
    bureau = load_csv("bureau.csv")
    bb = load_csv("bureau_balance.csv")
    if bureau.empty: return pd.DataFrame()

    if not bb.empty:
        bb = one_hot_encode(bb)
        bb_agg = bb.groupby("SK_ID_BUREAU").agg({"MONTHS_BALANCE": ["min", "max", "size"], **{col: ["mean"] for col in bb.columns if col.startswith("STATUS_")}})
        bb_agg.columns = [f"BB_{clean_column_name(col)}_{stat.upper()}" for col, stat in bb_agg.columns]
        bureau = bureau.join(bb_agg, how="left", on="SK_ID_BUREAU")

    # 2nd Place: Bureau Interest and Trend
    bureau["BURO_CREDIT_ACTIVE_BINARY"] = (bureau["CREDIT_ACTIVE"] == "Active").astype(int)
    full_agg = aggregate_table(bureau.drop(columns=["SK_ID_BUREAU"] if "SK_ID_BUREAU" in bureau.columns else []), "SK_ID_CURR", "BURO")
    
    recent_bureau = bureau[bureau["DAYS_CREDIT"] >= -365]
    recent_agg = aggregate_table(recent_bureau, "SK_ID_CURR", "BURO_RECENT")
    
    if not recent_agg.empty:
        full_agg = full_agg.merge(recent_agg, on="SK_ID_CURR", how="left")
        # 2nd Place: Trend Ratio
        full_agg["BURO_TREND_DEBT_RATIO"] = full_agg["BURO_RECENT_AMT_CREDIT_SUM_DEBT_MEAN"] / (full_agg["BURO_AMT_CREDIT_SUM_DEBT_MEAN"] + 1)
    return full_agg

def previous_application_features() -> pd.DataFrame:
    prev = load_csv("previous_application.csv")
    if prev.empty: return pd.DataFrame()
    for col in [c for c in prev.columns if c.startswith("DAYS_")]: prev[col].replace(365243, np.nan, inplace=True)

    # Interest rate estimate
    prev["PREV_INTEREST_EST"] = (prev["AMT_ANNUITY"] * prev["CNT_PAYMENT"]) / (prev["AMT_CREDIT"] + 1) - 1
    
    full_agg = aggregate_table(prev.drop(columns=["SK_ID_PREV"]), "SK_ID_CURR", "PREV")
    recent_prev = prev[prev["DAYS_DECISION"] >= -365]
    recent_agg = aggregate_table(recent_prev, "SK_ID_CURR", "PREV_RECENT")
    
    if not recent_agg.empty:
        full_agg = full_agg.merge(recent_agg, on="SK_ID_CURR", how="left")
    return full_agg

def pos_cash_features() -> pd.DataFrame:
    pos = load_csv("POS_CASH_balance.csv")
    if pos.empty: return pd.DataFrame()
    full_agg = aggregate_table(pos.drop(columns=["SK_ID_PREV"]), "SK_ID_CURR", "POS")
    recent_agg = aggregate_table(pos[pos["MONTHS_BALANCE"] >= -12], "SK_ID_CURR", "POS_RECENT")
    return full_agg.merge(recent_agg, on="SK_ID_CURR", how="left") if not recent_agg.empty else full_agg

def installments_features() -> pd.DataFrame:
    ins = load_csv("installments_payments.csv")
    if ins.empty: return pd.DataFrame()
    ins["PAYMENT_PERC"] = ins["AMT_PAYMENT"] / (ins["AMT_INSTALMENT"] + 1)
    ins["DPD"] = (ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]).clip(lower=0)
    full_agg = aggregate_table(ins.drop(columns=["SK_ID_PREV"]), "SK_ID_CURR", "INSTAL")
    recent_agg = aggregate_table(ins[ins["DAYS_INSTALMENT"] >= -365], "SK_ID_CURR", "INSTAL_RECENT")
    return full_agg.merge(recent_agg, on="SK_ID_CURR", how="left") if not recent_agg.empty else full_agg

def credit_card_features() -> pd.DataFrame:
    cc = load_csv("credit_card_balance.csv")
    if cc.empty: return pd.DataFrame()
    full_agg = aggregate_table(cc.drop(columns=["SK_ID_PREV"]), "SK_ID_CURR", "CC")
    recent_agg = aggregate_table(cc[cc["MONTHS_BALANCE"] >= -12], "SK_ID_CURR", "CC_RECENT")
    return full_agg.merge(recent_agg, on="SK_ID_CURR", how="left") if not recent_agg.empty else full_agg

def build_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = load_csv("application_train.csv"), load_csv("application_test.csv")
    app = pd.concat([train, test], axis=0, ignore_index=True, sort=False)
    app = add_application_features(app)
    
    for table in [bureau_features(), previous_application_features(), pos_cash_features(), installments_features(), credit_card_features()]:
        if not table.empty: app = app.merge(table, on="SK_ID_CURR", how="left")
    
    print("Final encoding and cleaning...")
    app = one_hot_encode(app)
    app = reduce_memory(app)
    
    train_df, test_df = app[app["TARGET"].notnull()].copy(), app[app["TARGET"].isnull()].copy()
    print("Computing Neighbors Target Mean (1st Place strategy)...")
    train_df["NEIGHBORS_TARGET_MEAN"], test_df["NEIGHBORS_TARGET_MEAN"] = neighbors_target_mean(train_df, test_df)
    return train_df, test_df

def train_and_predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, float]:
    excluded = {"TARGET", "SK_ID_CURR"}
    features = [col for col in train_df.columns if col not in excluded]
    
    # Cleaning feature names and data types
    for col in features:
        if not pd.api.types.is_numeric_dtype(train_df[col]):
            combined = pd.concat([train_df[col], test_df[col]], axis=0)
            factors, _ = pd.factorize(combined)
            train_df[col], test_df[col] = factors[:len(train_df)], factors[len(train_df):]

    new_names = [clean_column_name(f) for f in features]
    train_df.rename(columns=dict(zip(features, new_names)), inplace=True)
    test_df.rename(columns=dict(zip(features, new_names)), inplace=True)
    features = new_names

    X, y, X_test = train_df[features], train_df["TARGET"].astype(int), test_df[features]
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

    oof_ensemble = np.zeros(len(train_df))
    preds_ensemble = np.zeros(len(test_df))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]

        # 1. LightGBM (Standard)
        lgb_params = {"n_estimators": 10000, "learning_rate": 0.02, "num_leaves": 34, "colsample_bytree": 0.9, "subsample": 0.8, "verbosity": -1}
        try:
            lgb_m = lgb.LGBMClassifier(**lgb_params, device="cuda", gpu_use_dp=False, random_state=RANDOM_STATE)
            lgb_m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="auc", callbacks=[lgb.early_stopping(200)])
        except:
            lgb_m = lgb.LGBMClassifier(**lgb_params, device="cpu", random_state=RANDOM_STATE)
            lgb_m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="auc", callbacks=[lgb.early_stopping(200)])
        
        # 2. XGBoost (GPU)
        xgb_m = xgb.XGBClassifier(
            n_estimators=10000, learning_rate=0.02, max_depth=6, tree_method="hist", 
            device="cuda", random_state=RANDOM_STATE, early_stopping_rounds=200
        )
        
        xgb_m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        # 3. CatBoost (GPU - 2nd Place Strategy)
        cat_m = CatBoostClassifier(iterations=10000, learning_rate=0.02, depth=6, eval_metric="AUC", task_type="GPU", devices="0", random_seed=RANDOM_STATE, verbose=200)
        cat_m.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=200)

        # Weighted Ensemble (LGB: 0.4, XGB: 0.3, CAT: 0.3)
        l_p, x_p, c_p = lgb_m.predict_proba(X_val)[:,1], xgb_m.predict_proba(X_val)[:,1], cat_m.predict_proba(X_val)[:,1]
        oof_ensemble[valid_idx] = (l_p * 0.4 + x_p * 0.3 + c_p * 0.3)
        
        l_t, x_t, c_t = lgb_m.predict_proba(X_test)[:,1], xgb_m.predict_proba(X_test)[:,1], cat_m.predict_proba(X_test)[:,1]
        preds_ensemble += (l_t * 0.4 + x_t * 0.3 + c_t * 0.3) / N_SPLITS
        print(f"Fold {fold} Ensemble AUC: {roc_auc_score(y_val, oof_ensemble[valid_idx]):.6f}")

    print(f"Full CV Ensemble AUC: {roc_auc_score(y, oof_ensemble):.6f}")
    return preds_ensemble, roc_auc_score(y, oof_ensemble)

def main() -> None:
    train_df, test_df = build_dataset()
    preds, auc = train_and_predict(train_df, test_df)
    submission = pd.DataFrame({"SK_ID_CURR": test_df["SK_ID_CURR"].astype(int), "TARGET": preds})
    submission.to_csv("submission2.csv", index=False)
    print(f"Saved submission2.csv (CV: {auc:.6f})")

if __name__ == "__main__":
    main()
