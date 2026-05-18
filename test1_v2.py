from __future__ import annotations

import re
import warnings
from pathlib import Path

import lightgbm as lgb
import xgboost as xgb
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
    cat_cols = [
        col
        for col in df.columns
        if not pd.api.types.is_numeric_dtype(df[col]) and col not in ["TARGET", "SK_ID_CURR"]
    ]
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


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace({0: np.nan})


def aggregate_table(
    df: pd.DataFrame,
    group_key: str,
    prefix: str,
    numeric_aggs: list[str] | None = None,
    categorical_aggs: list[str] | None = None,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    numeric_aggs = numeric_aggs or ["mean", "max", "min", "sum", "var"]
    categorical_aggs = categorical_aggs or ["mean", "sum"]

    num_cols = [
        col
        for col in df.columns
        if col != group_key
        and pd.api.types.is_numeric_dtype(df[col])
        and not set(df[col].dropna().unique()).issubset({0, 1})
    ]
    cat_cols = [
        col
        for col in df.columns
        if col != group_key and (df[col].dtype == "object" or set(df[col].dropna().unique()).issubset({0, 1}))
    ]

    if any(df[col].dtype == "object" for col in cat_cols):
        df = one_hot_encode(df)
        cat_cols = [c for c in df.columns if any(c.startswith(oc) for oc in cat_cols) and c != group_key]

    agg_dict: dict[str, list[str]] = {}
    for col in num_cols:
        agg_dict[col] = numeric_aggs
    for col in cat_cols:
        agg_dict[col] = categorical_aggs

    grouped = df.groupby(group_key).agg(agg_dict)
    grouped.columns = [f"{prefix}_{clean_column_name(col)}_{stat.upper()}" for col, stat in grouped.columns]
    grouped.reset_index(inplace=True)
    return reduce_memory(grouped)


def aggregate_last_k(df: pd.DataFrame, group_key: str, sort_col: str, prefix: str, windows: tuple[int, ...]) -> list[pd.DataFrame]:
    if df.empty or sort_col not in df.columns:
        return []
    ordered = df.sort_values([group_key, sort_col], ascending=[True, False], kind="mergesort")
    outputs: list[pd.DataFrame] = []
    for k in windows:
        subset = ordered.groupby(group_key, group_keys=False).head(k)
        outputs.append(aggregate_table(subset, group_key, f"{prefix}_LAST_{k}"))
    return outputs


def aggregate_first_k(df: pd.DataFrame, group_key: str, sort_col: str, prefix: str, windows: tuple[int, ...]) -> list[pd.DataFrame]:
    if df.empty or sort_col not in df.columns:
        return []
    ordered = df.sort_values([group_key, sort_col], ascending=[True, True], kind="mergesort")
    outputs: list[pd.DataFrame] = []
    for k in windows:
        subset = ordered.groupby(group_key, group_keys=False).head(k)
        outputs.append(aggregate_table(subset, group_key, f"{prefix}_FIRST_{k}"))
    return outputs


def merge_feature_tables(base: pd.DataFrame, tables: list[pd.DataFrame]) -> pd.DataFrame:
    merged = base
    for table in tables:
        if not table.empty:
            merged = merged.merge(table, on="SK_ID_CURR", how="left")
    return reduce_memory(merged)


def add_application_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["DAYS_EMPLOYED"].replace(365243, np.nan, inplace=True)
    df["DAYS_LAST_PHONE_CHANGE"].replace(0, np.nan, inplace=True)

    df["APP_CREDIT_INCOME_RATIO"] = safe_divide(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
    df["APP_ANNUITY_INCOME_RATIO"] = safe_divide(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
    df["APP_CREDIT_ANNUITY_RATIO"] = safe_divide(df["AMT_CREDIT"], df["AMT_ANNUITY"])
    df["APP_GOODS_CREDIT_RATIO"] = safe_divide(df["AMT_GOODS_PRICE"], df["AMT_CREDIT"])
    df["APP_CREDIT_GOODS_RATIO"] = safe_divide(df["AMT_CREDIT"], df["AMT_GOODS_PRICE"])
    df["APP_INCOME_PER_PERSON"] = safe_divide(df["AMT_INCOME_TOTAL"], df["CNT_FAM_MEMBERS"])
    df["APP_EMPLOYED_BIRTH_RATIO"] = safe_divide(df["DAYS_EMPLOYED"], df["DAYS_BIRTH"])
    df["APP_ANNUITY_LENGTH"] = safe_divide(df["AMT_ANNUITY"], df["AMT_CREDIT"])
    df["APP_CREDIT_DOWNPAYMENT"] = df["AMT_GOODS_PRICE"] - df["AMT_CREDIT"]
    df["APP_AGE_INT"] = (-df["DAYS_BIRTH"] / 365).fillna(0).astype(np.int16)

    df["APP_EXT_SOURCE_1_BY_3"] = safe_divide(df["EXT_SOURCE_1"], df["EXT_SOURCE_3"])
    df["APP_EXT_SOURCE_2_BY_3"] = safe_divide(df["EXT_SOURCE_2"], df["EXT_SOURCE_3"])
    df["APP_EXT_SOURCES_PROD"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
    df["APP_EXT_SOURCES_MEAN"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    df["APP_EXT_SOURCES_STD"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].std(axis=1)
    df["APP_EXT_SOURCES_NAN_COUNT"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].isnull().sum(axis=1)

    if "REGION_ID_POPULATION" in df.columns:
        df["APP_REGION_ID"] = pd.factorize(df["REGION_ID_POPULATION"])[0].astype(np.int32)

    return df


def neighbors_target_mean(train_df: pd.DataFrame, test_df: pd.DataFrame, n_neighbors: int = 500) -> tuple[pd.Series, pd.Series]:
    cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "APP_CREDIT_ANNUITY_RATIO"]
    data = pd.concat([train_df[cols + ["TARGET"]], test_df[cols]], axis=0)
    data_imputed = data[cols].fillna(data[cols].median())
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_imputed)

    train_scaled = data_scaled[: len(train_df)]
    test_scaled = data_scaled[len(train_df) :]
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=-1)

    oof_neighbors = np.zeros(len(train_df))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for tr_idx, val_idx in skf.split(train_df, train_df["TARGET"]):
        knn.fit(train_scaled[tr_idx], train_df["TARGET"].iloc[tr_idx])
        oof_neighbors[val_idx] = knn.predict(train_scaled[val_idx])

    knn.fit(train_scaled, train_df["TARGET"])
    test_neighbors = knn.predict(test_scaled)
    return pd.Series(oof_neighbors, index=train_df.index), pd.Series(test_neighbors, index=test_df.index)


def bureau_features() -> pd.DataFrame:
    bureau = load_csv("bureau.csv")
    bb = load_csv("bureau_balance.csv")
    if bureau.empty:
        return pd.DataFrame()

    if not bb.empty:
        bb = one_hot_encode(bb)
        bb_agg = bb.groupby("SK_ID_BUREAU").agg(
            {
                "MONTHS_BALANCE": ["min", "max", "size"],
                **{col: ["mean"] for col in bb.columns if col.startswith("STATUS_")},
            }
        )
        bb_agg.columns = [f"BB_{clean_column_name(col)}_{stat.upper()}" for col, stat in bb_agg.columns]
        bureau = bureau.join(bb_agg, how="left", on="SK_ID_BUREAU")

    bureau["DAYS_CREDIT_ENDDATE"].replace(365243, np.nan, inplace=True)
    bureau["AMT_CREDIT_SUM_DEBT_OVER_SUM"] = safe_divide(bureau["AMT_CREDIT_SUM_DEBT"], bureau["AMT_CREDIT_SUM"])

    base = bureau.drop(columns=["SK_ID_BUREAU"] if "SK_ID_BUREAU" in bureau.columns else [])
    full_agg = aggregate_table(base, "SK_ID_CURR", "BURO")
    recent_12m = aggregate_table(bureau[bureau["DAYS_CREDIT"] >= -365], "SK_ID_CURR", "BURO_RECENT_12M")

    active = bureau[bureau["CREDIT_ACTIVE"] == "Active"].copy()
    if not active.empty:
        active_last = (
            active.sort_values(["SK_ID_CURR", "DAYS_CREDIT"], ascending=[True, False], kind="mergesort")
            .groupby("SK_ID_CURR")
            .first()
            .reset_index()
        )
        active_last = active_last[["SK_ID_CURR", "DAYS_CREDIT"]].rename(columns={"DAYS_CREDIT": "LAST_ACTIVE_DAYS_CREDIT"})
        full_agg = full_agg.merge(active_last, on="SK_ID_CURR", how="left")

    if not recent_12m.empty:
        full_agg = full_agg.merge(recent_12m, on="SK_ID_CURR", how="left")
    return reduce_memory(full_agg)


def previous_application_features() -> pd.DataFrame:
    prev = load_csv("previous_application.csv")
    if prev.empty:
        return pd.DataFrame()

    for col in [c for c in prev.columns if c.startswith("DAYS_")]:
        prev[col].replace(365243, np.nan, inplace=True)

    prev["PREV_APP_CREDIT_PERC"] = safe_divide(prev["AMT_APPLICATION"], prev["AMT_CREDIT"])
    prev["PREV_CREDIT_GOODS_RATIO"] = safe_divide(prev["AMT_CREDIT"], prev["AMT_GOODS_PRICE"])

    base = prev.drop(columns=["SK_ID_PREV"])
    full_agg = aggregate_table(base, "SK_ID_CURR", "PREV")

    last_product = (
        prev.sort_values(["SK_ID_CURR", "DAYS_DECISION"], ascending=[True, False], kind="mergesort")
        .groupby("SK_ID_CURR")
        .first()
        .reset_index()
    )
    keep_cols = ["SK_ID_CURR"]
    if "PRODUCT_COMBINATION" in last_product.columns:
        keep_cols.append("PRODUCT_COMBINATION")
    if len(keep_cols) > 1:
        last_product = pd.get_dummies(last_product[keep_cols], columns=["PRODUCT_COMBINATION"], dummy_na=True)
        last_product.columns = [
            "SK_ID_CURR" if col == "SK_ID_CURR" else f"PREV_LAST_{clean_column_name(col)}"
            for col in last_product.columns
        ]
        full_agg = full_agg.merge(last_product, on="SK_ID_CURR", how="left")

    feature_tables = [
        aggregate_table(prev[prev["DAYS_DECISION"] >= -730], "SK_ID_CURR", "PREV_RECENT_24M"),
        *aggregate_last_k(base, "SK_ID_CURR", "DAYS_DECISION", "PREV", (3, 5)),
        *aggregate_first_k(base, "SK_ID_CURR", "DAYS_DECISION", "PREV", (2, 4)),
    ]
    return merge_feature_tables(full_agg, feature_tables)


def pos_cash_features() -> pd.DataFrame:
    pos = load_csv("POS_CASH_balance.csv")
    if pos.empty:
        return pd.DataFrame()

    base = pos.drop(columns=["SK_ID_PREV"])
    full_agg = aggregate_table(base, "SK_ID_CURR", "POS")
    feature_tables = [
        aggregate_table(pos[pos["MONTHS_BALANCE"] >= -12].drop(columns=["SK_ID_PREV"]), "SK_ID_CURR", "POS_RECENT_12M"),
        *aggregate_last_k(base, "SK_ID_CURR", "MONTHS_BALANCE", "POS", (3, 5, 10)),
    ]
    return merge_feature_tables(full_agg, feature_tables)


def installments_features() -> pd.DataFrame:
    ins = load_csv("installments_payments.csv")
    if ins.empty:
        return pd.DataFrame()

    ins["PAYMENT_PERC"] = safe_divide(ins["AMT_PAYMENT"], ins["AMT_INSTALMENT"])
    ins["PAYMENT_DIFF"] = ins["AMT_PAYMENT"] - ins["AMT_INSTALMENT"]
    ins["DPD"] = (ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]).clip(lower=0)
    ins["IS_PAST_DUE"] = (ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"] > 0).astype(np.int8)

    full_agg = aggregate_table(ins.drop(columns=["SK_ID_PREV"]), "SK_ID_CURR", "INSTAL")
    feature_tables = [
        aggregate_table(ins[ins["DAYS_INSTALMENT"] >= -60].drop(columns=["SK_ID_PREV"]), "SK_ID_CURR", "INSTAL_LAST_60D"),
        aggregate_table(ins[ins["DAYS_INSTALMENT"] >= -90].drop(columns=["SK_ID_PREV"]), "SK_ID_CURR", "INSTAL_LAST_90D"),
        aggregate_table(ins[ins["DAYS_INSTALMENT"] >= -180].drop(columns=["SK_ID_PREV"]), "SK_ID_CURR", "INSTAL_LAST_180D"),
        aggregate_table(ins[ins["DAYS_INSTALMENT"] >= -365].drop(columns=["SK_ID_PREV"]), "SK_ID_CURR", "INSTAL_LAST_365D"),
        aggregate_table(ins[ins["IS_PAST_DUE"] == 1].drop(columns=["SK_ID_PREV"]), "SK_ID_CURR", "INSTAL_PAST_DUE"),
        *aggregate_last_k(ins.drop(columns=["SK_ID_PREV"]), "SK_ID_CURR", "DAYS_INSTALMENT", "INSTAL", (2, 3, 5)),
    ]
    for inst_num in (1, 2, 3, 4):
        feature_tables.append(
            aggregate_table(
                ins[ins["NUM_INSTALMENT_NUMBER"] == inst_num].drop(columns=["SK_ID_PREV"]),
                "SK_ID_CURR",
                f"INSTAL_NUM_{inst_num}",
            )
        )
    return merge_feature_tables(full_agg, feature_tables)


def credit_card_features() -> pd.DataFrame:
    cc = load_csv("credit_card_balance.csv")
    if cc.empty:
        return pd.DataFrame()

    base = cc.drop(columns=["SK_ID_PREV"])
    full_agg = aggregate_table(base, "SK_ID_CURR", "CC")
    feature_tables = [
        aggregate_table(cc[cc["MONTHS_BALANCE"] >= -12].drop(columns=["SK_ID_PREV"]), "SK_ID_CURR", "CC_RECENT_12M"),
        *aggregate_last_k(base, "SK_ID_CURR", "MONTHS_BALANCE", "CC", (3, 5, 10)),
    ]
    return merge_feature_tables(full_agg, feature_tables)


def build_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = load_csv("application_train.csv")
    test = load_csv("application_test.csv")

    app = pd.concat([train, test], axis=0, ignore_index=True, sort=False)
    app = add_application_features(app)
    app = reduce_memory(app)

    feature_tables = [
        bureau_features(),
        previous_application_features(),
        pos_cash_features(),
        installments_features(),
        credit_card_features(),
    ]

    for table in feature_tables:
        if table.empty:
            continue
        app = app.merge(table, on="SK_ID_CURR", how="left")
        app = reduce_memory(app)

    print("Final encoding...")
    app = one_hot_encode(app)
    app = reduce_memory(app)

    train_df = app[app["TARGET"].notnull()].copy()
    test_df = app[app["TARGET"].isnull()].copy()

    print("Computing Neighbors Target Mean...")
    train_df["NEIGHBORS_TARGET_MEAN"], test_df["NEIGHBORS_TARGET_MEAN"] = neighbors_target_mean(train_df, test_df)
    return train_df, test_df


def train_and_predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, float]:
    excluded = {"TARGET", "SK_ID_CURR"}
    features = [col for col in train_df.columns if col not in excluded]

    for col in features:
        if not pd.api.types.is_numeric_dtype(train_df[col]):
            combined = pd.concat([train_df[col], test_df[col]], axis=0)
            factors, _ = pd.factorize(combined)
            train_df[col] = factors[: len(train_df)]
            test_df[col] = factors[len(train_df) :]

    new_feature_names = [clean_column_name(f) for f in features]
    train_df.rename(columns=dict(zip(features, new_feature_names)), inplace=True)
    test_df.rename(columns=dict(zip(features, new_feature_names)), inplace=True)
    features = new_feature_names

    X = train_df[features]
    y = train_df["TARGET"].astype(int)
    X_test = test_df[features]

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

    oof_ensemble = np.zeros(len(train_df))
    preds_ensemble = np.zeros(len(test_df))

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        lgb_params = {
            "n_estimators": 10000,
            "learning_rate": 0.02,
            "num_leaves": 34,
            "colsample_bytree": 0.9,
            "subsample": 0.8,
            "max_depth": 8,
            "reg_alpha": 0.1,
            "reg_lambda": 10.0,
            "min_split_gain": 0.01,
            "min_child_weight": 30,
            "verbosity": -1,
            "random_state": RANDOM_STATE,
        }

        try:
            lgb_model = lgb.LGBMClassifier(**lgb_params, device="cuda", gpu_use_dp=False)
            lgb_model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(200), lgb.log_evaluation(200)],
            )
        except lgb.basic.LightGBMError as exc:
            if "CUDA" in str(exc) or "GPU" in str(exc):
                print("LightGBM GPU/CUDA not enabled in this build. Falling back to CPU...")
                lgb_model = lgb.LGBMClassifier(**lgb_params, device="cpu")
                lgb_model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_valid, y_valid)],
                    eval_metric="auc",
                    callbacks=[lgb.early_stopping(200), lgb.log_evaluation(200)],
                )
            else:
                raise exc

        try:
            xgb_model = xgb.XGBClassifier(
                n_estimators=10000,
                learning_rate=0.02,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                random_state=RANDOM_STATE,
                tree_method="hist",
                device="cuda",
                early_stopping_rounds=200,
            )
            xgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        except xgb.core.XGBoostError as exc:
            if "CUDA" in str(exc) or "GPU" in str(exc) or "device" in str(exc).lower():
                print("XGBoost GPU/CUDA not enabled in this build. Falling back to CPU...")
                xgb_model = xgb.XGBClassifier(
                    n_estimators=10000,
                    learning_rate=0.02,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    random_state=RANDOM_STATE,
                    tree_method="hist",
                    device="cpu",
                    early_stopping_rounds=200,
                )
                xgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
            else:
                raise exc

        lgb_oof = lgb_model.predict_proba(X_valid)[:, 1]
        xgb_oof = xgb_model.predict_proba(X_valid)[:, 1]
        lgb_preds = lgb_model.predict_proba(X_test)[:, 1]
        xgb_preds = xgb_model.predict_proba(X_test)[:, 1]

        oof_ensemble[valid_idx] = lgb_oof * 0.7 + xgb_oof * 0.3
        preds_ensemble += (lgb_preds * 0.7 + xgb_preds * 0.3) / N_SPLITS

        fold_auc = roc_auc_score(y_valid, oof_ensemble[valid_idx])
        print(f"Fold {fold} Ensemble AUC: {fold_auc:.6f}")

    full_auc = roc_auc_score(y, oof_ensemble)
    print(f"Full CV Ensemble AUC: {full_auc:.6f}")
    return preds_ensemble, full_auc


def main() -> None:
    train_df, test_df = build_dataset()
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    preds, auc = train_and_predict(train_df, test_df)

    submission = pd.DataFrame(
        {
            "SK_ID_CURR": test_df["SK_ID_CURR"].astype(int),
            "TARGET": preds,
        }
    )
    output_path = Path(__file__).resolve().parent / "submission1_v2.csv"
    submission.to_csv(output_path, index=False)
    print(f"Saved submission to {output_path}")
    print(f"Final CV: {auc:.6f}")


if __name__ == "__main__":
    main()
