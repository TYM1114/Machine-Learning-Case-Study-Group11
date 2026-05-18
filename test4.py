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
USE_CATBOOST = True
LAST_K_WINDOWS = (1, 3, 5, 10)
PREV_FIRST_K_WINDOWS = (2, 4)
MAX_FEATURES = 900


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


def merge_feature_tables(base: pd.DataFrame, tables: list[pd.DataFrame]) -> pd.DataFrame:
    merged = base
    for table in tables:
        if not table.empty:
            merged = merged.merge(table, on="SK_ID_CURR", how="left")
    return reduce_memory(merged)


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


def aggregate_last_k(
    df: pd.DataFrame,
    group_key: str,
    sort_col: str,
    prefix: str,
    windows: tuple[int, ...],
) -> list[pd.DataFrame]:
    if df.empty or sort_col not in df.columns:
        return []

    ordered = df.sort_values([group_key, sort_col], ascending=[True, False], kind="mergesort")
    tables: list[pd.DataFrame] = []
    for k in windows:
        last_k = ordered.groupby(group_key, group_keys=False).head(k)
        tables.append(aggregate_table(last_k, group_key, f"{prefix}_LAST_{k}"))
    return tables


def aggregate_first_k(
    df: pd.DataFrame,
    group_key: str,
    sort_col: str,
    prefix: str,
    windows: tuple[int, ...],
) -> list[pd.DataFrame]:
    if df.empty or sort_col not in df.columns:
        return []

    ordered = df.sort_values([group_key, sort_col], ascending=[True, True], kind="mergesort")
    tables: list[pd.DataFrame] = []
    for k in windows:
        first_k = ordered.groupby(group_key, group_keys=False).head(k)
        tables.append(aggregate_table(first_k, group_key, f"{prefix}_FIRST_{k}"))
    return tables


def add_application_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["DAYS_EMPLOYED"].replace(365243, np.nan, inplace=True)
    df["DAYS_LAST_PHONE_CHANGE"].replace(0, np.nan, inplace=True)

    df["APP_CREDIT_INCOME_RATIO"] = safe_divide(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
    df["APP_ANNUITY_INCOME_RATIO"] = safe_divide(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
    df["APP_CREDIT_ANNUITY_RATIO"] = safe_divide(df["AMT_CREDIT"], df["AMT_ANNUITY"])
    df["APP_GOODS_CREDIT_RATIO"] = safe_divide(df["AMT_GOODS_PRICE"], df["AMT_CREDIT"])
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
    return df


def neighbors_target_mean(
    train_df: pd.DataFrame, test_df: pd.DataFrame, n_neighbors: int = 500
) -> tuple[pd.Series, pd.Series]:
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
    bureau["AMT_CREDIT_SUM_DEBT_OVER_SUM"] = safe_divide(
        bureau["AMT_CREDIT_SUM_DEBT"], bureau["AMT_CREDIT_SUM"]
    )
    bureau["BURO_CREDIT_ACTIVE_BINARY"] = (bureau["CREDIT_ACTIVE"] == "Active").astype(np.int8)

    full_agg = aggregate_table(
        bureau.drop(columns=["SK_ID_BUREAU"] if "SK_ID_BUREAU" in bureau.columns else []),
        "SK_ID_CURR",
        "BURO",
    )
    tables = [
        aggregate_table(bureau[bureau["DAYS_CREDIT"] >= -365], "SK_ID_CURR", "BURO_RECENT_12M"),
        aggregate_table(bureau[bureau["DAYS_CREDIT"] >= -730], "SK_ID_CURR", "BURO_RECENT_24M"),
        *aggregate_last_k(
            bureau.drop(columns=["SK_ID_BUREAU"] if "SK_ID_BUREAU" in bureau.columns else []),
            "SK_ID_CURR",
            "DAYS_CREDIT",
            "BURO",
            LAST_K_WINDOWS,
        ),
    ]
    full_agg = merge_feature_tables(full_agg, tables)

    ratio_pairs = [
        ("BURO_RECENT_12M_AMT_CREDIT_SUM_DEBT_MEAN", "BURO_AMT_CREDIT_SUM_DEBT_MEAN", "BURO_TREND_DEBT_RATIO_12M"),
        ("BURO_RECENT_24M_AMT_CREDIT_SUM_DEBT_MEAN", "BURO_AMT_CREDIT_SUM_DEBT_MEAN", "BURO_TREND_DEBT_RATIO_24M"),
        ("BURO_LAST_1_AMT_CREDIT_SUM_DEBT_MEAN", "BURO_AMT_CREDIT_SUM_DEBT_MEAN", "BURO_LAST1_DEBT_RATIO"),
        ("BURO_LAST_3_AMT_CREDIT_SUM_DEBT_MEAN", "BURO_AMT_CREDIT_SUM_DEBT_MEAN", "BURO_LAST3_DEBT_RATIO"),
    ]
    for num_col, den_col, out_col in ratio_pairs:
        if num_col in full_agg.columns and den_col in full_agg.columns:
            full_agg[out_col] = safe_divide(full_agg[num_col], full_agg[den_col])

    return reduce_memory(full_agg)


def previous_application_features() -> pd.DataFrame:
    prev = load_csv("previous_application.csv")
    if prev.empty:
        return pd.DataFrame()

    for col in [c for c in prev.columns if c.startswith("DAYS_")]:
        prev[col].replace(365243, np.nan, inplace=True)

    prev["PREV_APP_CREDIT_PERC"] = safe_divide(prev["AMT_APPLICATION"], prev["AMT_CREDIT"])
    prev["PREV_INTEREST_EST"] = safe_divide(prev["AMT_ANNUITY"] * prev["CNT_PAYMENT"], prev["AMT_CREDIT"]) - 1
    prev["PREV_CREDIT_GOODS_RATIO"] = safe_divide(prev["AMT_CREDIT"], prev["AMT_GOODS_PRICE"])

    full_agg = aggregate_table(prev.drop(columns=["SK_ID_PREV"]), "SK_ID_CURR", "PREV")
    tables = [
        aggregate_table(prev[prev["DAYS_DECISION"] >= -365], "SK_ID_CURR", "PREV_RECENT_12M"),
        aggregate_table(prev[prev["DAYS_DECISION"] >= -730], "SK_ID_CURR", "PREV_RECENT_24M"),
        *aggregate_last_k(prev.drop(columns=["SK_ID_PREV"]), "SK_ID_CURR", "DAYS_DECISION", "PREV", (3, 5)),
        *aggregate_first_k(prev.drop(columns=["SK_ID_PREV"]), "SK_ID_CURR", "DAYS_DECISION", "PREV", PREV_FIRST_K_WINDOWS),
    ]
    full_agg = merge_feature_tables(full_agg, tables)
    return full_agg


def pos_cash_features() -> pd.DataFrame:
    pos = load_csv("POS_CASH_balance.csv")
    if pos.empty:
        return pd.DataFrame()

    full_agg = aggregate_table(pos.drop(columns=["SK_ID_PREV"]), "SK_ID_CURR", "POS")
    tables = [
        aggregate_table(pos[pos["MONTHS_BALANCE"] >= -12].drop(columns=["SK_ID_PREV"]), "SK_ID_CURR", "POS_RECENT_12M"),
        *aggregate_last_k(pos.drop(columns=["SK_ID_PREV"]), "SK_ID_CURR", "MONTHS_BALANCE", "POS", LAST_K_WINDOWS),
    ]
    return merge_feature_tables(full_agg, tables)


def installments_features() -> pd.DataFrame:
    ins = load_csv("installments_payments.csv")
    if ins.empty:
        return pd.DataFrame()

    ins["PAYMENT_PERC"] = safe_divide(ins["AMT_PAYMENT"], ins["AMT_INSTALMENT"])
    ins["PAYMENT_DIFF"] = ins["AMT_PAYMENT"] - ins["AMT_INSTALMENT"]
    ins["DPD"] = (ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]).clip(lower=0)
    ins["IS_PAST_DUE"] = (ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"] > 0).astype(np.int8)

    full_agg = aggregate_table(ins.drop(columns=["SK_ID_PREV"]), "SK_ID_CURR", "INSTAL")
    tables = [
        aggregate_table(ins[ins["DAYS_INSTALMENT"] >= -365].drop(columns=["SK_ID_PREV"]), "SK_ID_CURR", "INSTAL_RECENT_12M"),
        aggregate_table(ins[ins["IS_PAST_DUE"] == 1].drop(columns=["SK_ID_PREV"]), "SK_ID_CURR", "INSTAL_PAST_DUE"),
        *aggregate_last_k(ins.drop(columns=["SK_ID_PREV"]), "SK_ID_CURR", "DAYS_INSTALMENT", "INSTAL", LAST_K_WINDOWS),
    ]
    for inst_num in (1, 2, 3, 4):
        tables.append(
            aggregate_table(
                ins[ins["NUM_INSTALMENT_NUMBER"] == inst_num].drop(columns=["SK_ID_PREV"]),
                "SK_ID_CURR",
                f"INSTAL_NUM_{inst_num}",
            )
        )
    return merge_feature_tables(full_agg, tables)


def credit_card_features() -> pd.DataFrame:
    cc = load_csv("credit_card_balance.csv")
    if cc.empty:
        return pd.DataFrame()

    full_agg = aggregate_table(cc.drop(columns=["SK_ID_PREV"]), "SK_ID_CURR", "CC")
    tables = [
        aggregate_table(cc[cc["MONTHS_BALANCE"] >= -12].drop(columns=["SK_ID_PREV"]), "SK_ID_CURR", "CC_RECENT_12M"),
        *aggregate_last_k(cc.drop(columns=["SK_ID_PREV"]), "SK_ID_CURR", "MONTHS_BALANCE", "CC", LAST_K_WINDOWS),
    ]
    return merge_feature_tables(full_agg, tables)


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


def select_top_features(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    max_features: int = MAX_FEATURES,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if X.shape[1] <= max_features:
        print(f"Feature selection skipped: only {X.shape[1]} features.")
        return X, X_test, list(X.columns)

    selector_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    train_idx, valid_idx = next(selector_skf.split(X, y))

    selector = lgb.LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=48,
        colsample_bytree=0.8,
        subsample=0.8,
        reg_alpha=0.1,
        reg_lambda=5.0,
        min_child_weight=20,
        importance_type="gain",
        random_state=RANDOM_STATE,
        verbosity=-1,
    )
    selector.fit(
        X.iloc[train_idx],
        y.iloc[train_idx],
        eval_set=[(X.iloc[valid_idx], y.iloc[valid_idx])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )

    importance = pd.Series(selector.feature_importances_, index=X.columns).sort_values(ascending=False)
    selected = importance[importance > 0].head(max_features).index.tolist()
    if not selected:
        selected = importance.head(max_features).index.tolist()

    must_keep = [col for col in X.columns if col == "NEIGHBORS_TARGET_MEAN" or col.startswith("APP_EXT_SOURCE")]
    selected = list(dict.fromkeys(selected + must_keep))
    print(f"Selected {len(selected)} features out of {X.shape[1]}.")
    return X[selected], X_test[selected], selected


def find_best_weights(model_oof: dict[str, np.ndarray], y: pd.Series) -> dict[str, float]:
    names = list(model_oof.keys())
    if len(names) == 2:
        best_auc = -1.0
        best_weights = None
        for w0 in np.linspace(0.0, 1.0, 21):
            weights = np.array([w0, 1.0 - w0])
            preds = sum(model_oof[name] * weight for name, weight in zip(names, weights))
            auc = roc_auc_score(y, preds)
            if auc > best_auc:
                best_auc = auc
                best_weights = weights
        return dict(zip(names, best_weights))

    best_auc = -1.0
    best_weights = None
    for w0 in np.linspace(0.0, 1.0, 11):
        for w1 in np.linspace(0.0, 1.0 - w0, int((1.0 - w0) / 0.1) + 1):
            w2 = 1.0 - w0 - w1
            weights = np.array([w0, w1, w2])
            preds = sum(model_oof[name] * weight for name, weight in zip(names, weights))
            auc = roc_auc_score(y, preds)
            if auc > best_auc:
                best_auc = auc
                best_weights = weights
    return dict(zip(names, best_weights))


def train_lgb_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> lgb.LGBMClassifier:
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
        model = lgb.LGBMClassifier(**lgb_params, device="cuda", gpu_use_dp=False)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)],
        )
    except lgb.basic.LightGBMError as exc:
        if "CUDA" in str(exc) or "GPU" in str(exc):
            model = lgb.LGBMClassifier(**lgb_params, device="cpu")
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)],
            )
        else:
            raise exc
    return model


def train_xgb_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> xgb.XGBClassifier:
    xgb_params = {
        "n_estimators": 10000,
        "learning_rate": 0.02,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "random_state": RANDOM_STATE,
        "tree_method": "hist",
        "early_stopping_rounds": 200,
    }
    try:
        model = xgb.XGBClassifier(**xgb_params, device="cuda")
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    except xgb.core.XGBoostError as exc:
        if "CUDA" in str(exc) or "device" in str(exc).lower() or "gpu" in str(exc).lower():
            model = xgb.XGBClassifier(**xgb_params, device="cpu")
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        else:
            raise exc
    return model


def train_cat_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> CatBoostClassifier:
    cat_params = {
        "iterations": 5000,
        "learning_rate": 0.03,
        "depth": 6,
        "eval_metric": "AUC",
        "random_seed": RANDOM_STATE,
        "verbose": False,
    }
    try:
        model = CatBoostClassifier(**cat_params, task_type="GPU", devices="0")
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=200)
    except Exception:
        model = CatBoostClassifier(**cat_params, task_type="CPU")
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=200)
    return model


def train_and_predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, float, dict[str, float], list[str]]:
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

    X, X_test, selected_features = select_top_features(X, y, X_test)

    model_oof: dict[str, np.ndarray] = {
        "lgb": np.zeros(len(train_df)),
        "xgb": np.zeros(len(train_df)),
    }
    model_test_preds: dict[str, np.ndarray] = {
        "lgb": np.zeros(len(test_df)),
        "xgb": np.zeros(len(test_df)),
    }
    if USE_CATBOOST:
        model_oof["cat"] = np.zeros(len(train_df))
        model_test_preds["cat"] = np.zeros(len(test_df))

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        lgb_model = train_lgb_model(X_train, y_train, X_valid, y_valid)
        xgb_model = train_xgb_model(X_train, y_train, X_valid, y_valid)

        model_oof["lgb"][valid_idx] = lgb_model.predict_proba(X_valid)[:, 1]
        model_oof["xgb"][valid_idx] = xgb_model.predict_proba(X_valid)[:, 1]
        model_test_preds["lgb"] += lgb_model.predict_proba(X_test)[:, 1] / N_SPLITS
        model_test_preds["xgb"] += xgb_model.predict_proba(X_test)[:, 1] / N_SPLITS

        if USE_CATBOOST:
            cat_model = train_cat_model(X_train, y_train, X_valid, y_valid)
            model_oof["cat"][valid_idx] = cat_model.predict_proba(X_valid)[:, 1]
            model_test_preds["cat"] += cat_model.predict_proba(X_test)[:, 1] / N_SPLITS

        fold_weights = find_best_weights({name: preds[valid_idx] for name, preds in model_oof.items()}, y_valid)
        fold_blend = sum(model_oof[name][valid_idx] * weight for name, weight in fold_weights.items())
        fold_auc = roc_auc_score(y_valid, fold_blend)
        print(f"Fold {fold} best weights: {fold_weights}, AUC: {fold_auc:.6f}")

    best_weights = find_best_weights(model_oof, y)
    final_oof = sum(model_oof[name] * weight for name, weight in best_weights.items())
    final_test = sum(model_test_preds[name] * weight for name, weight in best_weights.items())
    full_auc = roc_auc_score(y, final_oof)
    print(f"Full CV Ensemble AUC: {full_auc:.6f}")
    print(f"OOF-selected weights: {best_weights}")
    return final_test, full_auc, best_weights, selected_features


def main() -> None:
    train_df, test_df = build_dataset()
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    preds, auc, weights, selected_features = train_and_predict(train_df, test_df)

    submission = pd.DataFrame(
        {
            "SK_ID_CURR": test_df["SK_ID_CURR"].astype(int),
            "TARGET": preds,
        }
    )
    output_path = Path(__file__).resolve().parent / "submission4.csv"
    submission.to_csv(output_path, index=False)
    print(f"Saved submission to {output_path}")
    print(f"Final CV: {auc:.6f}, weights: {weights}")
    print(f"Selected feature count: {len(selected_features)}")


if __name__ == "__main__":
    main()
