from __future__ import annotations

import gc
import re
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)

DATA_DIR = Path(__file__).resolve().parent / "home-credit-default-risk"
N_FOLDS = 5
SEED = 42
TE_SMOOTHING = 40
TE_MIN_SAMPLES = 80
DEBUG = False


def has_gpu() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


HAS_GPU = has_gpu()


def read_csv(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    print(f"Loading {path.name} ...")
    df = pd.read_csv(path, usecols=usecols, nrows=5000 if DEBUG else None)
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
        elif df[col].dtype == "int64":
            if df[col].min() >= np.iinfo(np.int32).min and df[col].max() <= np.iinfo(np.int32).max:
                df[col] = df[col].astype("int32")
    return df


def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        col_type = df[col].dtype
        if str(col_type).startswith("float"):
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif str(col_type).startswith("int"):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif col_type == bool:
            df[col] = df[col].astype(np.int8)
    return df


def clean_names(df: pd.DataFrame) -> pd.DataFrame:
    clean = {c: re.sub(r"[^A-Za-z0-9_]+", "_", str(c)) for c in df.columns}
    clean = {k: re.sub(r"_+", "_", v).strip("_") for k, v in clean.items()}
    return df.rename(columns=clean)


def is_numeric_col(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def is_categorical_col(series: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series) or pd.api.types.is_categorical_dtype(series)


def time_weighted_agg(
    df: pd.DataFrame,
    group_col: str,
    value_cols: list[str],
    time_col: str,
    prefix: str,
    decay: float = 0.002,
) -> pd.DataFrame:
    ids = df[group_col].unique()
    result = pd.DataFrame({group_col: ids})
    w = np.exp(decay * df[time_col].values.astype("float64"))
    for vc in value_cols:
        mask = df[vc].notna().values
        if mask.sum() == 0:
            result[f"{prefix}_{vc}_TWMEAN"] = np.nan
            continue
        grp_vals = df[group_col].values[mask]
        wv = df[vc].values[mask].astype("float64") * w[mask]
        ww = w[mask]
        temp = pd.DataFrame({group_col: grp_vals, "_wv": wv, "_w": ww})
        agg = temp.groupby(group_col)[["_wv", "_w"]].sum()
        agg[f"{prefix}_{vc}_TWMEAN"] = (agg["_wv"] / agg["_w"]).astype("float32")
        result = result.merge(agg[[f"{prefix}_{vc}_TWMEAN"]].reset_index(), on=group_col, how="left")
    return reduce_memory_usage(result)


def compute_trend(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    time_col: str,
    prefix: str,
) -> pd.DataFrame:
    temp = df[[group_col, value_col, time_col]].dropna().copy()
    if len(temp) == 0:
        return pd.DataFrame({group_col: df[group_col].unique(), f"{prefix}_{value_col}_TREND": np.nan})
    gcounts = temp.groupby(group_col)[value_col].transform("count")
    temp = temp[gcounts >= 3].copy()
    if len(temp) == 0:
        return pd.DataFrame({group_col: df[group_col].unique(), f"{prefix}_{value_col}_TREND": np.nan})
    g = temp.groupby(group_col)
    mt = g[time_col].transform("mean").astype("float64")
    mv = g[value_col].transform("mean").astype("float64")
    dt = temp[time_col].astype("float64") - mt
    dv = temp[value_col].astype("float64") - mv
    temp["_dtdv"] = dt * dv
    temp["_dt2"] = dt**2
    agg = temp.groupby(group_col)[["_dtdv", "_dt2"]].sum()
    agg[f"{prefix}_{value_col}_TREND"] = (agg["_dtdv"] / (agg["_dt2"] + 1e-8)).astype("float32")
    return agg[[f"{prefix}_{value_col}_TREND"]].reset_index()


def agg_time_window(
    df: pd.DataFrame,
    group_col: str,
    cols: list[str],
    time_col: str,
    cutoff: int,
    prefix: str,
) -> pd.DataFrame:
    sub = df[df[time_col] >= cutoff]
    if len(sub) == 0:
        return pd.DataFrame({group_col: df[group_col].unique()})
    valid_cols = [c for c in cols if c in sub.columns]
    if not valid_cols:
        return pd.DataFrame({group_col: df[group_col].unique()})
    agg = sub.groupby(group_col)[valid_cols].agg(["mean", "max", "sum"])
    agg.columns = [f"{prefix}_{c[0]}_{c[1].upper()}" for c in agg.columns]
    cnt = sub.groupby(group_col).size().reset_index(name=f"{prefix}_COUNT")
    agg = agg.reset_index().merge(cnt, on=group_col, how="left")
    return reduce_memory_usage(agg)


def add_frequency_features(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = pd.concat([train_df.drop(columns=["TARGET"], errors="ignore"), test_df], axis=0, ignore_index=True)
    for col in cols:
        if col not in full.columns:
            continue
        vc = full[col].fillna("__nan__").value_counts(dropna=False)
        full[f"{col}_FREQ"] = full[col].fillna("__nan__").map(vc).astype("float32")
        full[f"{col}_FREQ_NORM"] = (full[f"{col}_FREQ"] / len(full)).astype("float32")
    out_train = full.iloc[: len(train_df)].copy()
    out_test = full.iloc[len(train_df) :].copy()
    if "TARGET" in train_df.columns:
        out_train["TARGET"] = train_df["TARGET"].values
    return reduce_memory_usage(out_train), reduce_memory_usage(out_test)


def add_groupby_ratio_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = pd.concat([train_df.drop(columns=["TARGET"], errors="ignore"), test_df], axis=0, ignore_index=True)
    for cat in cat_cols:
        if cat not in full.columns:
            continue
        for num in num_cols:
            if num not in full.columns:
                continue
            gp = full.groupby(cat)[num].agg(["mean", "median", "std"]).reset_index()
            gp.columns = [cat, f"GB_{cat}_{num}_MEAN", f"GB_{cat}_{num}_MEDIAN", f"GB_{cat}_{num}_STD"]
            full = full.merge(gp, on=cat, how="left")
            full[f"GB_{cat}_{num}_DIFF"] = full[num] - full[f"GB_{cat}_{num}_MEAN"]
            full[f"GB_{cat}_{num}_RATIO"] = full[num] / (full[f"GB_{cat}_{num}_MEAN"] + 1e-6)
    full = reduce_memory_usage(full)
    out_train = full.iloc[: len(train_df)].copy()
    out_test = full.iloc[len(train_df) :].copy()
    if "TARGET" in train_df.columns:
        out_train["TARGET"] = train_df["TARGET"].values
    return out_train, out_test


def add_target_encoding(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    cols: list[str],
    n_splits: int = 5,
    smoothing: int = 40,
    min_samples_leaf: int = 80,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.copy()
    test_df = test_df.copy()
    global_mean = train_df[target_col].mean()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for col in cols:
        if col not in train_df.columns:
            continue
        new_col = f"{col}_TE"
        if new_col in train_df.columns:
            continue
        tr_enc = np.zeros(len(train_df), dtype="float32")
        for tr_idx, va_idx in skf.split(train_df, train_df[target_col]):
            tr_fold = train_df.iloc[tr_idx]
            stats = tr_fold.groupby(col)[target_col].agg(["mean", "count"])
            smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
            smooth[stats["count"] < min_samples_leaf] = global_mean
            tr_enc[va_idx] = train_df.iloc[va_idx][col].map(smooth).fillna(global_mean).values.astype("float32")
        full_stats = train_df.groupby(col)[target_col].agg(["mean", "count"])
        full_smooth = (full_stats["count"] * full_stats["mean"] + smoothing * global_mean) / (
            full_stats["count"] + smoothing
        )
        full_smooth[full_stats["count"] < min_samples_leaf] = global_mean
        te_enc = test_df[col].map(full_smooth).fillna(global_mean).values.astype("float32")
        train_df[new_col] = tr_enc
        test_df[new_col] = te_enc
    return reduce_memory_usage(train_df), reduce_memory_usage(test_df)


def application_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["DAYS_EMPLOYED"] = out["DAYS_EMPLOYED"].replace(365243, np.nan)
    out["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype("int8")

    def row_sum_numeric(frame: pd.DataFrame, cols: list[str], dtype: str = "float32") -> pd.Series:
        cols = [c for c in cols if c in frame.columns]
        if not cols:
            return pd.Series(np.zeros(len(frame), dtype=dtype), index=frame.index)
        block = frame[cols].apply(pd.to_numeric, errors="coerce")
        return block.sum(axis=1).astype(dtype)

    ext = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    out["EXT_MEAN"] = out[ext].mean(axis=1)
    out["EXT_STD"] = out[ext].std(axis=1)
    out["EXT_PROD"] = out["EXT_SOURCE_1"] * out["EXT_SOURCE_2"] * out["EXT_SOURCE_3"]
    out["EXT_MIN"] = out[ext].min(axis=1)
    out["EXT_MAX"] = out[ext].max(axis=1)
    out["EXT_NANCOUNT"] = out[ext].isna().sum(axis=1)
    out["EXT_S1xS2"] = out["EXT_SOURCE_1"] * out["EXT_SOURCE_2"]
    out["EXT_S1xS3"] = out["EXT_SOURCE_1"] * out["EXT_SOURCE_3"]
    out["EXT_S2xS3"] = out["EXT_SOURCE_2"] * out["EXT_SOURCE_3"]
    out["EXT_S2divS3"] = out["EXT_SOURCE_2"] / (out["EXT_SOURCE_3"] + 1e-4)
    out["EXT_S1divS2"] = out["EXT_SOURCE_1"] / (out["EXT_SOURCE_2"] + 1e-4)
    for i in [1, 2, 3]:
        col = f"EXT_SOURCE_{i}"
        out[f"{col}_SQ"] = out[col] ** 2
        out[f"{col}_CB"] = out[col] ** 3
    out["EXT_S2xBIRTH"] = out["EXT_SOURCE_2"] * out["DAYS_BIRTH"]
    out["EXT_S1xBIRTH"] = out["EXT_SOURCE_1"] * out["DAYS_BIRTH"]
    out["EXT_S3xBIRTH"] = out["EXT_SOURCE_3"] * out["DAYS_BIRTH"]
    out["EXT_S2xEMPL"] = out["EXT_SOURCE_2"] * out["DAYS_EMPLOYED"]
    out["EXT_S3xEMPL"] = out["EXT_SOURCE_3"] * out["DAYS_EMPLOYED"]

    out["GP1"] = out["EXT_SOURCE_2"] ** 2 * out["EXT_SOURCE_3"]
    out["GP2"] = out["EXT_SOURCE_1"] * out["DAYS_BIRTH"] / (out["AMT_ANNUITY"] + 1)
    out["GP3"] = out["EXT_SOURCE_2"] * out["REGION_RATING_CLIENT_W_CITY"]
    out["GP4"] = out["EXT_SOURCE_3"] * np.log1p(np.abs(out["DAYS_BIRTH"]))
    out["GP5"] = out["AMT_ANNUITY"] * out["EXT_SOURCE_3"] / (out["AMT_INCOME_TOTAL"] + 1)
    out["GP6"] = out["EXT_SOURCE_1"] * out["DAYS_ID_PUBLISH"] / (out["DAYS_BIRTH"] + 1)
    out["GP7"] = out["EXT_SOURCE_2"] * out["AMT_CREDIT"] / (out["AMT_GOODS_PRICE"] + 1)
    out["GP8"] = out["EXT_SOURCE_1"] * out["EXT_SOURCE_2"] * out["EXT_SOURCE_3"] / (out["AMT_CREDIT"] + 1)
    out["GP9"] = out["EXT_MEAN"] * out["DAYS_EMPLOYED"] / (out["DAYS_BIRTH"] - 1)
    out["GP10"] = (out["AMT_GOODS_PRICE"] - out["AMT_CREDIT"]) * out["EXT_SOURCE_2"] / (out["AMT_ANNUITY"] + 1)

    out["CREDIT_INCOME_RATIO"] = out["AMT_CREDIT"] / (out["AMT_INCOME_TOTAL"] + 1)
    out["ANNUITY_INCOME_RATIO"] = out["AMT_ANNUITY"] / (out["AMT_INCOME_TOTAL"] + 1)
    out["CREDIT_ANNUITY_RATIO"] = out["AMT_CREDIT"] / (out["AMT_ANNUITY"] + 1)
    out["CREDIT_GOODS_RATIO"] = out["AMT_CREDIT"] / (out["AMT_GOODS_PRICE"] + 1)
    out["GOODS_INCOME_RATIO"] = out["AMT_GOODS_PRICE"] / (out["AMT_INCOME_TOTAL"] + 1)
    out["INCOME_PER_CHILD"] = out["AMT_INCOME_TOTAL"] / (out["CNT_CHILDREN"] + 1)
    out["INCOME_PER_FAM"] = out["AMT_INCOME_TOTAL"] / (out["CNT_FAM_MEMBERS"] + 1)
    out["ANNUITY_CREDIT_RATIO"] = out["AMT_ANNUITY"] / (out["AMT_CREDIT"] + 1)
    out["PAYMENT_LENGTH"] = out["AMT_CREDIT"] / (out["AMT_ANNUITY"] + 1)
    out["DOWN_PAYMENT"] = out["AMT_GOODS_PRICE"] - out["AMT_CREDIT"]
    out["DOWN_PAYMENT_RATIO"] = out["DOWN_PAYMENT"] / (out["AMT_GOODS_PRICE"] + 1)

    out["DAYS_BIRTH_YRS"] = out["DAYS_BIRTH"] / -365.25
    out["DAYS_EMPLOYED_YRS"] = out["DAYS_EMPLOYED"] / -365.25
    out["EMPLOYED_TO_BIRTH"] = out["DAYS_EMPLOYED"] / (out["DAYS_BIRTH"] + 1)
    out["CAR_AGE_TO_BIRTH"] = out["OWN_CAR_AGE"] / (out["DAYS_BIRTH_YRS"] + 1)
    out["ID_PUBLISH_TO_BIRTH"] = out["DAYS_ID_PUBLISH"] / (out["DAYS_BIRTH"] + 1)
    out["PHONE_TO_BIRTH"] = out["DAYS_LAST_PHONE_CHANGE"] / (out["DAYS_BIRTH"] + 1)
    out["PHONE_TO_EMPLOYED"] = out["DAYS_LAST_PHONE_CHANGE"] / (out["DAYS_EMPLOYED"] + 1)
    out["REG_TO_BIRTH"] = out["DAYS_REGISTRATION"] / (out["DAYS_BIRTH"] + 1)
    out["AGE_RANGE"] = pd.cut(
        out["DAYS_BIRTH_YRS"], bins=[0, 25, 30, 35, 40, 45, 50, 55, 60, 65, 100], labels=False
    )
    out["INCOME_EMPLOYED"] = out["AMT_INCOME_TOTAL"] * out["DAYS_EMPLOYED_YRS"]

    doc_cols = [c for c in out.columns if "FLAG_DOCUMENT" in c]
    out["DOCUMENT_COUNT"] = row_sum_numeric(out, doc_cols, dtype="float32")
    out["DEF_30_RATIO"] = out["DEF_30_CNT_SOCIAL_CIRCLE"] / (out["OBS_30_CNT_SOCIAL_CIRCLE"] + 1)
    out["DEF_60_RATIO"] = out["DEF_60_CNT_SOCIAL_CIRCLE"] / (out["OBS_60_CNT_SOCIAL_CIRCLE"] + 1)
    out["APP_NULLS"] = out.isna().sum(axis=1).astype("int16")
    out["CITY_RATING_x_EXT2"] = out["REGION_RATING_CLIENT_W_CITY"] * out["EXT_SOURCE_2"]

    out["EMPLOYED_TO_ID"] = out["DAYS_EMPLOYED"] / (out["DAYS_ID_PUBLISH"] + 1)
    out["ID_TO_BIRTH_RATIO"] = out["DAYS_ID_PUBLISH"] / (out["DAYS_BIRTH"] + 1)
    out["REG_TO_EMPLOYED_RATIO"] = out["DAYS_REGISTRATION"] / (out["DAYS_EMPLOYED"] + 1)
    out["CREDIT_PER_PERSON"] = out["AMT_CREDIT"] / (out["CNT_FAM_MEMBERS"] + 1)
    out["ANNUITY_PER_PERSON"] = out["AMT_ANNUITY"] / (out["CNT_FAM_MEMBERS"] + 1)
    out["INCOME_CREDIT_PERC"] = out["AMT_INCOME_TOTAL"] / (out["AMT_CREDIT"] + 1)
    out["INCOME_ANNUITY_PERC"] = out["AMT_INCOME_TOTAL"] / (out["AMT_ANNUITY"] + 1)
    out["EXT_RANGE"] = out["EXT_MAX"] - out["EXT_MIN"]
    out["EXT_SOURCE_SPREAD"] = out["EXT_STD"] / (out["EXT_MEAN"] + 1e-4)
    out["PHONE_MINUS_REG"] = out["DAYS_LAST_PHONE_CHANGE"] - out["DAYS_REGISTRATION"]
    out["CAR_EMPLOYED_RATIO"] = out["OWN_CAR_AGE"] / (out["DAYS_EMPLOYED_YRS"] + 1)
    out["CHILDREN_RATIO"] = out["CNT_CHILDREN"] / (out["CNT_FAM_MEMBERS"] + 1)
    out["OBS_30_60_RATIO"] = out["OBS_30_CNT_SOCIAL_CIRCLE"] / (out["OBS_60_CNT_SOCIAL_CIRCLE"] + 1)
    out["DEF_30_60_RATIO"] = out["DEF_30_CNT_SOCIAL_CIRCLE"] / (out["DEF_60_CNT_SOCIAL_CIRCLE"] + 1)
    amt_req_cols = [c for c in out.columns if c.startswith("AMT_REQ_CREDIT_BUREAU_")]
    out["AMT_REQ_SUM"] = row_sum_numeric(out, amt_req_cols, dtype="float32")
    explicit_contact_cols = [
        c
        for c in [
            "FLAG_MOBIL",
            "FLAG_EMP_PHONE",
            "FLAG_WORK_PHONE",
            "FLAG_CONT_MOBILE",
            "FLAG_PHONE",
            "FLAG_EMAIL",
        ]
        if c in out.columns
    ]
    if explicit_contact_cols:
        out["FLAG_CONTACTS_SUM"] = row_sum_numeric(out, explicit_contact_cols, dtype="float32")

    out["CREDIT_TERM"] = out["AMT_ANNUITY"] / (out["AMT_CREDIT"] + 1)
    out["DAYS_EMPLOYED_PERC"] = out["DAYS_EMPLOYED"] / (out["DAYS_BIRTH"] + 1)
    out["INCOME_CREDIT_PERC2"] = out["AMT_INCOME_TOTAL"] / (out["AMT_CREDIT"] + 1)
    out["EXT_WEIGHTED"] = 2 * out["EXT_SOURCE_2"] + out["EXT_SOURCE_3"] + 0.5 * out["EXT_SOURCE_1"]
    out["REGION_POP_x_EXT"] = out["REGION_POPULATION_RELATIVE"] * out["EXT_MEAN"]
    out["HOUR_APPR_x_EXT2"] = out["HOUR_APPR_PROCESS_START"] * out["EXT_SOURCE_2"]
    out["LIVE_REGION_DIFF"] = (
        out["REG_REGION_NOT_LIVE_REGION"].astype(float)
        + out["REG_REGION_NOT_WORK_REGION"].astype(float)
        + out.get("LIVE_REGION_NOT_WORK_REGION", pd.Series(0, index=out.index)).astype(float)
    )
    return reduce_memory_usage(out)


def bureau_and_balance_features() -> pd.DataFrame:
    bureau = read_csv(DATA_DIR / "bureau.csv")
    bb = read_csv(DATA_DIR / "bureau_balance.csv")

    bb_counts = bb.pivot_table(index="SK_ID_BUREAU", columns="STATUS", values="MONTHS_BALANCE", aggfunc="count", fill_value=0)
    bb_counts.columns = [f"BB_STATUS_{c}" for c in bb_counts.columns]
    bb_counts = bb_counts.reset_index()
    bb_months = bb.groupby("SK_ID_BUREAU")["MONTHS_BALANCE"].agg(
        BB_MONTHS_MIN="min",
        BB_MONTHS_MAX="max",
        BB_MONTHS_SIZE="size",
    ).reset_index()
    bb_agg = bb_months.merge(bb_counts, on="SK_ID_BUREAU", how="left")
    bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")
    del bb, bb_counts, bb_months, bb_agg
    gc.collect()

    bureau["CREDIT_DURATION"] = bureau["DAYS_CREDIT_ENDDATE"] - bureau["DAYS_CREDIT"]
    bureau["ENDDATE_DIFF"] = bureau["DAYS_CREDIT_ENDDATE"] - bureau["DAYS_ENDDATE_FACT"]
    bureau["DEBT_CREDIT_RATIO"] = bureau["AMT_CREDIT_SUM_DEBT"] / (bureau["AMT_CREDIT_SUM"] + 1)
    bureau["OVERDUE_DEBT_RATIO"] = bureau["AMT_CREDIT_SUM_OVERDUE"] / (bureau["AMT_CREDIT_SUM_DEBT"] + 1)
    bureau["AMT_ANNUITY_CREDIT"] = bureau["AMT_ANNUITY"] / (bureau["AMT_CREDIT_SUM"] + 1)
    bureau["CREDIT_OVERDUE_RATIO"] = bureau["AMT_CREDIT_SUM_OVERDUE"] / (bureau["AMT_CREDIT_SUM"] + 1)
    bureau["DAYS_CREDIT_UPDATE_DIFF"] = bureau["DAYS_CREDIT_UPDATE"] - bureau["DAYS_CREDIT"]

    num_cols = [c for c in bureau.columns if is_numeric_col(bureau[c]) and c not in ["SK_ID_BUREAU", "SK_ID_CURR"]]
    buro_num = bureau.groupby("SK_ID_CURR")[num_cols].agg(["min", "max", "mean", "sum", "var"])
    buro_num.columns = [f"BURO_{c[0]}_{c[1].upper()}" for c in buro_num.columns]
    buro_feat = buro_num.reset_index()

    cat_cols = [c for c in bureau.columns if is_categorical_col(bureau[c])]
    if cat_cols:
        buro_cat = pd.get_dummies(bureau[["SK_ID_CURR"] + cat_cols], columns=cat_cols, dummy_na=True)
        buro_cat = buro_cat.groupby("SK_ID_CURR").mean().reset_index()
        buro_cat.columns = ["SK_ID_CURR"] + [f"BURO_{c}" for c in buro_cat.columns if c != "SK_ID_CURR"]
        buro_feat = buro_feat.merge(buro_cat, on="SK_ID_CURR", how="left")

    buro_feat = buro_feat.merge(bureau.groupby("SK_ID_CURR").size().reset_index(name="BURO_COUNT"), on="SK_ID_CURR", how="left")

    for status in ["Active", "Closed"]:
        sub = bureau[bureau["CREDIT_ACTIVE"] == status]
        if len(sub) > 0:
            key = ["AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "DAYS_CREDIT", "DAYS_CREDIT_ENDDATE", "DEBT_CREDIT_RATIO"]
            key = [c for c in key if c in sub.columns]
            sa = sub.groupby("SK_ID_CURR")[key].agg(["mean", "sum", "max", "min"])
            sa.columns = [f"BURO_{status.upper()}_{c[0]}_{c[1].upper()}" for c in sa.columns]
            sa = sa.reset_index()
            sc = sub.groupby("SK_ID_CURR").size().reset_index(name=f"BURO_{status.upper()}_COUNT")
            buro_feat = buro_feat.merge(sa.merge(sc, on="SK_ID_CURR", how="left"), on="SK_ID_CURR", how="left")

    tw_cols = ["AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "CREDIT_DAY_OVERDUE", "DEBT_CREDIT_RATIO"]
    tw_cols = [c for c in tw_cols if c in bureau.columns]
    for days, label in [(-180, "6M"), (-365, "1Y"), (-730, "2Y"), (-1095, "3Y"), (-1825, "5Y")]:
        tw = agg_time_window(bureau, "SK_ID_CURR", tw_cols, "DAYS_CREDIT", days, f"BURO_{label}")
        buro_feat = buro_feat.merge(tw, on="SK_ID_CURR", how="left")

    tw_feats = time_weighted_agg(
        bureau,
        "SK_ID_CURR",
        ["AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "DEBT_CREDIT_RATIO"],
        "DAYS_CREDIT",
        "BURO",
        decay=0.001,
    )
    buro_feat = buro_feat.merge(tw_feats, on="SK_ID_CURR", how="left")

    for col in ["AMT_CREDIT_SUM_DEBT", "DEBT_CREDIT_RATIO"]:
        trend = compute_trend(bureau, "SK_ID_CURR", col, "DAYS_CREDIT", "BURO")
        buro_feat = buro_feat.merge(trend, on="SK_ID_CURR", how="left")

    bureau_sorted = bureau.sort_values("DAYS_CREDIT", ascending=False)
    last_bureau = bureau_sorted.groupby("SK_ID_CURR").first().reset_index()
    for col in ["DAYS_CREDIT", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "DEBT_CREDIT_RATIO", "CREDIT_DAY_OVERDUE"]:
        if col in last_bureau.columns:
            buro_feat = buro_feat.merge(
                last_bureau[["SK_ID_CURR", col]].rename(columns={col: f"BURO_LAST_{col}"}),
                on="SK_ID_CURR",
                how="left",
            )

    buro_feat = buro_feat.merge(
        bureau.groupby("SK_ID_CURR")["CREDIT_TYPE"].nunique().reset_index(name="BURO_CREDIT_TYPE_NUNIQUE"),
        on="SK_ID_CURR",
        how="left",
    )
    buro_feat = buro_feat.merge(
        (bureau.groupby("SK_ID_CURR")["CREDIT_DAY_OVERDUE"].max() > 0).astype("int8").reset_index(name="BURO_OVERDUE_EVER"),
        on="SK_ID_CURR",
        how="left",
    )

    del bureau
    gc.collect()
    return reduce_memory_usage(buro_feat)


def previous_application_features() -> pd.DataFrame:
    prev = read_csv(DATA_DIR / "previous_application.csv")
    for col in [c for c in prev.columns if "DAYS_" in c]:
        prev[col] = prev[col].replace(365243, np.nan)
    prev["APP_CREDIT_RATIO"] = prev["AMT_APPLICATION"] / (prev["AMT_CREDIT"] + 1)
    prev["CREDIT_GOODS_P"] = prev["AMT_CREDIT"] / (prev["AMT_GOODS_PRICE"] + 1)
    prev["APP_GOODS_RATIO"] = prev["AMT_APPLICATION"] / (prev["AMT_GOODS_PRICE"] + 1)
    prev["DAYS_FIRST_DUE_DIFF"] = prev["DAYS_FIRST_DUE"] - prev["DAYS_FIRST_DRAWING"]
    prev["DAYS_LAST_DUE_DIFF"] = prev["DAYS_LAST_DUE_1ST_VERSION"] - prev["DAYS_LAST_DUE"]
    prev["DOWN_PAYMENT_P"] = prev["AMT_DOWN_PAYMENT"] / (prev["AMT_CREDIT"] + 1)
    prev["INTEREST_SHARE"] = prev["CNT_PAYMENT"] * prev["AMT_ANNUITY"] - prev["AMT_CREDIT"]
    prev["INTEREST_RATE"] = prev["INTEREST_SHARE"] / (prev["AMT_CREDIT"] + 1)

    num_cols = [c for c in prev.columns if is_numeric_col(prev[c]) and c not in ["SK_ID_CURR", "SK_ID_PREV"]]
    prev_num = prev.groupby("SK_ID_CURR")[num_cols].agg(["min", "max", "mean", "sum", "var"])
    prev_num.columns = [f"PREV_{c[0]}_{c[1].upper()}" for c in prev_num.columns]
    prev_feat = prev_num.reset_index()

    cat_cols = [c for c in prev.columns if is_categorical_col(prev[c])]
    if cat_cols:
        prev_cat = pd.get_dummies(prev[["SK_ID_CURR"] + cat_cols], columns=cat_cols, dummy_na=True)
        prev_cat = prev_cat.groupby("SK_ID_CURR").mean().reset_index()
        prev_cat.columns = ["SK_ID_CURR"] + [f"PREV_{c}" for c in prev_cat.columns if c != "SK_ID_CURR"]
        prev_feat = prev_feat.merge(prev_cat, on="SK_ID_CURR", how="left")

    prev_feat = prev_feat.merge(prev.groupby("SK_ID_CURR").size().reset_index(name="PREV_COUNT"), on="SK_ID_CURR", how="left")

    for status in ["Approved", "Refused", "Canceled"]:
        sub = prev[prev["NAME_CONTRACT_STATUS"] == status]
        if len(sub) > 0:
            sa = sub.groupby("SK_ID_CURR")[["AMT_CREDIT", "AMT_APPLICATION", "AMT_ANNUITY", "DAYS_DECISION"]].agg(
                ["mean", "max", "min"]
            )
            sa.columns = [f"PREV_{status.upper()}_{c[0]}_{c[1].upper()}" for c in sa.columns]
            sa = sa.reset_index()
            sc = sub.groupby("SK_ID_CURR").size().reset_index(name=f"PREV_{status.upper()}_COUNT")
            prev_feat = prev_feat.merge(sa.merge(sc, on="SK_ID_CURR", how="left"), on="SK_ID_CURR", how="left")

    for ctype in ["Cash loans", "Revolving loans"]:
        sub = prev[prev["NAME_CONTRACT_TYPE"] == ctype]
        if len(sub) > 0:
            label = "CASH" if "Cash" in ctype else "REVOLV"
            sa = sub.groupby("SK_ID_CURR")[["AMT_CREDIT", "AMT_ANNUITY", "APP_CREDIT_RATIO"]].agg(["mean", "sum", "max"])
            sa.columns = [f"PREV_{label}_{c[0]}_{c[1].upper()}" for c in sa.columns]
            sa = sa.reset_index()
            sc = sub.groupby("SK_ID_CURR").size().reset_index(name=f"PREV_{label}_COUNT")
            prev_feat = prev_feat.merge(sa.merge(sc, on="SK_ID_CURR", how="left"), on="SK_ID_CURR", how="left")

    tw_cols = ["AMT_CREDIT", "AMT_ANNUITY", "APP_CREDIT_RATIO", "INTEREST_RATE"]
    tw_cols = [c for c in tw_cols if c in prev.columns]
    for days, label in [(-180, "6M"), (-365, "1Y"), (-730, "2Y"), (-1095, "3Y")]:
        tw = agg_time_window(prev, "SK_ID_CURR", tw_cols, "DAYS_DECISION", days, f"PREV_{label}")
        prev_feat = prev_feat.merge(tw, on="SK_ID_CURR", how="left")

    tw_feats = time_weighted_agg(
        prev,
        "SK_ID_CURR",
        ["AMT_CREDIT", "AMT_ANNUITY", "APP_CREDIT_RATIO"],
        "DAYS_DECISION",
        "PREV",
        decay=0.001,
    )
    prev_feat = prev_feat.merge(tw_feats, on="SK_ID_CURR", how="left")

    app_rate = prev.groupby("SK_ID_CURR")["NAME_CONTRACT_STATUS"].apply(lambda x: (x == "Approved").mean()).reset_index(name="PREV_APPROVAL_RATE")
    prev_feat = prev_feat.merge(app_rate, on="SK_ID_CURR", how="left")

    prev_sorted = prev.sort_values("DAYS_DECISION", ascending=False)
    last_prev = prev_sorted.groupby("SK_ID_CURR").first().reset_index()
    for col in ["DAYS_DECISION", "AMT_CREDIT", "APP_CREDIT_RATIO", "INTEREST_RATE"]:
        if col in last_prev.columns:
            prev_feat = prev_feat.merge(
                last_prev[["SK_ID_CURR", col]].rename(columns={col: f"PREV_LAST_{col}"}),
                on="SK_ID_CURR",
                how="left",
            )

    del prev
    gc.collect()
    return reduce_memory_usage(prev_feat)


def pos_cash_features() -> pd.DataFrame:
    pos = read_csv(DATA_DIR / "POS_CASH_balance.csv")
    pos["SK_DPD_RATIO"] = pos["SK_DPD"] / (pos["SK_DPD_DEF"] + 1)
    pos["LATE_POS"] = (pos["SK_DPD"] > 0).astype("int8")

    num_cols = [c for c in pos.columns if is_numeric_col(pos[c]) and c not in ["SK_ID_CURR", "SK_ID_PREV"]]
    pn = pos.groupby("SK_ID_CURR")[num_cols].agg(["min", "max", "mean", "sum", "var"])
    pn.columns = [f"POS_{c[0]}_{c[1].upper()}" for c in pn.columns]
    pn = pn.reset_index()

    if "NAME_CONTRACT_STATUS" in pos.columns:
        pc = pd.get_dummies(pos[["SK_ID_CURR", "NAME_CONTRACT_STATUS"]], columns=["NAME_CONTRACT_STATUS"], dummy_na=True)
        pc = pc.groupby("SK_ID_CURR").mean().reset_index()
        pc.columns = ["SK_ID_CURR"] + [f"POS_{c}" for c in pc.columns if c != "SK_ID_CURR"]
        pn = pn.merge(pc, on="SK_ID_CURR", how="left")

    pn = pn.merge(pos.groupby("SK_ID_CURR").size().reset_index(name="POS_COUNT"), on="SK_ID_CURR", how="left")
    pn = pn.merge(pos.groupby("SK_ID_CURR")["LATE_POS"].mean().reset_index(name="POS_LATE_RATE"), on="SK_ID_CURR", how="left")

    tw_cols_pos = ["SK_DPD", "SK_DPD_DEF", "CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE"]
    for months, label in [(-3, "3M"), (-6, "6M"), (-12, "12M"), (-24, "24M")]:
        tw = agg_time_window(pos, "SK_ID_CURR", tw_cols_pos, "MONTHS_BALANCE", months, f"POS_{label}")
        pn = pn.merge(tw, on="SK_ID_CURR", how="left")

    loan_agg = pos.groupby(["SK_ID_CURR", "SK_ID_PREV"]).agg(
        POS_PL_DPD_MAX=("SK_DPD", "max"),
        POS_PL_DPD_MEAN=("SK_DPD", "mean"),
        POS_PL_LATE_RATE=("LATE_POS", "mean"),
        POS_PL_MONTHS=("MONTHS_BALANCE", "count"),
    ).reset_index()
    pl_cols = [c for c in loan_agg.columns if c.startswith("POS_PL_")]
    pos_pl = loan_agg.groupby("SK_ID_CURR")[pl_cols].agg(["mean", "max", "std"])
    pos_pl.columns = [f"{c[0]}_{c[1].upper()}" for c in pos_pl.columns]
    pos_pl = pos_pl.reset_index()
    pn = pn.merge(pos_pl, on="SK_ID_CURR", how="left")

    tw_feats = time_weighted_agg(pos, "SK_ID_CURR", ["SK_DPD", "CNT_INSTALMENT_FUTURE"], "MONTHS_BALANCE", "POS", decay=0.02)
    pn = pn.merge(tw_feats, on="SK_ID_CURR", how="left")

    trend = compute_trend(pos, "SK_ID_CURR", "SK_DPD", "MONTHS_BALANCE", "POS")
    pn = pn.merge(trend, on="SK_ID_CURR", how="left")

    if "NAME_CONTRACT_STATUS" in pos.columns:
        completed = pos[pos["NAME_CONTRACT_STATUS"] == "Completed"]
        comp_rate = completed.groupby("SK_ID_CURR").size() / pos.groupby("SK_ID_CURR").size()
        comp_rate = comp_rate.reset_index(name="POS_COMPLETED_RATE")
        pn = pn.merge(comp_rate, on="SK_ID_CURR", how="left")

    del pos, loan_agg
    gc.collect()
    return reduce_memory_usage(pn)


def credit_card_features() -> pd.DataFrame:
    cc = read_csv(DATA_DIR / "credit_card_balance.csv")
    cc["CC_BAL_LIM_RATIO"] = cc["AMT_BALANCE"] / (cc["AMT_CREDIT_LIMIT_ACTUAL"] + 1)
    cc["CC_PAY_TOTAL_RATIO"] = cc["AMT_PAYMENT_TOTAL_CURRENT"] / (cc["AMT_TOTAL_RECEIVABLE"] + 1)
    cc["CC_DRAW_LIM"] = cc["AMT_DRAWINGS_CURRENT"] / (cc["AMT_CREDIT_LIMIT_ACTUAL"] + 1)
    cc["CC_LATE"] = (cc["SK_DPD"] > 0).astype("int8")
    cc["CC_MIN_PAY_RATIO"] = cc["AMT_INST_MIN_REGULARITY"] / (cc["AMT_PAYMENT_CURRENT"] + 1)

    num_cols = [c for c in cc.columns if is_numeric_col(cc[c]) and c not in ["SK_ID_CURR", "SK_ID_PREV"]]
    cn = cc.groupby("SK_ID_CURR")[num_cols].agg(["min", "max", "mean", "sum", "var"])
    cn.columns = [f"CC_{c[0]}_{c[1].upper()}" for c in cn.columns]
    cn = cn.reset_index()
    cn = cn.merge(cc.groupby("SK_ID_CURR").size().reset_index(name="CC_COUNT"), on="SK_ID_CURR", how="left")
    cn = cn.merge(cc.groupby("SK_ID_CURR")["CC_LATE"].mean().reset_index(name="CC_LATE_RATE"), on="SK_ID_CURR", how="left")

    tw_cols_cc = ["AMT_BALANCE", "CC_BAL_LIM_RATIO", "CC_DRAW_LIM", "SK_DPD"]
    for months, label in [(-3, "3M"), (-6, "6M"), (-12, "12M"), (-24, "24M")]:
        tw = agg_time_window(cc, "SK_ID_CURR", tw_cols_cc, "MONTHS_BALANCE", months, f"CC_{label}")
        cn = cn.merge(tw, on="SK_ID_CURR", how="left")

    loan_agg = cc.groupby(["SK_ID_CURR", "SK_ID_PREV"]).agg(
        CC_PL_BAL_LIM_MAX=("CC_BAL_LIM_RATIO", "max"),
        CC_PL_BAL_LIM_MEAN=("CC_BAL_LIM_RATIO", "mean"),
        CC_PL_DRAW_MEAN=("CC_DRAW_LIM", "mean"),
        CC_PL_DPD_MAX=("SK_DPD", "max"),
        CC_PL_LATE_RATE=("CC_LATE", "mean"),
    ).reset_index()
    pl_cols = [c for c in loan_agg.columns if c.startswith("CC_PL_")]
    cc_pl = loan_agg.groupby("SK_ID_CURR")[pl_cols].agg(["mean", "max", "std"])
    cc_pl.columns = [f"{c[0]}_{c[1].upper()}" for c in cc_pl.columns]
    cc_pl = cc_pl.reset_index()
    cn = cn.merge(cc_pl, on="SK_ID_CURR", how="left")

    tw_feats = time_weighted_agg(cc, "SK_ID_CURR", ["AMT_BALANCE", "CC_BAL_LIM_RATIO", "SK_DPD"], "MONTHS_BALANCE", "CC", decay=0.02)
    cn = cn.merge(tw_feats, on="SK_ID_CURR", how="left")

    for col in ["AMT_BALANCE", "CC_BAL_LIM_RATIO"]:
        trend = compute_trend(cc, "SK_ID_CURR", col, "MONTHS_BALANCE", "CC")
        cn = cn.merge(trend, on="SK_ID_CURR", how="left")

    del cc, loan_agg
    gc.collect()
    return reduce_memory_usage(cn)


def installments_features() -> pd.DataFrame:
    ins = read_csv(DATA_DIR / "installments_payments.csv")
    ins["PAYMENT_PERC"] = (ins["AMT_PAYMENT"] / (ins["AMT_INSTALMENT"] + 0.001)).replace([np.inf, -np.inf], np.nan).astype("float32")
    ins["PAYMENT_DIFF"] = (ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]).astype("float32")
    ins["DPD"] = np.maximum(ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"], 0).astype("float32")
    ins["DBD"] = np.maximum(ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"], 0).astype("float32")
    ins["LATE_PAYMENT"] = (ins["DPD"] > 0).astype("int8")
    ins["SIGNIFICANT_UNDERPAY"] = (ins["PAYMENT_DIFF"] > 100).astype("int8")

    num_cols = [c for c in ins.columns if is_numeric_col(ins[c]) and c not in ["SK_ID_CURR", "SK_ID_PREV"]]
    i_n = ins.groupby("SK_ID_CURR")[num_cols].agg(["min", "max", "mean", "sum", "var"])
    i_n.columns = [f"INS_{c[0]}_{c[1].upper()}" for c in i_n.columns]
    i_n = i_n.reset_index()
    i_n = i_n.merge(ins.groupby("SK_ID_CURR").size().reset_index(name="INS_COUNT"), on="SK_ID_CURR", how="left")
    i_n = i_n.merge(ins.groupby("SK_ID_CURR")["LATE_PAYMENT"].mean().reset_index(name="INS_LATE_RATE"), on="SK_ID_CURR", how="left")
    i_n = i_n.merge(
        ins.groupby("SK_ID_CURR")["SIGNIFICANT_UNDERPAY"].mean().reset_index(name="INS_SIGUNDERPAY_RATE"),
        on="SK_ID_CURR",
        how="left",
    )

    tw_cols_ins = ["DPD", "PAYMENT_PERC", "PAYMENT_DIFF", "LATE_PAYMENT"]
    for days, label in [(-180, "6M"), (-365, "1Y"), (-730, "2Y")]:
        tw = agg_time_window(ins, "SK_ID_CURR", tw_cols_ins, "DAYS_INSTALMENT", days, f"INS_{label}")
        i_n = i_n.merge(tw, on="SK_ID_CURR", how="left")

    loan_agg = ins.groupby(["SK_ID_CURR", "SK_ID_PREV"]).agg(
        INS_PL_DPD_MEAN=("DPD", "mean"),
        INS_PL_DPD_MAX=("DPD", "max"),
        INS_PL_LATE_SUM=("LATE_PAYMENT", "sum"),
        INS_PL_LATE_RATE=("LATE_PAYMENT", "mean"),
        INS_PL_PAYPERC_MEAN=("PAYMENT_PERC", "mean"),
        INS_PL_PAYPERC_MIN=("PAYMENT_PERC", "min"),
        INS_PL_PAYDIFF_MAX=("PAYMENT_DIFF", "max"),
        INS_PL_COUNT=("DPD", "size"),
    ).reset_index()
    pl_cols = [c for c in loan_agg.columns if c.startswith("INS_PL_")]
    ins_pl = loan_agg.groupby("SK_ID_CURR")[pl_cols].agg(["mean", "max", "std"])
    ins_pl.columns = [f"{c[0]}_{c[1].upper()}" for c in ins_pl.columns]
    ins_pl = ins_pl.reset_index()
    i_n = i_n.merge(ins_pl, on="SK_ID_CURR", how="left")

    tw_feats = time_weighted_agg(ins, "SK_ID_CURR", ["DPD", "PAYMENT_PERC", "PAYMENT_DIFF"], "DAYS_INSTALMENT", "INS", decay=0.001)
    i_n = i_n.merge(tw_feats, on="SK_ID_CURR", how="left")

    for col in ["DPD", "PAYMENT_PERC"]:
        trend = compute_trend(ins, "SK_ID_CURR", col, "DAYS_INSTALMENT", "INS")
        i_n = i_n.merge(trend, on="SK_ID_CURR", how="left")

    ins_sorted = ins.sort_values("DAYS_INSTALMENT", ascending=False)
    for k in [3, 5, 10, 30]:
        last_k = ins_sorted.groupby("SK_ID_CURR").head(k)
        lk_agg = last_k.groupby("SK_ID_CURR").agg(
            **{
                f"INS_LAST{k}_DPD_MEAN": ("DPD", "mean"),
                f"INS_LAST{k}_DPD_MAX": ("DPD", "max"),
                f"INS_LAST{k}_PAYPERC_MEAN": ("PAYMENT_PERC", "mean"),
                f"INS_LAST{k}_PAYDIFF_MEAN": ("PAYMENT_DIFF", "mean"),
                f"INS_LAST{k}_LATE_RATE": ("LATE_PAYMENT", "mean"),
            }
        ).reset_index()
        i_n = i_n.merge(lk_agg, on="SK_ID_CURR", how="left")

    if "NUM_INSTALMENT_VERSION" in ins.columns:
        ver_agg = ins.groupby("SK_ID_CURR")["NUM_INSTALMENT_VERSION"].agg(
            INS_VERSION_NUNIQUE="nunique",
            INS_VERSION_MAX="max",
            INS_VERSION_MEAN="mean",
        ).reset_index()
        i_n = i_n.merge(ver_agg, on="SK_ID_CURR", how="left")

    del ins, loan_agg, ins_sorted
    gc.collect()
    return reduce_memory_usage(i_n)


def sub_model_features_fixed(table_path: Path, target_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = read_csv(table_path)
    for col in df.columns:
        if "DAYS_" in col:
            df[col] = df[col].replace(365243, np.nan)

    feat_cols = [c for c in df.columns if is_numeric_col(df[c]) and c not in ["SK_ID_CURR", "SK_ID_PREV", "SK_ID_BUREAU"]]
    train_ids_set = set(target_df["SK_ID_CURR"].values)
    mask_train = df["SK_ID_CURR"].isin(train_ids_set)
    df_train_rows = df[mask_train].copy()
    df_test_rows = df[~mask_train].copy()
    df_train_rows = df_train_rows.merge(target_df[["SK_ID_CURR", "TARGET"]], on="SK_ID_CURR", how="inner")

    X_train_all = df_train_rows[feat_cols].replace([np.inf, -np.inf], np.nan)
    y_train_all = df_train_rows["TARGET"].astype(int)
    groups_train = df_train_rows["SK_ID_CURR"]

    oof_preds = np.zeros(len(df_train_rows))
    fitted_models: list[lgb.LGBMClassifier] = []
    gkf = GroupKFold(n_splits=5)
    for tr_idx, va_idx in gkf.split(X_train_all, y_train_all, groups=groups_train):
        m = lgb.LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.5,
            reg_alpha=0.1,
            reg_lambda=0.1,
            min_child_samples=100,
            random_state=SEED,
            n_jobs=-1,
            verbosity=-1,
        )
        m.fit(
            X_train_all.iloc[tr_idx],
            y_train_all.iloc[tr_idx],
            eval_set=[(X_train_all.iloc[va_idx], y_train_all.iloc[va_idx])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        oof_preds[va_idx] = m.predict_proba(X_train_all.iloc[va_idx])[:, 1]
        fitted_models.append(m)

    auc = roc_auc_score(y_train_all, oof_preds)
    print(f"  {prefix} sub-model OOF AUC: {auc:.4f}")
    df_train_rows["_SUB_PRED"] = oof_preds

    X_test_rows = df_test_rows[feat_cols].replace([np.inf, -np.inf], np.nan)
    test_preds = np.zeros(len(df_test_rows))
    for m in fitted_models:
        test_preds += m.predict_proba(X_test_rows)[:, 1] / len(fitted_models)
    df_test_rows["_SUB_PRED"] = test_preds

    all_rows = pd.concat(
        [
            df_train_rows[["SK_ID_CURR", "_SUB_PRED"]],
            df_test_rows[["SK_ID_CURR", "_SUB_PRED"]],
        ],
        axis=0,
    )
    sub_agg = all_rows.groupby("SK_ID_CURR")["_SUB_PRED"].agg(
        **{
            f"{prefix}_SUB_MEAN": "mean",
            f"{prefix}_SUB_MAX": "max",
            f"{prefix}_SUB_MIN": "min",
            f"{prefix}_SUB_STD": "std",
        }
    ).reset_index()
    high_risk = (all_rows[all_rows["_SUB_PRED"] > 0.15].groupby("SK_ID_CURR").size().reset_index(name=f"{prefix}_SUB_HIGHRISK"))
    sub_agg = sub_agg.merge(high_risk, on="SK_ID_CURR", how="left")
    sub_agg[f"{prefix}_SUB_HIGHRISK"] = sub_agg[f"{prefix}_SUB_HIGHRISK"].fillna(0)

    del df, df_train_rows, df_test_rows, all_rows
    gc.collect()
    return reduce_memory_usage(sub_agg)


def build_base_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    app_train_raw = read_csv(DATA_DIR / "application_train.csv")
    app_test_raw = read_csv(DATA_DIR / "application_test.csv")
    app_train = application_features(app_train_raw)
    app_test = application_features(app_test_raw)
    print(f"Application: train {app_train.shape}, test {app_test.shape}")
    del app_train_raw, app_test_raw
    gc.collect()

    buro_feat = bureau_and_balance_features()
    print("Bureau:", buro_feat.shape)
    prev_feat = previous_application_features()
    print("Prev:", prev_feat.shape)
    pos_feat = pos_cash_features()
    print("POS:", pos_feat.shape)
    cc_feat = credit_card_features()
    print("CC:", cc_feat.shape)
    ins_feat = installments_features()
    print("Installments:", ins_feat.shape)

    target_df = app_train[["SK_ID_CURR", "TARGET"]].copy()
    print("Training sub-model on previous_application rows...")
    prev_sub = sub_model_features_fixed(DATA_DIR / "previous_application.csv", target_df, "PREV")
    print("Training sub-model on bureau rows...")
    buro_sub = sub_model_features_fixed(DATA_DIR / "bureau.csv", target_df, "BURO")
    print("Training sub-model on installments rows...")
    ins_sub = sub_model_features_fixed(DATA_DIR / "installments_payments.csv", target_df, "INS")
    print("Sub-model features done.")

    feats = [buro_feat, prev_feat, pos_feat, cc_feat, ins_feat, prev_sub, buro_sub, ins_sub]
    train = app_train.copy()
    test = app_test.copy()
    for feat in feats:
        train = train.merge(feat, on="SK_ID_CURR", how="left")
        test = test.merge(feat, on="SK_ID_CURR", how="left")
    print(f"Merged train: {train.shape}, test: {test.shape}")

    del app_train, app_test, buro_feat, prev_feat, pos_feat, cc_feat, ins_feat, prev_sub, buro_sub, ins_sub
    gc.collect()
    return reduce_memory_usage(train), reduce_memory_usage(test)


def add_feature_block(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Adding frequency / groupby / target-encoding features...")
    groupby_cat_cols = [
        "NAME_EDUCATION_TYPE",
        "ORGANIZATION_TYPE",
        "OCCUPATION_TYPE",
        "NAME_INCOME_TYPE",
        "CODE_GENDER",
        "AGE_RANGE",
    ]
    groupby_num_cols = [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "EXT_MEAN",
        "CREDIT_ANNUITY_RATIO",
        "ANNUITY_INCOME_RATIO",
        "DAYS_EMPLOYED_YRS",
    ]

    cat_cols_for_freq = [c for c in train.columns if is_categorical_col(train[c])]
    train, test = add_frequency_features(train, test, cat_cols_for_freq)

    combo_candidates = [
        ("NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE"),
        ("CODE_GENDER", "NAME_FAMILY_STATUS"),
        ("OCCUPATION_TYPE", "ORGANIZATION_TYPE"),
        ("AGE_RANGE", "NAME_EDUCATION_TYPE"),
    ]
    for c1, c2 in combo_candidates:
        if c1 in train.columns and c2 in train.columns:
            new_col = f"{c1}__{c2}"
            train[new_col] = train[c1].astype(str) + "__" + train[c2].astype(str)
            test[new_col] = test[c1].astype(str) + "__" + test[c2].astype(str)

    gp_cat_cols = [c for c in groupby_cat_cols if c in train.columns]
    gp_num_cols = [c for c in groupby_num_cols if c in train.columns]
    train, test = add_groupby_ratio_features(train, test, gp_cat_cols, gp_num_cols)

    te_cols = [
        c
        for c in [
            "NAME_EDUCATION_TYPE",
            "ORGANIZATION_TYPE",
            "OCCUPATION_TYPE",
            "NAME_INCOME_TYPE",
            "CODE_GENDER",
            "NAME_HOUSING_TYPE",
            "AGE_RANGE",
            "NAME_EDUCATION_TYPE__NAME_INCOME_TYPE",
            "CODE_GENDER__NAME_FAMILY_STATUS",
            "OCCUPATION_TYPE__ORGANIZATION_TYPE",
            "AGE_RANGE__NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS",
            "NAME_CONTRACT_TYPE",
        ]
        if c in train.columns
    ]
    train, test = add_target_encoding(
        train,
        test,
        "TARGET",
        te_cols,
        n_splits=N_FOLDS,
        smoothing=TE_SMOOTHING,
        min_samples_leaf=TE_MIN_SAMPLES,
        seed=SEED,
    )
    print(f"After feature block: train {train.shape}, test {test.shape}")
    return reduce_memory_usage(train), reduce_memory_usage(test)


def add_knn_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    target = train["TARGET"].astype("int8")
    print("Computing KNN target features...")
    knn_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "CREDIT_ANNUITY_RATIO"]
    x_knn_all = pd.concat([train[knn_cols], test[knn_cols]], axis=0).fillna(-999).values
    scaler_knn = StandardScaler()
    x_knn_all = scaler_knn.fit_transform(x_knn_all)
    x_knn_tr = x_knn_all[: len(train)]
    x_knn_te = x_knn_all[len(train) :]
    skf_knn = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for k in [200, 500]:
        oof_k = np.zeros(len(train))
        test_k = np.zeros(len(test))
        for tr_idx, va_idx in skf_knn.split(x_knn_tr, target):
            knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean", n_jobs=-1)
            knn.fit(x_knn_tr[tr_idx], target.values[tr_idx])
            oof_k[va_idx] = knn.predict_proba(x_knn_tr[va_idx])[:, 1]
            test_k += knn.predict_proba(x_knn_te)[:, 1] / N_FOLDS
        train[f"KNN_TARGET_{k}"] = oof_k.astype("float32")
        test[f"KNN_TARGET_{k}"] = test_k.astype("float32")
        print(f"  KNN k={k} OOF AUC: {roc_auc_score(target, oof_k):.6f}")

    del x_knn_all, x_knn_tr, x_knn_te
    gc.collect()
    print(f"After KNN: train {train.shape}, test {test.shape}")
    return reduce_memory_usage(train), reduce_memory_usage(test), target


def prepare_matrices(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[int]]:
    train_ids = train["SK_ID_CURR"].copy()
    test_ids = test["SK_ID_CURR"].copy()

    cat_train = train.drop(columns=["SK_ID_CURR"]).copy()
    cat_test = test.drop(columns=["SK_ID_CURR"]).copy()
    cat_feature_names = [c for c in cat_train.columns if c != "TARGET" and is_categorical_col(cat_train[c])]
    for col in cat_feature_names:
        cat_train[col] = cat_train[col].fillna("__nan__").astype(str)
        cat_test[col] = cat_test[col].fillna("__nan__").astype(str)

    target = train["TARGET"].astype("int8").copy()
    train = train.drop(columns=["TARGET", "SK_ID_CURR"])
    test = test.drop(columns=["SK_ID_CURR"])

    obj_cols = [c for c in train.columns if is_categorical_col(train[c])]
    for col in obj_cols:
        le = LabelEncoder()
        all_vals = pd.concat([train[col], test[col]], axis=0).astype(str).fillna("nan")
        le.fit(all_vals)
        train[col] = le.transform(train[col].astype(str).fillna("nan")).astype("int32")
        test[col] = le.transform(test[col].astype(str).fillna("nan")).astype("int32")

    train = train.replace([np.inf, -np.inf], np.nan)
    test = test.replace([np.inf, -np.inf], np.nan)
    cat_train = cat_train.replace([np.inf, -np.inf], np.nan)
    cat_test = cat_test.replace([np.inf, -np.inf], np.nan)
    train, test = train.align(test, join="inner", axis=1)

    cat_train_features = cat_train.drop(columns=["TARGET"]).copy()
    cat_train_features, cat_test = cat_train_features.align(cat_test, join="inner", axis=1)
    cat_train = pd.concat([cat_train[["TARGET"]].reset_index(drop=True), cat_train_features.reset_index(drop=True)], axis=1)
    cat_feature_names = [c for c in cat_feature_names if c in cat_train.columns]
    for col in cat_train.columns:
        if col not in cat_feature_names and col != "TARGET":
            cat_train[col] = pd.to_numeric(cat_train[col], errors="coerce")
            cat_test[col] = pd.to_numeric(cat_test[col], errors="coerce")

    train = clean_names(train)
    test = clean_names(test)
    cat_train = clean_names(cat_train)
    cat_test = clean_names(cat_test)
    cat_feature_names = [re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9_]+", "_", c)).strip("_") for c in cat_feature_names]

    drop_cols = [c for c in train.columns if train[c].isna().all() or train[c].nunique(dropna=False) <= 1]
    if drop_cols:
        train = train.drop(columns=drop_cols)
        test = test.drop(columns=drop_cols)
        print(f"Dropped {len(drop_cols)} zero-variance/all-NaN")
        cat_drop = [c for c in drop_cols if c in cat_train.columns]
        if cat_drop:
            cat_train = cat_train.drop(columns=cat_drop)
            cat_test = cat_test.drop(columns=cat_drop)

    print(f"Before corr filter: {train.shape[1]}")
    sample = train.sample(min(30000, len(train)), random_state=SEED)
    corr = sample.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr = [col for col in upper.columns if any(upper[col] > 0.985)]
    if high_corr:
        train = train.drop(columns=high_corr)
        test = test.drop(columns=high_corr)
        cat_drop = [c for c in high_corr if c in cat_train.columns]
        if cat_drop:
            cat_train = cat_train.drop(columns=cat_drop)
            cat_test = cat_test.drop(columns=cat_drop)
        cat_feature_names = [c for c in cat_feature_names if c in cat_train.columns]
        print(f"Dropped {len(high_corr)} correlated (>0.985)")

    cat_y = cat_train["TARGET"].astype("int8").copy()
    cat_train = cat_train.drop(columns=["TARGET"])
    cat_train, cat_test = cat_train.align(cat_test, join="inner", axis=1)
    cat_feature_names = [c for c in cat_feature_names if c in cat_train.columns]
    cat_feature_indices = [cat_train.columns.get_loc(c) for c in cat_feature_names]

    print(f"CatBoost categorical features: {len(cat_feature_names)}")
    print(f"Final: train {train.shape}, test {test.shape}")
    return train_ids, test_ids, target, cat_y, train, test, cat_feature_names, cat_feature_indices, cat_train, cat_test


def null_importance_selection(train: pd.DataFrame, test: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    print("=== Null-Importance Feature Selection ===")
    print(f"Starting features: {train.shape[1]}")
    feat_names = list(train.columns)
    ni_sample = min(60000, len(train))
    ni_idx = train.sample(ni_sample, random_state=SEED).index
    x_ni = train.loc[ni_idx].reset_index(drop=True)
    y_ni = target.loc[ni_idx].reset_index(drop=True)

    def get_importances(x: pd.DataFrame, y: pd.Series, shuffle: bool = False, seed: int = 0) -> np.ndarray:
        if shuffle:
            y = y.sample(frac=1, random_state=seed).reset_index(drop=True)
        fold_imp = np.zeros(x.shape[1])
        skf_ni = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        for ti, vi in skf_ni.split(x, y):
            m = lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=40,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.3,
                min_child_samples=50,
                random_state=seed,
                n_jobs=-1,
                verbosity=-1,
            )
            m.fit(
                x.iloc[ti],
                y.iloc[ti],
                eval_set=[(x.iloc[vi], y.iloc[vi])],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(20, verbose=False)],
            )
            fold_imp += m.feature_importances_
        return fold_imp / 3

    actual_imp = get_importances(x_ni, y_ni, shuffle=False, seed=SEED)
    n_null_runs = 5
    null_imps = np.zeros((x_ni.shape[1], n_null_runs))
    for i in range(n_null_runs):
        print(f"  Null run {i + 1}/{n_null_runs}...")
        null_imps[:, i] = get_importances(x_ni, y_ni, shuffle=True, seed=100 + i)
    null_80 = np.percentile(null_imps, 80, axis=1)
    score_vs_null = actual_imp / (null_80 + 1)
    drop_null = [feat_names[j] for j in range(len(feat_names)) if score_vs_null[j] < 1.0]
    print(f"  Features below null threshold: {len(drop_null)}")
    if drop_null:
        train = train.drop(columns=drop_null)
        test = test.drop(columns=drop_null)
        feat_names = list(train.columns)
        print(f"  After null-importance pruning: {train.shape[1]} features")
    else:
        print("  No features pruned.")
    del x_ni, y_ni
    gc.collect()
    return train, test, feat_names


def get_feature_subsets(train: pd.DataFrame, target: pd.Series, feat_names: list[str]) -> tuple[list[str], list[str], list[str]]:
    print("Getting feature importances for subset creation...")
    m_imp = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=48,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.3,
        min_child_samples=50,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    skf_imp = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    imp_arr = np.zeros(len(feat_names))
    for ti, vi in skf_imp.split(train, target):
        m_imp.fit(
            train.iloc[ti],
            target.iloc[ti],
            eval_set=[(train.iloc[vi], target.iloc[vi])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        imp_arr += m_imp.feature_importances_
    imp_arr /= 3
    imp_rank = pd.DataFrame({"feature": feat_names, "imp": imp_arr}).sort_values("imp", ascending=False)
    feat_all = feat_names
    feat_top400 = imp_rank.head(min(400, len(feat_names)))["feature"].tolist()
    feat_no_gp = [f for f in feat_names if not f.startswith("GP")]
    print(f"Subset ALL:    {len(feat_all)} features")
    print(f"Subset TOP400: {len(feat_top400)} features")
    print(f"Subset NO_GP:  {len(feat_no_gp)} features")
    return feat_all, feat_top400, feat_no_gp


def train_lgb_a(train: pd.DataFrame, test: pd.DataFrame, target: pd.Series) -> tuple[np.ndarray, np.ndarray, StratifiedKFold]:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_lgb1 = np.zeros(len(train))
    test_lgb1 = np.zeros(len(test))
    lgb_params_a = dict(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        n_estimators=10000,
        learning_rate=0.01,
        num_leaves=48,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.25,
        reg_alpha=0.05,
        reg_lambda=0.1,
        min_child_samples=40,
        min_child_weight=30,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    for fold, (ti, vi) in enumerate(skf.split(train, target), start=1):
        print(f"\n--- LGB_A Fold {fold} ---")
        m = lgb.LGBMClassifier(**lgb_params_a)
        m.fit(
            train.iloc[ti],
            target.iloc[ti],
            eval_set=[(train.iloc[vi], target.iloc[vi])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(200, verbose=True), lgb.log_evaluation(500)],
        )
        oof_lgb1[vi] = m.predict_proba(train.iloc[vi])[:, 1]
        test_lgb1 += m.predict_proba(test)[:, 1] / N_FOLDS
        print(f"Fold {fold} AUC: {roc_auc_score(target.iloc[vi], oof_lgb1[vi]):.6f}")
    print(f"\n=== LGB_A OOF: {roc_auc_score(target, oof_lgb1):.6f} ===")
    return oof_lgb1, test_lgb1, skf


def train_lgb_b(train: pd.DataFrame, test: pd.DataFrame, target: pd.Series, feat_top400: list[str]) -> tuple[np.ndarray, np.ndarray]:
    skf2 = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=123)
    oof_lgb2 = np.zeros(len(train))
    test_lgb2 = np.zeros(len(test))
    train_b = train[feat_top400]
    test_b = test[feat_top400]
    lgb_params_b = dict(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        n_estimators=10000,
        learning_rate=0.008,
        num_leaves=34,
        max_depth=5,
        subsample=0.75,
        colsample_bytree=0.35,
        reg_alpha=0.05,
        reg_lambda=0.2,
        min_child_samples=60,
        min_child_weight=50,
        random_state=123,
        n_jobs=-1,
        verbosity=-1,
    )
    for fold, (ti, vi) in enumerate(skf2.split(train_b, target), start=1):
        print(f"\n--- LGB_B Fold {fold} ---")
        m = lgb.LGBMClassifier(**lgb_params_b)
        m.fit(
            train_b.iloc[ti],
            target.iloc[ti],
            eval_set=[(train_b.iloc[vi], target.iloc[vi])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(200, verbose=True), lgb.log_evaluation(500)],
        )
        oof_lgb2[vi] = m.predict_proba(train_b.iloc[vi])[:, 1]
        test_lgb2 += m.predict_proba(test_b)[:, 1] / N_FOLDS
        print(f"Fold {fold} AUC: {roc_auc_score(target.iloc[vi], oof_lgb2[vi]):.6f}")
    print(f"\n=== LGB_B OOF: {roc_auc_score(target, oof_lgb2):.6f} ===")
    del train_b, test_b
    gc.collect()
    return oof_lgb2, test_lgb2


def train_lgb_seed_avg(train: pd.DataFrame, test: pd.DataFrame, target: pd.Series, feat_no_gp: list[str]) -> tuple[np.ndarray, np.ndarray]:
    oof_lgb3 = np.zeros(len(train))
    test_lgb3 = np.zeros(len(test))
    train_c = train[feat_no_gp]
    test_c = test[feat_no_gp]
    n_seeds = 3
    for seed in [456, 789, 1234]:
        skf3 = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        oof_seed = np.zeros(len(train))
        test_seed = np.zeros(len(test))
        for ti, vi in skf3.split(train_c, target):
            m = lgb.LGBMClassifier(
                objective="binary",
                metric="auc",
                boosting_type="gbdt",
                n_estimators=10000,
                learning_rate=0.01,
                num_leaves=40,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.3,
                reg_alpha=0.1,
                reg_lambda=0.15,
                min_child_samples=50,
                min_child_weight=40,
                random_state=seed,
                n_jobs=-1,
                verbosity=-1,
            )
            m.fit(
                train_c.iloc[ti],
                target.iloc[ti],
                eval_set=[(train_c.iloc[vi], target.iloc[vi])],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(200, verbose=False)],
            )
            oof_seed[vi] = m.predict_proba(train_c.iloc[vi])[:, 1]
            test_seed += m.predict_proba(test_c)[:, 1] / N_FOLDS
        print(f"  Seed {seed} OOF AUC: {roc_auc_score(target, oof_seed):.6f}")
        oof_lgb3 += oof_seed / n_seeds
        test_lgb3 += test_seed / n_seeds
    print(f"\n=== LGB_SEED_AVG OOF: {roc_auc_score(target, oof_lgb3):.6f} ===")
    del train_c, test_c
    gc.collect()
    return oof_lgb3, test_lgb3


def train_xgb_model(train: pd.DataFrame, test: pd.DataFrame, target: pd.Series, skf: StratifiedKFold) -> tuple[np.ndarray, np.ndarray]:
    oof_xgb = np.zeros(len(train))
    test_xgb = np.zeros(len(test))
    xgb_version = tuple(int(x) for x in xgb.__version__.split(".")[:2])
    if xgb_version >= (2, 0):
        xgb_extra = {"device": "cuda", "tree_method": "hist"} if HAS_GPU else {"tree_method": "hist"}
    else:
        xgb_extra = {"tree_method": "gpu_hist"} if HAS_GPU else {"tree_method": "hist"}
    print(f"XGBoost {xgb.__version__}, params: {xgb_extra}")
    for fold, (ti, vi) in enumerate(skf.split(train, target), start=1):
        print(f"\n--- XGB Fold {fold} ---")
        try:
            m = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                n_estimators=10000,
                learning_rate=0.01,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.3,
                reg_alpha=0.1,
                reg_lambda=1.0,
                min_child_weight=40,
                gamma=0.1,
                random_state=SEED,
                n_jobs=-1,
                verbosity=0,
                early_stopping_rounds=200,
                **xgb_extra,
            )
            m.fit(train.iloc[ti], target.iloc[ti], eval_set=[(train.iloc[vi], target.iloc[vi])], verbose=500)
        except xgb.core.XGBoostError:
            m = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                n_estimators=10000,
                learning_rate=0.01,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.3,
                reg_alpha=0.1,
                reg_lambda=1.0,
                min_child_weight=40,
                gamma=0.1,
                random_state=SEED,
                n_jobs=-1,
                verbosity=0,
                early_stopping_rounds=200,
                tree_method="hist",
            )
            m.fit(train.iloc[ti], target.iloc[ti], eval_set=[(train.iloc[vi], target.iloc[vi])], verbose=500)
        oof_xgb[vi] = m.predict_proba(train.iloc[vi])[:, 1]
        test_xgb += m.predict_proba(test)[:, 1] / N_FOLDS
        print(f"Fold {fold} AUC: {roc_auc_score(target.iloc[vi], oof_xgb[vi]):.6f}")
    print(f"\n=== XGB OOF: {roc_auc_score(target, oof_xgb):.6f} ===")
    return oof_xgb, test_xgb


def train_cat_model(
    cat_train: pd.DataFrame,
    cat_test: pd.DataFrame,
    cat_y: pd.Series,
    cat_feature_indices: list[int],
    skf: StratifiedKFold,
) -> tuple[np.ndarray, np.ndarray]:
    oof_cat = np.zeros(len(cat_train))
    test_cat = np.zeros(len(cat_test))
    cat_task = "GPU" if HAS_GPU else "CPU"
    print(f"CatBoost device: {cat_task} | categorical cols: {len(cat_feature_indices)}")
    for fold, (ti, vi) in enumerate(skf.split(cat_train, cat_y), start=1):
        print(f"\n--- CAT Fold {fold} ---")
        cat_params = dict(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=10000,
            learning_rate=0.03,
            depth=7,
            l2_leaf_reg=3.0,
            random_seed=SEED + fold,
            verbose=500,
            early_stopping_rounds=300,
            task_type=cat_task,
            bootstrap_type="Bernoulli",
            subsample=0.8,
            grow_policy="SymmetricTree",
            leaf_estimation_iterations=3,
        )
        if cat_task == "CPU":
            cat_params["rsm"] = 0.3
        try:
            train_pool = Pool(cat_train.iloc[ti], label=cat_y.iloc[ti], cat_features=cat_feature_indices)
            valid_pool = Pool(cat_train.iloc[vi], label=cat_y.iloc[vi], cat_features=cat_feature_indices)
            test_pool = Pool(cat_test, cat_features=cat_feature_indices)
            m = CatBoostClassifier(**cat_params)
            m.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        except Exception:
            cat_params["task_type"] = "CPU"
            cat_params["rsm"] = 0.3
            train_pool = Pool(cat_train.iloc[ti], label=cat_y.iloc[ti], cat_features=cat_feature_indices)
            valid_pool = Pool(cat_train.iloc[vi], label=cat_y.iloc[vi], cat_features=cat_feature_indices)
            test_pool = Pool(cat_test, cat_features=cat_feature_indices)
            m = CatBoostClassifier(**cat_params)
            m.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        oof_cat[vi] = m.predict_proba(valid_pool)[:, 1]
        test_cat += m.predict_proba(test_pool)[:, 1] / N_FOLDS
        print(f"Fold {fold} AUC: {roc_auc_score(cat_y.iloc[vi], oof_cat[vi]):.6f}")
    print(f"\n=== CAT OOF: {roc_auc_score(cat_y, oof_cat):.6f} ===")
    return oof_cat, test_cat


def rank_norm(a: np.ndarray) -> np.ndarray:
    return rankdata(a) / len(a)


def build_ensemble(
    train: pd.DataFrame,
    test: pd.DataFrame,
    test_ids: pd.Series,
    target: pd.Series,
    model_outputs: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, float, str]:
    print("=== Individual Model OOF AUCs ===")
    for name, (oof, _) in model_outputs.items():
        print(f"  {name}: {roc_auc_score(target, oof):.6f}")

    print("\n=== Pairwise Rank Correlations ===")
    names = list(model_outputs.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = np.corrcoef(rank_norm(model_outputs[names[i]][0]), rank_norm(model_outputs[names[j]][0]))[0, 1]
            print(f"  {names[i]:12s} vs {names[j]:12s}: {r:.4f}")

    print("\n=== Level-2 Stacking ===")
    raw_stack_cols = []
    for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "CREDIT_ANNUITY_RATIO", "KNN_TARGET_500", "ANNUITY_INCOME_RATIO"]:
        if c in train.columns:
            raw_stack_cols.append(c)
    oof_stack = np.column_stack([model_outputs[n][0] for n in names])
    test_stack = np.column_stack([model_outputs[n][1] for n in names])
    if raw_stack_cols:
        raw_tr = train[raw_stack_cols].fillna(-999).values.astype("float32")
        raw_te = test[raw_stack_cols].fillna(-999).values.astype("float32")
        oof_stack = np.column_stack([oof_stack, raw_tr])
        test_stack = np.column_stack([test_stack, raw_te])

    oof_stack_lr = np.zeros(len(train))
    test_stack_lr = np.zeros(len(test))
    skf_stack = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=789)
    for ti, vi in skf_stack.split(oof_stack, target):
        lr = LogisticRegression(C=0.35, max_iter=2000, solver="lbfgs", random_state=789)
        lr.fit(oof_stack[ti], target.values[ti])
        oof_stack_lr[vi] = lr.predict_proba(oof_stack[vi])[:, 1]
        test_stack_lr += lr.predict_proba(test_stack)[:, 1] / N_FOLDS
    stack_lr_auc = roc_auc_score(target, oof_stack_lr)
    print(f"  Logistic stacked OOF AUC: {stack_lr_auc:.6f}")

    print("\n=== Constrained Rank-Blending ===")
    blend_oof = [rank_norm(model_outputs[n][0]) for n in names] + [rank_norm(oof_stack_lr)]
    blend_test = [rank_norm(model_outputs[n][1]) for n in names] + [rank_norm(test_stack_lr)]
    blend_names = names + ["stack_lr"]

    def neg_auc(w: np.ndarray) -> float:
        w = np.clip(w, 0, 1)
        w = w / (w.sum() + 1e-12)
        blend = sum(wi * ri for wi, ri in zip(w, blend_oof))
        return -roc_auc_score(target, blend)

    n_models = len(blend_names)
    w0 = np.ones(n_models) / n_models
    bounds = [(0.0, 0.60)] * n_models
    cons = ({"type": "eq", "fun": lambda w: np.sum(np.clip(w, 0, 1)) - 1.0},)

    best_result = None
    for method in ["SLSQP", "Powell"]:
        try:
            result = minimize(
                neg_auc,
                w0,
                method=method,
                bounds=bounds,
                constraints=cons if method == "SLSQP" else (),
                options={"maxiter": 3000, "ftol": 1e-10},
            )
            if best_result is None or result.fun < best_result.fun:
                best_result = result
        except Exception as exc:
            print(f"{method} failed: {exc}")

    if best_result is None:
        raise RuntimeError("Blend optimization failed.")

    best_w = np.clip(best_result.x, 0, 1)
    best_w = best_w / (best_w.sum() + 1e-12)
    blend_auc = -best_result.fun
    print("Optimized weights:")
    for name, w in zip(blend_names, best_w):
        print(f"  {name:12s}: {w:.4f}")
    print(f"Optimized OOF AUC: {blend_auc:.6f}")

    test_pred_blend = sum(w * r for w, r in zip(best_w, blend_test))
    candidates = {
        "blend": (blend_auc, test_pred_blend),
        "stack_lr": (stack_lr_auc, test_stack_lr),
    }
    best_name = max(candidates, key=lambda k: candidates[k][0])
    final_auc = candidates[best_name][0]
    test_pred = candidates[best_name][1]
    print(f"\nUsing ensemble source: {best_name} with OOF AUC {final_auc:.6f}")

    submission = pd.DataFrame({"SK_ID_CURR": test_ids.astype(int), "TARGET": test_pred})
    return submission, final_auc, best_name


def main() -> None:
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"GPU available: {HAS_GPU}")

    train, test = build_base_tables()
    train, test = add_feature_block(train, test)
    train, test, target = add_knn_features(train, test)

    (
        _train_ids,
        test_ids,
        target,
        cat_y,
        train_num,
        test_num,
        _cat_feature_names,
        cat_feature_indices,
        cat_train,
        cat_test,
    ) = prepare_matrices(train, test)

    train_num, test_num, feat_names = null_importance_selection(train_num, test_num, target)

    cat_keep = [c for c in train_num.columns if c in cat_train.columns]
    cat_train = cat_train[cat_keep].copy()
    cat_test = cat_test[cat_keep].copy()
    cat_feature_indices = [idx for idx, col in enumerate(cat_train.columns) if is_categorical_col(cat_train[col])]

    feat_all, feat_top400, feat_no_gp = get_feature_subsets(train_num, target, feat_names)
    train_num = train_num[feat_all].copy()
    test_num = test_num[feat_all].copy()

    oof_lgb1, test_lgb1, skf = train_lgb_a(train_num, test_num, target)
    oof_lgb2, test_lgb2 = train_lgb_b(train_num, test_num, target, feat_top400)
    oof_lgb3, test_lgb3 = train_lgb_seed_avg(train_num, test_num, target, feat_no_gp)
    oof_xgb, test_xgb = train_xgb_model(train_num, test_num, target, skf)
    oof_cat, test_cat = train_cat_model(cat_train, cat_test, cat_y, cat_feature_indices, skf)

    outputs = {
        "lgb_a": (oof_lgb1, test_lgb1),
        "lgb_b": (oof_lgb2, test_lgb2),
        "lgb_seed": (oof_lgb3, test_lgb3),
        "xgb": (oof_xgb, test_xgb),
        "cat": (oof_cat, test_cat),
    }
    submission, final_auc, best_name = build_ensemble(train_num, test_num, test_ids, target, outputs)
    out_path = Path(__file__).resolve().parent / "submission6.csv"
    submission.to_csv(out_path, index=False)
    print(f"\n=== FINAL OOF AUC (selection metric): {final_auc:.6f} ===")
    print(f"Using ensemble source: {best_name}")
    print(f"Saved submission to {out_path}")
    print(submission.head())


if __name__ == "__main__":
    main()
