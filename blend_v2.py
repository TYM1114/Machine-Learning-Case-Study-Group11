from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
FILES = {
    "sub1": BASE_DIR / "submission1_v2.csv",
    "sub3": BASE_DIR / "submission3.csv",
    "sub4": BASE_DIR / "submission4.csv",
    "sub_blend": BASE_DIR / "submission_blend.csv",
}


def rank_normalize(series: pd.Series) -> pd.Series:
    return series.rank(method="average", pct=True)


def load_submissions() -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for name, path in FILES.items():
        df = pd.read_csv(path).rename(columns={"TARGET": name})
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on="SK_ID_CURR", how="inner")

    if merged is None:
        raise ValueError("No submissions loaded.")
    return merged


def save_blend(merged: pd.DataFrame, name: str, weights: dict[str, float], use_rank: bool) -> None:
    pred = pd.Series(0.0, index=merged.index)
    for key, weight in weights.items():
        values = rank_normalize(merged[key]) if use_rank else merged[key]
        pred += values * weight

    output = merged[["SK_ID_CURR"]].copy()
    output["TARGET"] = pred
    output_path = BASE_DIR / f"{name}.csv"
    output.to_csv(output_path, index=False)
    mode = "rank" if use_rank else "raw"
    print(f"Saved {output_path.name} | mode={mode} | weights={weights}")


def main() -> None:
    merged = load_submissions()
    corr = merged.drop(columns=["SK_ID_CURR"]).corr().round(6)
    print("Prediction correlation:")
    print(corr.to_string())

    candidates = [
        ("submission_blend_v2_a", {"sub4": 0.85, "sub_blend": 0.15}, True),
        ("submission_blend_v2_b", {"sub4": 0.70, "sub_blend": 0.15, "sub1": 0.15}, True),
        ("submission_blend_v2_c", {"sub4": 0.60, "sub3": 0.25, "sub_blend": 0.15}, True),
        ("submission_blend_v2_d", {"sub4": 0.55, "sub1": 0.20, "sub3": 0.15, "sub_blend": 0.10}, True),
        ("submission_blend_v2_raw", {"sub4": 0.70, "sub3": 0.20, "sub1": 0.10}, False),
    ]

    for name, weights, use_rank in candidates:
        save_blend(merged, name, weights, use_rank)


if __name__ == "__main__":
    main()
