from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SUBMISSION_1 = BASE_DIR / "submission.csv"
SUBMISSION_2 = BASE_DIR / "submission2.csv"
OUTPUT_PATH = BASE_DIR / "submission_blend.csv"

# Keep the stronger model as the anchor and blend a smaller share of the weaker one.
WEIGHT_SUBMISSION_1 = 0.85
WEIGHT_SUBMISSION_2 = 0.15
USE_RANK_BLEND = True


def rank_normalize(series: pd.Series) -> pd.Series:
    return series.rank(method="average", pct=True)


def main() -> None:
    sub1 = pd.read_csv(SUBMISSION_1)
    sub2 = pd.read_csv(SUBMISSION_2)

    merged = sub1.merge(sub2, on="SK_ID_CURR", suffixes=("_1", "_2"), how="inner")
    if len(merged) != len(sub1) or len(merged) != len(sub2):
        raise ValueError("Submission files do not align on SK_ID_CURR.")

    if USE_RANK_BLEND:
        pred1 = rank_normalize(merged["TARGET_1"])
        pred2 = rank_normalize(merged["TARGET_2"])
    else:
        pred1 = merged["TARGET_1"]
        pred2 = merged["TARGET_2"]

    merged["TARGET"] = pred1 * WEIGHT_SUBMISSION_1 + pred2 * WEIGHT_SUBMISSION_2
    output = merged[["SK_ID_CURR", "TARGET"]]
    output.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved blended submission to {OUTPUT_PATH}")
    print(f"Blend mode: {'rank' if USE_RANK_BLEND else 'raw'}")
    print(f"Weights: submission.csv={WEIGHT_SUBMISSION_1}, submission2.csv={WEIGHT_SUBMISSION_2}")


if __name__ == "__main__":
    main()
