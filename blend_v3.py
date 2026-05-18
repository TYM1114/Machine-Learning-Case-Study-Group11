from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PREFERRED_FILES = [
    ("sub6", "submission6.csv"),
    ("sub4", "submission4.csv"),
    ("sub3", "submission3.csv"),
    ("sub1", "submission1_v2.csv"),
    ("sub_blend", "submission_blend.csv"),
    ("sub_blend_v2_raw", "submission_blend_v2_raw.csv"),
    ("sub_blend_v2_d", "submission_blend_v2_d.csv"),
    ("sub5", "submission5.csv"),
    ("sub2", "submission2.csv"),
    ("sub0", "submission.csv"),
]


def rank_normalize(series: pd.Series) -> pd.Series:
    return series.rank(method="average", pct=True)


def discover_files() -> dict[str, Path]:
    search_roots = [BASE_DIR, BASE_DIR / "sub"]
    found: dict[str, Path] = {}

    for key, filename in PREFERRED_FILES:
        for root in search_roots:
            path = root / filename
            if path.exists():
                found[key] = path
                break

    discovered = {path.name: path for path in BASE_DIR.rglob("submission*.csv")}
    for key, filename in PREFERRED_FILES:
        if key not in found and filename in discovered:
            found[key] = discovered[filename]

    if "sub6" not in found:
        available = sorted(p.name for p in discovered.values())
        raise FileNotFoundError(
            "submission6.csv not found. "
            f"Available submission files: {available if available else 'none'}"
        )
    if len(found) < 2:
        raise FileNotFoundError("Need at least one additional submission file besides submission6.csv to blend.")
    return found


def load_submissions(files: dict[str, Path]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for name, path in files.items():
        df = pd.read_csv(path).rename(columns={"TARGET": name})
        cols = ["SK_ID_CURR", name]
        if merged is None:
            merged = df[cols].copy()
        else:
            merged = merged.merge(df[cols], on="SK_ID_CURR", how="inner")
    if merged is None:
        raise ValueError("No submissions loaded.")
    return merged


def correlation_report(merged: pd.DataFrame) -> pd.DataFrame:
    pred_cols = [c for c in merged.columns if c != "SK_ID_CURR"]
    corr = merged[pred_cols].corr().round(6)
    print("Prediction correlation:")
    print(corr.to_string())
    if "sub6" in corr.columns:
        sub6_corr = corr["sub6"].drop("sub6").sort_values()
        print("\nLowest-correlation helpers vs sub6:")
        print(sub6_corr.to_string())
    return corr


def save_blend(merged: pd.DataFrame, name: str, weights: dict[str, float], use_rank: bool) -> None:
    usable = {k: v for k, v in weights.items() if k in merged.columns}
    total = sum(usable.values())
    if total <= 0:
        print(f"Skip {name}: no usable components.")
        return
    usable = {k: v / total for k, v in usable.items()}

    pred = pd.Series(0.0, index=merged.index)
    for key, weight in usable.items():
        values = rank_normalize(merged[key]) if use_rank else merged[key]
        pred += values * weight

    output = merged[["SK_ID_CURR"]].copy()
    output["TARGET"] = pred
    output_dir = BASE_DIR / "sub" if (BASE_DIR / "sub").exists() else BASE_DIR
    output_path = output_dir / f"{name}.csv"
    output.to_csv(output_path, index=False)
    mode = "rank" if use_rank else "raw"
    print(f"Saved {output_path.name} | mode={mode} | weights={usable}")


def build_candidates(files: dict[str, Path]) -> list[tuple[str, dict[str, float], bool]]:
    candidates: list[tuple[str, dict[str, float], bool]] = [
        ("submission_blend_v3_a", {"sub6": 0.90, "sub_blend": 0.10}, True),
        ("submission_blend_v3_b", {"sub6": 0.80, "sub_blend": 0.10, "sub1": 0.10}, True),
        ("submission_blend_v3_c", {"sub6": 0.75, "sub4": 0.15, "sub_blend": 0.10}, False),
        ("submission_blend_v3_d", {"sub6": 0.70, "sub_blend_v2_raw": 0.20, "sub1": 0.10}, False),
        ("submission_blend_v3_e", {"sub6": 0.70, "sub_blend_v2_d": 0.20, "sub_blend": 0.10}, True),
        ("submission_blend_v3_f", {"sub6": 0.65, "sub4": 0.15, "sub3": 0.10, "sub_blend": 0.10}, False),
    ]

    available = set(files.keys())
    valid_candidates: list[tuple[str, dict[str, float], bool]] = []
    for name, weights, use_rank in candidates:
        if "sub6" not in weights:
            continue
        if any(key in available for key in weights if key != "sub6"):
            valid_candidates.append((name, weights, use_rank))
    return valid_candidates


def main() -> None:
    files = discover_files()
    print("Loaded files:")
    for key, path in files.items():
        print(f"  {key}: {path.name}")

    merged = load_submissions(files)
    correlation_report(merged)

    candidates = build_candidates(files)
    if not candidates:
        raise RuntimeError("No valid blend candidates after file discovery.")

    print("\nGenerating candidates:")
    for name, weights, use_rank in candidates:
        save_blend(merged, name, weights, use_rank)


if __name__ == "__main__":
    main()
