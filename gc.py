import argparse
import os
import re

import matplotlib.pyplot as plt
import pandas as pd


def parse_alpha_from_folder(folder_name: str):
    """Parse folder names such as alpha_01, alpha_003, or alpha_1."""
    match = re.match(r"^alpha_(\d+)$", folder_name)
    if not match:
        return None

    digits = match.group(1)
    if digits.startswith("0"):
        frac = digits[1:]
        return 0.0 if frac == "" else float("0." + frac)
    return float(digits)


def extract_accuracy(csv_path: str, mode: str = "last"):
    """Extract the accuracy column from a CSV file."""
    df = pd.read_csv(csv_path)
    if "accuracy" not in df.columns:
        raise ValueError(f"{csv_path} does not contain an accuracy column; columns={list(df.columns)}")

    values = df["accuracy"].dropna()
    if len(values) == 0:
        raise ValueError(f"{csv_path} has an empty accuracy column")

    if mode == "mean":
        return float(values.mean())
    return float(values.iloc[-1])


def main():
    parser = argparse.ArgumentParser(
        description="Summarize ACC/ASR over alpha_* folders and plot comparison figures."
    )
    parser.add_argument("--root", type=str, default=".", help="Root directory containing alpha_* folders.")
    parser.add_argument("--out", type=str, default="alpha_plots", help="Output directory.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["last", "mean"],
        default="last",
        help="How to read accuracy from each CSV: last row or mean value.",
    )
    parser.add_argument(
        "--save_single_col",
        action="store_true",
        help="Also save extracted single-column accuracy CSV files for checking.",
    )
    args = parser.parse_args()

    root_dir = os.path.abspath(args.root)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, name)
        if not os.path.isdir(folder_path):
            continue

        alpha_val = parse_alpha_from_folder(name)
        if alpha_val is None:
            continue

        acc_csv = os.path.join(folder_path, "test_result.csv")
        asr_csv = os.path.join(folder_path, "posiontest_result.csv")

        if not os.path.exists(acc_csv):
            print(f"[skip] {name}: missing test_result.csv")
            continue
        if not os.path.exists(asr_csv):
            print(f"[skip] {name}: missing posiontest_result.csv")
            continue

        try:
            acc = extract_accuracy(acc_csv, mode=args.mode)
            asr = extract_accuracy(asr_csv, mode=args.mode)
        except Exception as exc:
            print(f"[error] failed to process {name}: {exc}")
            continue

        rows.append({"folder": name, "alpha": alpha_val, "acc": acc, "asr": asr})

        if args.save_single_col:
            df_acc = pd.read_csv(acc_csv)
            df_acc[["accuracy"]].to_csv(
                os.path.join(out_dir, f"{name}_test_result_acc_only.csv"), index=False
            )

            df_asr = pd.read_csv(asr_csv)
            df_asr[["accuracy"]].to_csv(
                os.path.join(out_dir, f"{name}_posiontest_result_asr_only.csv"), index=False
            )

    if not rows:
        print("[done] no usable alpha_* folders or CSV files were found")
        return

    summary = pd.DataFrame(rows).sort_values(by="alpha", ascending=True).reset_index(drop=True)

    summary_path = os.path.join(out_dir, "alpha_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"[save] summary: {summary_path}")

    plt.figure()
    plt.plot(summary["alpha"], summary["acc"], marker="o")
    plt.xlabel("alpha")
    plt.ylabel("ACC (test_result.csv accuracy)")
    plt.title("ACC vs alpha")
    plt.grid(True, linestyle="--", linewidth=0.5)
    acc_fig = os.path.join(out_dir, "acc_vs_alpha.png")
    plt.savefig(acc_fig, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[save] figure: {acc_fig}")

    plt.figure()
    plt.plot(summary["alpha"], summary["asr"], marker="o")
    plt.xlabel("alpha")
    plt.ylabel("ASR (posiontest_result.csv accuracy)")
    plt.title("ASR vs alpha")
    plt.grid(True, linestyle="--", linewidth=0.5)
    asr_fig = os.path.join(out_dir, "asr_vs_alpha.png")
    plt.savefig(asr_fig, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[save] figure: {asr_fig}")

    print("\n===== Summary (sorted by alpha) =====")
    print(summary[["alpha", "acc", "asr", "folder"]].to_string(index=False))


if __name__ == "__main__":
    main()
