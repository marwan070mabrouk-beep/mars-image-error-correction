from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "results" / "final_bsc_metrics_restartable.csv"
SUMMARY = ROOT / "results" / "final_results_summary.csv"
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

METHODS = ["uncoded", "hamming", "ldpc"]
LABELS = {"uncoded": "Uncoded", "hamming": "Hamming (7,4)", "ldpc": "LDPC"}
COLORS = {"uncoded": "#555555", "hamming": "#d95f02", "ldpc": "#1b9e77"}
BOOTSTRAP_SEED = 20261126


def read_rows() -> list[dict[str, str]]:
    with INPUT.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def floats(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, reps: int = 2000) -> tuple[float, float]:
    if values.size <= 1 or np.all(values == values[0]):
        return float(values[0]), float(values[0])
    indices = rng.integers(0, values.size, size=(reps, values.size))
    means = values[indices].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def pooled_psnr(mse: np.ndarray) -> float:
    mean_mse = float(np.mean(mse))
    return float("inf") if mean_mse == 0 else float(10 * np.log10(255.0**2 / mean_mse))


def bootstrap_pooled_psnr(mse: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if mse.size <= 1:
        value = pooled_psnr(mse)
        return value, value
    indices = rng.integers(0, mse.size, size=(2000, mse.size))
    means = mse[indices].mean(axis=1)
    values = np.where(means == 0, np.inf, 10 * np.log10(255.0**2 / means))
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("inf"), float("inf")
    return float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    result = []
    p_values = sorted({float(row["p"]) for row in rows})
    for p in p_values:
        for method in METHODS:
            group = [row for row in rows if float(row["p"]) == p and row["method"] == method]
            raw = floats(group, "raw_ber")
            decoded = floats(group, "decoded_ber")
            fer = floats(group, "fer")
            ssim = floats(group, "ssim")
            mse = floats(group, "mse")
            time_s = floats(group, "decoder_time_seconds")
            row = {
                "method": method,
                "p": p,
                "trials": len(group),
                "source_bits_per_trial": int(group[0]["source_bits"]),
                "transmitted_bits": int(group[0]["transmitted_bits"]),
                "native_k": int(group[0]["native_k"]),
                "nominal_code_rate": int(group[0]["native_k"]) / (int(group[0]["transmitted_bits"]) / int(group[0]["blocks"])),
                "effective_payload_rate": int(group[0]["source_bits"]) / int(group[0]["transmitted_bits"]),
                "raw_ber_pooled": float(np.sum([int(x["channel_flips"]) for x in group]) / np.sum([int(x["transmitted_bits"]) for x in group])),
                "decoded_ber_pooled": float(np.sum([int(x["decoded_bit_errors"]) for x in group]) / np.sum([int(x["source_bits"]) for x in group])),
                "decoded_ber_ci_low": bootstrap_mean_ci(decoded, rng)[0],
                "decoded_ber_ci_high": bootstrap_mean_ci(decoded, rng)[1],
                "fer_mean": float(np.mean(fer)),
                "fer_ci_low": bootstrap_mean_ci(fer, rng)[0],
                "fer_ci_high": bootstrap_mean_ci(fer, rng)[1],
                "mse_mean": float(np.mean(mse)),
                "psnr_pooled_db": pooled_psnr(mse),
                "psnr_ci_low_db": bootstrap_pooled_psnr(mse, rng)[0],
                "psnr_ci_high_db": bootstrap_pooled_psnr(mse, rng)[1],
                "ssim_mean": float(np.mean(ssim)),
                "ssim_ci_low": bootstrap_mean_ci(ssim, rng)[0],
                "ssim_ci_high": bootstrap_mean_ci(ssim, rng)[1],
                "exact_images": int(np.sum([x["exact_image"].lower() == "true" for x in group])),
                "nonconverged_blocks_mean": float(np.mean(floats(group, "ldpc_nonconverged_blocks"))),
                "iterations_mean": float(np.mean(floats(group, "ldpc_total_iterations"))),
                "decoder_time_mean_s": float(np.mean(time_s)),
                "decoder_time_median_s": float(np.median(time_s)),
            }
            result.append(row)
    return result


def write_summary(summary: list[dict[str, object]]) -> None:
    with SUMMARY.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)


def plot_metric(summary: list[dict[str, object]], metric: str, ylabel: str, filename: str, logy: bool = False, ylim=None) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=180)
    for method in METHODS:
        rows = [r for r in summary if r["method"] == method and float(r["p"]) > 0]
        x = np.asarray([float(r["p"]) for r in rows])
        y = np.asarray([float(r[metric]) for r in rows])
        if logy:
            y = np.where(y > 0, y, np.nan)
        ax.plot(x, y, marker="o", linewidth=2, label=LABELS[method], color=COLORS[method])
        if logy:
            for xx, rr in zip(x, rows):
                if float(rr[metric]) == 0:
                    ax.annotate("0", (xx, 1e-7), textcoords="offset points", xytext=(0, 5), ha="center", color=COLORS[method], fontsize=9)
    ax.set_xlabel("BSC crossover probability p")
    ax.set_ylabel(ylabel)
    ax.set_xticks([0.001, 0.005, 0.01, 0.02, 0.05])
    ax.grid(True, alpha=0.25)
    if logy:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def plot_psnr(summary: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=180)
    for method in METHODS:
        rows = [r for r in summary if r["method"] == method and float(r["p"]) > 0]
        x = np.asarray([float(r["p"]) for r in rows])
        y = np.asarray([min(float(r["psnr_pooled_db"]), 60.0) for r in rows])
        ax.plot(x, y, marker="o", linewidth=2, label=LABELS[method], color=COLORS[method])
        for xx, rr in zip(x, rows):
            if math.isinf(float(rr["psnr_pooled_db"])):
                ax.annotate("∞", (xx, 60.0), textcoords="offset points", xytext=(0, 7), ha="center", color=COLORS[method], fontsize=12)
    ax.set_xlabel("BSC crossover probability p")
    ax.set_ylabel("Pooled-MSE PSNR (dB; ∞ shown at 60 dB cap)")
    ax.set_xticks([0.001, 0.005, 0.01, 0.02, 0.05])
    ax.set_ylim(0, 62)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "psnr_vs_p.png", bbox_inches="tight")
    plt.close(fig)


def plot_raw_decoded_ber(summary: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=180)
    for method in METHODS:
        rows = [r for r in summary if r["method"] == method and float(r["p"]) > 0]
        x = np.asarray([float(r["p"]) for r in rows])
        raw = np.asarray([float(r["raw_ber_pooled"]) for r in rows])
        decoded = np.asarray([float(r["decoded_ber_pooled"]) for r in rows])
        ax.plot(x, raw, marker="o", linewidth=1.8, linestyle="--", color=COLORS[method], alpha=0.75, label=f"{LABELS[method]} raw")
        ax.plot(x, np.where(decoded > 0, decoded, np.nan), marker="o", linewidth=2.2, color=COLORS[method], label=f"{LABELS[method]} decoded")
        for xx, yy in zip(x, decoded):
            if yy == 0:
                ax.annotate("0", (xx, 1e-7), textcoords="offset points", xytext=(0, 5), ha="center", color=COLORS[method], fontsize=8)
    ax.set_xlabel("BSC crossover probability p")
    ax.set_ylabel("BER")
    ax.set_yscale("log")
    ax.set_xticks([0.001, 0.005, 0.01, 0.02, 0.05])
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "raw_decoded_ber_vs_p.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = read_rows()
    summary = summarize(rows)
    write_summary(summary)
    plot_metric(summary, "decoded_ber_pooled", "Decoded source BER", "decoded_ber_vs_p.png", logy=True)
    plot_raw_decoded_ber(summary)
    plot_metric(summary, "fer_mean", "Frame error rate (FER)", "fer_vs_p.png", ylim=(0, 1.05))
    plot_psnr(summary)
    plot_metric(summary, "ssim_mean", "Mean SSIM", "ssim_vs_p.png", ylim=(0, 1.02))
    plot_metric(summary, "decoder_time_mean_s", "Mean decoder time per image (s)", "decoder_time_vs_p.png")
    print(f"Read {len(rows)} raw rows")
    print(f"Wrote {SUMMARY}")
    print(f"Wrote figures under {FIG_DIR}")
    for row in summary:
        if float(row["p"]) in (0.001, 0.01, 0.05):
            print(row["p"], row["method"], "BER", row["decoded_ber_pooled"], "FER", row["fer_mean"], "PSNR", row["psnr_pooled_db"], "SSIM", row["ssim_mean"])


if __name__ == "__main__":
    main()
