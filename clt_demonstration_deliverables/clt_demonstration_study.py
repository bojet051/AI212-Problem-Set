
"""
Central Limit Theorem Demonstration
Author: Ely Jun-Pates
Course: AI212 / Statistical Inference

Purpose:
    This script demonstrates the Central Limit Theorem using repeated samples
    from three symmetric distributions:

        1. Uniform(-sqrt(3), sqrt(3))
        2. Normal(0, 1)
        3. Student's t distribution with df = 3

    For each distribution, it repeatedly samples observations, computes:
        - sample mean
        - sample median
        - trimmed mean

    It studies how the distributions of these estimators change as sample size
    increases. It also creates Q-Q plots and computes skewness and kurtosis.

Important:
    The t distribution with df = 3 is symmetric and heavy-tailed. Its variance
    exists but is large. This makes it useful for showing that the sample mean
    is sensitive to heavy-tailed observations.

How to run:
    python3 clt_demonstration_study.py

Required packages:
    numpy
    matplotlib

No scipy dependency is required.
"""

import math
import os
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# Configure Matplotlib before importing pyplot so the script works in headless
# or restricted environments where the user's home cache is not writable.
MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "clt_demonstration_mpl_config"
XDG_CACHE_DIR = Path(tempfile.gettempdir()) / "clt_demonstration_xdg_cache"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# 1. Reproducibility and output setup
# ---------------------------------------------------------------------
SEED = 212
rng = np.random.Generator(np.random.MT19937(seed=SEED))

OUTPUT_DIR = SCRIPT_DIR / "clt_output"
OUTPUT_DIR.mkdir(exist_ok=True)
SUMMARY_PATH = SCRIPT_DIR / "clt_simulation_summary.txt"

# Number of repeated experiments per sample size.
REPLICATIONS = 20_000

# Sample sizes to show convergence.
SAMPLE_SIZES = [5, 10, 30, 100]


# ---------------------------------------------------------------------
# 2. Basic statistical helper functions
# ---------------------------------------------------------------------
def sample_skewness(x):
    """
    Moment-based sample skewness:
        skew = m3 / m2^(3/2)

    For a perfectly symmetric distribution, skewness is 0.
    """
    x = np.asarray(x)
    mean = np.mean(x)
    centered = x - mean
    m2 = np.mean(centered**2)
    m3 = np.mean(centered**3)

    if m2 == 0:
        return 0.0

    return float(m3 / (m2 ** 1.5))


def excess_kurtosis(x):
    """
    Moment-based excess kurtosis:
        excess kurtosis = m4 / m2^2 - 3

    A normal distribution has excess kurtosis approximately 0.
    Positive excess kurtosis means heavier tails than normal.
    Negative excess kurtosis means lighter tails than normal.
    """
    x = np.asarray(x)
    mean = np.mean(x)
    centered = x - mean
    m2 = np.mean(centered**2)
    m4 = np.mean(centered**4)

    if m2 == 0:
        return -3.0

    return float(m4 / (m2**2) - 3.0)


def normal_quantile_approx(p):
    """
    Approximate inverse standard normal CDF using the Acklam approximation.

    This is used for Q-Q plots without requiring scipy.
    Source of method: Peter John Acklam's inverse normal approximation.
    """
    # Coefficients in rational approximations.
    a = [
        -3.969683028665376e+01,
         2.209460984245205e+02,
        -2.759285104469687e+02,
         1.383577518672690e+02,
        -3.066479806614716e+01,
         2.506628277459239e+00,
    ]
    b = [
        -5.447609879822406e+01,
         1.615858368580409e+02,
        -1.556989798598866e+02,
         6.680131188771972e+01,
        -1.328068155288572e+01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
         4.374664141464968e+00,
         2.938163982698783e+00,
    ]
    d = [
         7.784695709041462e-03,
         3.224671290700398e-01,
         2.445134137142996e+00,
         3.754408661907416e+00,
    ]

    p = np.asarray(p)
    out = np.zeros_like(p, dtype=float)

    p_low = 0.02425
    p_high = 1.0 - p_low

    low = p < p_low
    high = p > p_high
    mid = (~low) & (~high)

    if np.any(low):
        q = np.sqrt(-2.0 * np.log(p[low]))
        out[low] = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / (
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        )

    if np.any(mid):
        q = p[mid] - 0.5
        r = q*q
        out[mid] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (
            (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
        )

    if np.any(high):
        q = np.sqrt(-2.0 * np.log(1.0 - p[high]))
        out[high] = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / (
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        )

    return out


def trimmed_mean(data, proportion=0.10):
    """
    Compute a 10% trimmed mean row-wise.

    For each repeated sample, remove the lowest and highest 10%
    of values, then compute the mean of the remaining values.
    """
    data_sorted = np.sort(data, axis=1)
    n = data.shape[1]
    trim = int(math.floor(proportion * n))

    if trim == 0:
        return np.mean(data_sorted, axis=1)

    return np.mean(data_sorted[:, trim:n-trim], axis=1)


def summarize_estimator(values):
    """Return skewness, excess kurtosis, mean, variance, and standard deviation."""
    return {
        "mean": float(np.mean(values)),
        "variance": float(np.var(values, ddof=1)),
        "std": float(np.std(values, ddof=1)),
        "skewness": sample_skewness(values),
        "excess_kurtosis": excess_kurtosis(values),
    }


# ---------------------------------------------------------------------
# 3. Distribution generators
# ---------------------------------------------------------------------
def generate_samples(distribution_name, n, replications, rng):
    """
    Generate repeated samples from the selected distribution.
    Output shape is (replications, n).
    """
    if distribution_name == "Uniform":
        # Uniform(-sqrt(3), sqrt(3)) has mean 0 and variance 1.
        a = -math.sqrt(3)
        b = math.sqrt(3)
        return rng.uniform(a, b, size=(replications, n))

    if distribution_name == "Normal":
        # Standard normal distribution: mean 0, variance 1.
        return rng.normal(0.0, 1.0, size=(replications, n))

    if distribution_name == "t_df3":
        # Student's t distribution with df=3.
        # It is symmetric, heavy-tailed, mean 0, variance df/(df-2)=3.
        return rng.standard_t(df=3, size=(replications, n))

    raise ValueError(f"Unknown distribution: {distribution_name}")


# ---------------------------------------------------------------------
# 4. Plot functions
# ---------------------------------------------------------------------
def plot_estimator_histograms(results, distribution_label, n):
    """
    Plot histograms of sample mean, sample median, and trimmed mean
    for a given source distribution and sample size.
    """
    plt.figure(figsize=(9, 5))
    bins = 80

    plt.hist(results["mean"], bins=bins, density=True, alpha=0.45, label="Sample mean")
    plt.hist(results["median"], bins=bins, density=True, alpha=0.45, label="Sample median")
    plt.hist(results["trimmed"], bins=bins, density=True, alpha=0.45, label="10% trimmed mean")

    plt.title(f"Sampling Distributions of Estimators\n{distribution_label}, sample size n={n}")
    plt.xlabel("Estimator value")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    filename = f"hist_{distribution_label.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')}_n{n}.png"
    path = OUTPUT_DIR / filename
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def qq_plot(values, title, filename):
    """
    Create a normal Q-Q plot.

    If values are approximately normal, the points should lie close to
    the diagonal reference line.
    """
    values = np.sort(np.asarray(values))
    m = len(values)

    # Plotting positions.
    probs = (np.arange(1, m + 1) - 0.5) / m
    theoretical = normal_quantile_approx(probs)

    # Standardize sample values before comparing to N(0,1) quantiles.
    standardized = (values - np.mean(values)) / np.std(values, ddof=1)

    plt.figure(figsize=(6, 6))
    plt.scatter(theoretical, standardized, s=5, alpha=0.35)

    min_val = min(theoretical.min(), standardized.min())
    max_val = max(theoretical.max(), standardized.max())
    plt.plot([min_val, max_val], [min_val, max_val], linewidth=2)

    plt.title(title)
    plt.xlabel("Theoretical normal quantiles")
    plt.ylabel("Standardized sample quantiles")
    plt.tight_layout()

    path = OUTPUT_DIR / filename
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_variance_comparison(summary_records):
    """
    Compare variance of the sample mean, median, and trimmed mean
    for different distributions and sample sizes.
    """
    distributions = sorted(set(r["distribution"] for r in summary_records))

    paths = []
    for dist in distributions:
        records = [r for r in summary_records if r["distribution"] == dist]

        plt.figure(figsize=(8, 5))
        for estimator in ["mean", "median", "trimmed"]:
            xs = [r["n"] for r in records if r["estimator"] == estimator]
            ys = [r["variance"] for r in records if r["estimator"] == estimator]
            plt.plot(xs, ys, marker="o", label=estimator)

        plt.title(f"Estimator Variance vs Sample Size\n{dist}")
        plt.xlabel("Sample size n")
        plt.ylabel("Variance of estimator")
        plt.legend()
        plt.tight_layout()

        filename = f"variance_comparison_{dist.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')}.png"
        path = OUTPUT_DIR / filename
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(path)

    return paths


# ---------------------------------------------------------------------
# 5. Main experiment
# ---------------------------------------------------------------------
def run_experiment():
    """
    Run all CLT experiments and save plots and numerical summaries.
    """
    distributions = {
        "Uniform": "Uniform(-sqrt(3), sqrt(3))",
        "Normal": "Normal(0, 1)",
        "t_df3": "Student t(df=3)",
    }

    summary_records = []
    selected_qq_paths = []
    histogram_paths = []

    for dist_key, dist_label in distributions.items():
        for n in SAMPLE_SIZES:
            data = generate_samples(dist_key, n, REPLICATIONS, rng)

            estimator_results = {
                "mean": np.mean(data, axis=1),
                "median": np.median(data, axis=1),
                "trimmed": trimmed_mean(data, proportion=0.10),
            }

            # Save histogram plots for each distribution and sample size.
            histogram_paths.append(plot_estimator_histograms(estimator_results, dist_label, n))

            # Numerical summaries.
            for estimator_name, values in estimator_results.items():
                stats = summarize_estimator(values)
                summary_records.append({
                    "distribution": dist_label,
                    "n": n,
                    "estimator": estimator_name,
                    **stats
                })

            # Q-Q plots for the largest sample size only, to visually confirm asymptotic normality.
            if n == max(SAMPLE_SIZES):
                for estimator_name, values in estimator_results.items():
                    filename = (
                        f"qq_{dist_label.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')}"
                        f"_{estimator_name}_n{n}.png"
                    )
                    selected_qq_paths.append(
                        qq_plot(
                            values,
                            title=f"Normal Q-Q Plot: {dist_label}, {estimator_name}, n={n}",
                            filename=filename,
                        )
                    )

    variance_paths = plot_variance_comparison(summary_records)

    # Save summary table as CSV.
    csv_path = OUTPUT_DIR / "clt_summary_statistics.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        header = ["distribution", "n", "estimator", "mean", "variance", "std", "skewness", "excess_kurtosis"]
        f.write(",".join(header) + "\n")
        for r in summary_records:
            f.write(",".join(str(r[h]) for h in header) + "\n")

    # Save and print a concise summary for easy inclusion in the report.
    summary_lines = [
        "Central Limit Theorem Demonstration",
        f"Seed: {SEED}",
        f"Replications per setting: {REPLICATIONS}",
        f"Sample sizes: {SAMPLE_SIZES}",
        "",
        "Selected summary statistics for n=100:",
    ]

    for r in summary_records:
        if r["n"] == 100:
            summary_lines.append(
                f"{r['distribution']:30s} | {r['estimator']:8s} | "
                f"mean={r['mean']:+.5f} | var={r['variance']:.6f} | "
                f"skew={r['skewness']:+.4f} | excess kurt={r['excess_kurtosis']:+.4f}"
            )

    summary_lines.extend(
        [
            "",
            f"Saved CSV: {csv_path.relative_to(SCRIPT_DIR)}",
            f"Number of histogram plots: {len(histogram_paths)}",
            f"Number of Q-Q plots: {len(selected_qq_paths)}",
            f"Number of variance comparison plots: {len(variance_paths)}",
        ]
    )

    summary_text = "\n".join(summary_lines) + "\n"
    SUMMARY_PATH.write_text(summary_text, encoding="utf-8")
    print("\n" + summary_text)

    return {
        "summary_records": summary_records,
        "csv_path": csv_path,
        "histogram_paths": histogram_paths,
        "qq_paths": selected_qq_paths,
        "variance_paths": variance_paths,
        "summary_path": SUMMARY_PATH,
    }


if __name__ == "__main__":
    run_experiment()
