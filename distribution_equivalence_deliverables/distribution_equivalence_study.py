"""
Distribution Equivalence Study
Author: Ely Jun-Pates
Course: AI212 / Statistical Inference

Purpose:
    This script demonstrates three important distribution equivalences:

    1. Binomial(N, p) approaches Normal(Np, Np(1-p)) as N increases.
    2. Binomial(N, p) approaches Poisson(lambda) when N is large,
       p is small, and Np = lambda.
    3. Exponential(lambda) interarrival times generate a Poisson(lambda*T)
       count distribution over a fixed interval T.

Randomness:
    NumPy's MT19937 bit generator is used for reproducibility.

How to run:
    python3 distribution_equivalence_study.py

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
MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "distribution_equivalence_mpl_config"
XDG_CACHE_DIR = Path(tempfile.gettempdir()) / "distribution_equivalence_xdg_cache"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# 1. Reproducibility setup
# ---------------------------------------------------------------------
SEED = 212
OUTPUT_DIR = SCRIPT_DIR / "distribution_equivalence_output"
OUTPUT_DIR.mkdir(exist_ok=True)
SUMMARY_PATH = SCRIPT_DIR / "simulation_summary.txt"


def make_rng(seed_offset=0):
    """Create an independent MT19937-backed generator for an experiment."""
    return np.random.Generator(np.random.MT19937(seed=SEED + seed_offset))


# ---------------------------------------------------------------------
# 2. Helper mathematical functions
# ---------------------------------------------------------------------
def normal_pdf(x, mu, sigma):
    """Return the theoretical Normal(mu, sigma^2) PDF values."""
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )


def poisson_pmf(k_values, lam):
    """
    Return theoretical Poisson(lambda) PMF values.

    P(X=k) = exp(-lambda) lambda^k / k!

    The computation uses log probabilities for numerical stability:
    log P(X=k) = -lambda + k log(lambda) - log(k!)
    """
    probs = []
    for k in k_values:
        log_p = -lam + k * math.log(lam) - math.lgamma(k + 1)
        probs.append(math.exp(log_p))
    return np.array(probs)


def exp_interarrival_samples(rate, size, rng):
    """
    Generate Exp(rate) samples using inverse transform sampling.

    If U ~ Uniform(0,1), then X = -ln(1-U) / rate has Exp(rate).
    """
    u = rng.random(size)
    eps = np.finfo(float).eps
    u = np.clip(u, eps, 1.0 - eps)
    return -np.log(1.0 - u) / rate


def summarize_sample(samples, theoretical_mean, theoretical_variance):
    """Return sample-vs-theory summary statistics."""
    sample_mean = float(np.mean(samples))
    sample_variance = float(np.var(samples, ddof=1))
    return {
        "sample_mean": sample_mean,
        "theoretical_mean": float(theoretical_mean),
        "absolute_mean_error": abs(sample_mean - theoretical_mean),
        "sample_variance": sample_variance,
        "theoretical_variance": float(theoretical_variance),
        "absolute_variance_error": abs(sample_variance - theoretical_variance),
    }


# ---------------------------------------------------------------------
# 3. Part A: Binomial and Normal equivalence
# ---------------------------------------------------------------------
def binomial_normal_equivalence():
    """
    Demonstrate that Binomial(N,p) approaches Normal(Np, Np(1-p))
    by comparing several increasing values of N.
    """
    rng = make_rng(0)
    sample_size = 100_000
    p = 0.40
    n_values = [10, 30, 100, 500]
    rows = []

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for ax, n_trials in zip(axes, n_values):
        samples = rng.binomial(n=n_trials, p=p, size=sample_size)
        mu = n_trials * p
        var = n_trials * p * (1.0 - p)
        sigma = math.sqrt(var)

        x_grid = np.linspace(samples.min() - 3, samples.max() + 3, 600)
        bins = np.arange(samples.min() - 0.5, samples.max() + 1.5, 1)

        ax.hist(samples, bins=bins, density=True, alpha=0.55, label="Binomial")
        ax.plot(x_grid, normal_pdf(x_grid, mu, sigma), linewidth=2.0, label="Normal")
        ax.set_title(f"N={n_trials}, p={p}")
        ax.set_xlabel("Number of successes")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

        row = {
            "N": n_trials,
            "p": p,
            "sample_size": sample_size,
            "mu": mu,
            "sigma": sigma,
            **summarize_sample(samples, mu, var),
        }
        rows.append(row)

    fig.suptitle("Binomial-to-Normal Convergence as N Increases")
    fig.tight_layout()
    convergence_path = OUTPUT_DIR / "binomial_normal_convergence.png"
    fig.savefig(convergence_path, dpi=160)
    plt.close(fig)

    # Keep the original single-panel plot name for simple viewing/submission.
    n_trials = 100
    samples = rng.binomial(n=n_trials, p=p, size=sample_size)
    mu = n_trials * p
    var = n_trials * p * (1.0 - p)
    sigma = math.sqrt(var)
    x_grid = np.linspace(samples.min() - 5, samples.max() + 5, 600)

    plt.figure(figsize=(8, 5))
    plt.hist(
        samples,
        bins=np.arange(samples.min() - 0.5, samples.max() + 1.5, 1),
        density=True,
        alpha=0.55,
        label=f"Simulated Binomial(N={n_trials}, p={p})",
    )
    plt.plot(
        x_grid,
        normal_pdf(x_grid, mu, sigma),
        linewidth=2.5,
        label=f"Normal PDF: mu={mu:.1f}, sigma={sigma:.3f}",
    )
    plt.title("Binomial-Normal Equivalence")
    plt.xlabel("Number of successes")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    single_path = OUTPUT_DIR / "binomial_normal_equivalence.png"
    plt.savefig(single_path, dpi=160)
    plt.close()

    return {
        "distribution": "Binomial vs Normal",
        "sample_size": sample_size,
        "p": p,
        "n_values": n_values,
        "rows": rows,
        "plot": str(convergence_path.relative_to(SCRIPT_DIR)),
        "single_plot": str(single_path.relative_to(SCRIPT_DIR)),
    }


# ---------------------------------------------------------------------
# 4. Part B: Binomial and Poisson equivalence
# ---------------------------------------------------------------------
def binomial_poisson_equivalence():
    """
    Demonstrate that Binomial(N,p) approaches Poisson(lambda)
    when N is large, p is small, and Np = lambda.
    """
    rng = make_rng(1)
    sample_size = 100_000
    lam = 5.0
    n_trials = 1000
    p = lam / n_trials

    samples = rng.binomial(n=n_trials, p=p, size=sample_size)

    k_max = max(int(samples.max()), int(lam + 6 * math.sqrt(lam)))
    k_values = np.arange(0, k_max + 1)
    poisson_probs = poisson_pmf(k_values, lam)

    counts = np.bincount(samples, minlength=k_max + 1)[: k_max + 1]
    empirical_probs = counts / sample_size

    plt.figure(figsize=(8, 5))
    plt.bar(
        k_values - 0.20,
        empirical_probs,
        width=0.40,
        alpha=0.65,
        label=f"Simulated Binomial(N={n_trials}, p={p:.4f})",
    )
    plt.bar(
        k_values + 0.20,
        poisson_probs,
        width=0.40,
        alpha=0.65,
        label=f"Poisson PMF: lambda={lam}",
    )
    plt.title("Binomial-Poisson Equivalence")
    plt.xlabel("Number of successes/events")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    path = OUTPUT_DIR / "binomial_poisson_equivalence.png"
    plt.savefig(path, dpi=160)
    plt.close()

    return {
        "distribution": "Binomial vs Poisson",
        "N": n_trials,
        "p": p,
        "lambda": lam,
        "sample_size": sample_size,
        **summarize_sample(samples, lam, lam),
        "plot": str(path.relative_to(SCRIPT_DIR)),
    }


# ---------------------------------------------------------------------
# 5. Part C: Exponential interarrivals and Poisson counts
# ---------------------------------------------------------------------
def exponential_poisson_equivalence():
    """
    Simulate a Poisson process using exponential interarrival times.

    If interarrival times are iid Exp(lambda), then the number of arrivals
    in interval [0,T] follows Poisson(lambda*T).
    """
    rng = make_rng(2)
    num_trials = 50_000
    rate = 2.0
    interval_length = 4.0
    poisson_rate = rate * interval_length

    arrival_counts = []

    for _ in range(num_trials):
        total_time = 0.0
        count = 0

        # Generate arrivals until the next arrival would exceed T.
        while True:
            wait = exp_interarrival_samples(rate, 1, rng)[0]
            total_time += wait

            if total_time <= interval_length:
                count += 1
            else:
                break

        arrival_counts.append(count)

    arrival_counts = np.array(arrival_counts)

    k_max = max(int(arrival_counts.max()), int(poisson_rate + 6 * math.sqrt(poisson_rate)))
    k_values = np.arange(0, k_max + 1)
    poisson_probs = poisson_pmf(k_values, poisson_rate)

    counts = np.bincount(arrival_counts, minlength=k_max + 1)[: k_max + 1]
    empirical_probs = counts / num_trials

    plt.figure(figsize=(8, 5))
    plt.bar(
        k_values - 0.20,
        empirical_probs,
        width=0.40,
        alpha=0.65,
        label="Simulated counts from Exp(lambda) interarrivals",
    )
    plt.bar(
        k_values + 0.20,
        poisson_probs,
        width=0.40,
        alpha=0.65,
        label=f"Poisson PMF: lambda*T={poisson_rate}",
    )
    plt.title("Exponential Interarrival Times and Poisson Counts")
    plt.xlabel(f"Number of arrivals in interval T={interval_length}")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    path = OUTPUT_DIR / "exponential_poisson_equivalence.png"
    plt.savefig(path, dpi=160)
    plt.close()

    return {
        "distribution": "Exponential interarrivals vs Poisson counts",
        "lambda": rate,
        "T": interval_length,
        "lambda_T": poisson_rate,
        "num_trials": num_trials,
        **summarize_sample(arrival_counts, poisson_rate, poisson_rate),
        "plot": str(path.relative_to(SCRIPT_DIR)),
    }


def format_summary(summaries):
    """Format experiment summaries for console output and the summary file."""
    lines = ["Distribution Equivalence Study", f"Seed: {SEED}"]

    normal_summary = summaries[0]
    lines.append("")
    lines.append(normal_summary["distribution"])
    lines.append(f"  p: {normal_summary['p']:.6f}")
    lines.append(f"  sample_size: {normal_summary['sample_size']}")
    lines.append(f"  plot: {normal_summary['plot']}")
    lines.append("  convergence_rows:")
    for row in normal_summary["rows"]:
        lines.append(
            "    "
            f"N={row['N']}, "
            f"sample_mean={row['sample_mean']:.6f}, "
            f"theoretical_mean={row['theoretical_mean']:.6f}, "
            f"absolute_mean_error={row['absolute_mean_error']:.6f}, "
            f"sample_variance={row['sample_variance']:.6f}, "
            f"theoretical_variance={row['theoretical_variance']:.6f}, "
            f"absolute_variance_error={row['absolute_variance_error']:.6f}"
        )

    for summary in summaries[1:]:
        lines.append("")
        lines.append(summary["distribution"])
        for key, value in summary.items():
            if key == "distribution":
                continue
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.6f}")
            else:
                lines.append(f"  {key}: {value}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------
# 6. Run all experiments and save summaries
# ---------------------------------------------------------------------
if __name__ == "__main__":
    summaries = [
        binomial_normal_equivalence(),
        binomial_poisson_equivalence(),
        exponential_poisson_equivalence(),
    ]

    summary_text = format_summary(summaries)
    SUMMARY_PATH.write_text(summary_text, encoding="utf-8")
    print("\n" + summary_text)
