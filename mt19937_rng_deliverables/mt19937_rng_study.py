
"""
MT19937 Random Number Generation Study
Author: Ely Jun-Pates
Course: AI212 / Statistical Inference

Purpose:
    This script uses NumPy's MT19937 bit generator to produce uniform random
    numbers and transform them into samples from:
        1. Exponential(lambda)
        2. Normal(mu, sigma^2)
        3. Weibull(k, scale)

    It then plots normalized histograms and overlays the theoretical PDFs.

How to run:
    python3 mt19937_rng_study.py

Required packages:
    numpy
    matplotlib

Optional:
    scipy is NOT required. All theoretical PDFs are computed manually.
"""

import math
import os
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# Put Matplotlib/font caches in a writable temporary location before importing
# Matplotlib. This keeps the script usable in restricted or sandboxed environments.
MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "mt19937_mpl_config"
XDG_CACHE_DIR = Path(tempfile.gettempdir()) / "mt19937_xdg_cache"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# 1. Reproducibility and MT19937 setup
# ---------------------------------------------------------------------
# MT19937 is a pseudorandom bit generator. It generates deterministic
# sequences once a seed is fixed. This is useful for reproducible simulations.
SEED = 212
N = 100_000

OUTPUT_DIR = SCRIPT_DIR / "mt19937_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Use NumPy's modern Generator interface with the MT19937 bit generator.
bit_generator = np.random.MT19937(seed=SEED)
rng = np.random.Generator(bit_generator)


# ---------------------------------------------------------------------
# 2. Generate base uniform random numbers
# ---------------------------------------------------------------------
# MT19937 fundamentally generates pseudorandom bits/integers, which NumPy
# converts into floating-point numbers approximately distributed as U(0,1).
U_exp = rng.random(N)
U_norm1 = rng.random(N)
U_norm2 = rng.random(N)
U_weibull = rng.random(N)

# To avoid log(0), clip extremely small values if needed.
EPS = np.finfo(float).eps
U_exp = np.clip(U_exp, EPS, 1.0 - EPS)
U_norm1 = np.clip(U_norm1, EPS, 1.0 - EPS)
U_norm2 = np.clip(U_norm2, EPS, 1.0 - EPS)
U_weibull = np.clip(U_weibull, EPS, 1.0 - EPS)


# ---------------------------------------------------------------------
# 3. Transformation 1: Exponential(lambda)
# ---------------------------------------------------------------------
# If U ~ Uniform(0,1), then X = F^{-1}(U).
#
# For X ~ Exp(lambda),
#     F(x) = 1 - exp(-lambda x), x >= 0.
#
# Set U = 1 - exp(-lambda x):
#     exp(-lambda x) = 1 - U
#     -lambda x = ln(1 - U)
#     x = -ln(1 - U) / lambda
#
# Since 1-U is also Uniform(0,1), many codes use -ln(U)/lambda.
lambda_exp = 1.5
X_exp = -np.log(1.0 - U_exp) / lambda_exp


def exp_pdf(x, lam):
    """Theoretical PDF of Exp(lambda)."""
    y = lam * np.exp(-lam * x)
    y[x < 0] = 0
    return y


# ---------------------------------------------------------------------
# 4. Transformation 2: Normal(mu, sigma^2) using Box-Muller
# ---------------------------------------------------------------------
# The Box-Muller transform uses two independent U(0,1) variables.
#
# If U1, U2 ~ Uniform(0,1), independent:
#     Z = sqrt(-2 ln U1) cos(2 pi U2)
# is standard normal N(0,1).
#
# Then:
#     X = mu + sigma Z
# is N(mu, sigma^2).
mu = 10.0
sigma = 2.0

Z = np.sqrt(-2.0 * np.log(U_norm1)) * np.cos(2.0 * np.pi * U_norm2)
X_norm = mu + sigma * Z


def normal_pdf(x, mu, sigma):
    """Theoretical PDF of Normal(mu, sigma^2)."""
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )


# ---------------------------------------------------------------------
# 5. Transformation 3: Weibull(k, scale) using inverse transform
# ---------------------------------------------------------------------
# Chosen distribution: Weibull distribution.
#
# Motivation:
#   Weibull distributions are useful for reliability, survival time,
#   waiting-time, and engineering failure-time modeling.
#
# For Weibull(shape=k, scale=eta),
#     F(x) = 1 - exp(-(x/eta)^k), x >= 0.
#
# Set U = 1 - exp(-(x/eta)^k):
#     exp(-(x/eta)^k) = 1 - U
#     -(x/eta)^k = ln(1 - U)
#     (x/eta)^k = -ln(1 - U)
#     x = eta[-ln(1 - U)]^(1/k)
k_shape = 2.0
eta_scale = 3.0

X_weibull = eta_scale * (-np.log(1.0 - U_weibull)) ** (1.0 / k_shape)


def weibull_pdf(x, k, eta):
    """Theoretical PDF of Weibull(shape=k, scale=eta)."""
    y = (k / eta) * (x / eta) ** (k - 1.0) * np.exp(-((x / eta) ** k))
    y[x < 0] = 0
    return y


# ---------------------------------------------------------------------
# 6. Utility function for histogram + theoretical PDF
# ---------------------------------------------------------------------
def plot_histogram_with_pdf(samples, x_grid, pdf_values, title, xlabel, filename):
    """
    Plot a normalized histogram of simulated samples and overlay theoretical PDF.

    density=True makes the histogram integrate approximately to 1,
    so it can be compared directly with a probability density function.
    """
    plt.figure(figsize=(8, 5))
    plt.hist(samples, bins=80, density=True, alpha=0.55, label="MT19937 simulated samples")
    plt.plot(x_grid, pdf_values, linewidth=2.5, label="Theoretical PDF")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    path = OUTPUT_DIR / filename
    plt.savefig(path, dpi=160)
    plt.close()
    return path


# ---------------------------------------------------------------------
# 7. Make plots
# ---------------------------------------------------------------------
x_exp = np.linspace(0, np.percentile(X_exp, 99.7), 500)
exp_plot = plot_histogram_with_pdf(
    X_exp,
    x_exp,
    exp_pdf(x_exp, lambda_exp),
    title=f"Exponential Distribution: lambda = {lambda_exp}",
    xlabel="x",
    filename="exponential_mt19937.png",
)

x_norm = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 500)
norm_plot = plot_histogram_with_pdf(
    X_norm,
    x_norm,
    normal_pdf(x_norm, mu, sigma),
    title=f"Normal Distribution: mu = {mu}, sigma = {sigma}",
    xlabel="x",
    filename="normal_mt19937.png",
)

x_weibull = np.linspace(0, np.percentile(X_weibull, 99.7), 500)
weibull_plot = plot_histogram_with_pdf(
    X_weibull,
    x_weibull,
    weibull_pdf(x_weibull, k_shape, eta_scale),
    title=f"Weibull Distribution: shape = {k_shape}, scale = {eta_scale}",
    xlabel="x",
    filename="weibull_mt19937.png",
)


# ---------------------------------------------------------------------
# 8. Numerical summaries
# ---------------------------------------------------------------------
def summarize(samples):
    """Return common numerical summaries for a sample."""
    return {
        "sample_mean": float(np.mean(samples)),
        "sample_variance": float(np.var(samples, ddof=1)),
        "sample_min": float(np.min(samples)),
        "sample_max": float(np.max(samples)),
    }


summaries = {
    "Exponential(lambda=1.5)": summarize(X_exp),
    "Normal(mu=10, sigma=2)": summarize(X_norm),
    "Weibull(shape=2, scale=3)": summarize(X_weibull),
}

# Add theoretical means and variances.
summaries["Exponential(lambda=1.5)"]["theoretical_mean"] = 1.0 / lambda_exp
summaries["Exponential(lambda=1.5)"]["theoretical_variance"] = 1.0 / (lambda_exp**2)

summaries["Normal(mu=10, sigma=2)"]["theoretical_mean"] = mu
summaries["Normal(mu=10, sigma=2)"]["theoretical_variance"] = sigma**2

# Weibull theoretical mean = eta Gamma(1 + 1/k)
# Weibull theoretical variance = eta^2[Gamma(1+2/k) - Gamma(1+1/k)^2]
summaries["Weibull(shape=2, scale=3)"]["theoretical_mean"] = eta_scale * math.gamma(1 + 1 / k_shape)
summaries["Weibull(shape=2, scale=3)"]["theoretical_variance"] = eta_scale**2 * (
    math.gamma(1 + 2 / k_shape) - math.gamma(1 + 1 / k_shape) ** 2
)

print("\nMT19937 Random Number Generation Study")
print("Seed:", SEED)
print("Sample size per distribution:", N)
print("\nNumerical summaries:")
for name, stats in summaries.items():
    print(f"\n{name}")
    for key, value in stats.items():
        print(f"  {key}: {value:.6f}")

print("\nPlots saved to:")
print(" ", exp_plot)
print(" ", norm_plot)
print(" ", weibull_plot)
