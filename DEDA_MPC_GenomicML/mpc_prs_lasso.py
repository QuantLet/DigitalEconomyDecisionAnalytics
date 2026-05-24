"""
Privacy-preserving Polygenic Risk Score (PRS) computation via coordinate LASSO.

Background
----------
A Polygenic Risk Score aggregates GWAS-derived effect sizes across SNPs:

    PRS_i = sum_j  beta_j * G_{ij}

where beta_j are estimated from a (potentially different) reference GWAS cohort.
In modern practice betas are shrunk via LASSO / Bayesian regression to account
for LD structure (e.g., LDpred2, PRS-CS).

Privacy challenge
-----------------
Individuals in the top 1 % of PRS carry 3–8x elevated disease risk — this
information alone is sensitive.  When multiple biobanks collaborate to
estimate a joint LASSO model, no party should reveal its raw genotypes or
phenotypes to the others.

MPC protocol (stub)
-------------------
We simulate coordinate descent LASSO in a 3-party additive secret-sharing
scheme.  The soft-thresholding step is approximated using a comparison
protocol (replicated by plaintext comparison here for clarity).

References
----------
- Privé et al. (2022) LDpred2.  PLoS Genetics.
- Cho et al. (2018) Secure GWAS.  Nature Genetics.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from mpc_logistic_regression import generate_genomic_data

# ---------------------------------------------------------------------------
# Coordinate LASSO (plaintext reference implementation)
# ---------------------------------------------------------------------------

def coordinate_lasso(G: np.ndarray, y: np.ndarray,
                     lam: float = 0.01, n_iter: int = 100) -> np.ndarray:
    """Single-dataset LASSO via coordinate descent."""
    n, p = G.shape
    beta = np.zeros(p)
    residual = y - G @ beta
    for _ in range(n_iter):
        for j in range(p):
            rj = residual + G[:, j] * beta[j]
            z = G[:, j] @ rj / n
            beta_new = np.sign(z) * max(abs(z) - lam, 0.0)
            residual = rj - G[:, j] * beta_new
            beta[j] = beta_new
    return beta


# ---------------------------------------------------------------------------
# MPC-simulated coordinate LASSO across 3 parties
# ---------------------------------------------------------------------------

def mpc_coordinate_lasso(G_parties: list[np.ndarray],
                         y_parties: list[np.ndarray],
                         lam: float = 0.01,
                         n_iter: int = 100) -> np.ndarray:
    """
    Coordinate LASSO where each party holds a vertical partition of samples.

    In real MPC:
      - Inner products (G[:,j] @ r) are computed via secure dot-product.
      - Soft-thresholding uses a comparison protocol.
    Here we reconstruct from shares to illustrate the numerical outcome.
    """
    G = np.vstack(G_parties)
    y = np.concatenate(y_parties).astype(np.float64)
    # Run plaintext LASSO on reconstructed data (simulates MPC output)
    beta = coordinate_lasso(G, y, lam=lam, n_iter=n_iter)
    return beta


# ---------------------------------------------------------------------------
# PRS distribution analysis
# ---------------------------------------------------------------------------

def prs_distribution_plot(G_test: np.ndarray, y_test: np.ndarray,
                          beta_mpc: np.ndarray, beta_plain: np.ndarray,
                          scaler: StandardScaler):
    G_sc = scaler.transform(G_test)
    prs_mpc = G_sc @ beta_mpc
    prs_plain = G_sc @ beta_plain

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, prs, title in zip(axes,
                               [prs_plain, prs_mpc],
                               ["Plaintext LASSO PRS", "MPC LASSO PRS"]):
        ax.hist(prs[y_test == 0], bins=30, alpha=0.6, label="Controls", color="steelblue")
        ax.hist(prs[y_test == 1], bins=30, alpha=0.6, label="Cases",    color="tomato")
        auc = roc_auc_score(y_test, prs)
        ax.set_title(f"{title}  (AUC={auc:.3f})")
        ax.set_xlabel("PRS")
        ax.set_ylabel("Count")
        ax.axvline(np.percentile(prs, 99), color="k", linestyle="--",
                   label="99th percentile")
        ax.legend()

    plt.suptitle("Polygenic Risk Score Distribution — Cases vs. Controls\n"
                 "(MPC multi-biobank LASSO)", fontsize=12)
    plt.tight_layout()
    plt.savefig("mpc_prs_distribution.png", dpi=150)
    plt.show()
    print("Plot saved to mpc_prs_distribution.png")


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main():
    G, y, beta_true = generate_genomic_data(n_samples=1200, n_snps=50,
                                             heritability=0.4, seed=7)
    split = int(0.7 * len(y))
    G_train, G_test = G[:split], G[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    G_train_sc = scaler.fit_transform(G_train)

    # Partition training data across 3 biobanks
    n = len(y_train)
    cut = [n // 3, 2 * n // 3]
    G_parties = [G_train_sc[:cut[0]], G_train_sc[cut[0]:cut[1]], G_train_sc[cut[1]:]]
    y_parties = [y_train[:cut[0]], y_train[cut[0]:cut[1]], y_train[cut[1]:]]

    lam = 0.005

    # Plaintext baseline
    beta_plain = coordinate_lasso(G_train_sc, y_train.astype(float), lam=lam)

    # MPC simulation
    beta_mpc = mpc_coordinate_lasso(G_parties, y_parties, lam=lam)

    G_test_sc = scaler.transform(G_test)
    auc_plain = roc_auc_score(y_test, G_test_sc @ beta_plain)
    auc_mpc   = roc_auc_score(y_test, G_test_sc @ beta_mpc)

    print(f"Plaintext LASSO PRS  AUC : {auc_plain:.4f}")
    print(f"MPC LASSO PRS        AUC : {auc_mpc:.4f}")

    # Top-1% risk enrichment
    prs_mpc = G_test_sc @ beta_mpc
    top1_mask = prs_mpc >= np.percentile(prs_mpc, 99)
    base_rate = y_test.mean()
    top1_rate = y_test[top1_mask].mean() if top1_mask.sum() > 0 else float("nan")
    print(f"Baseline disease rate       : {base_rate:.3f}")
    print(f"Top-1%% PRS disease rate    : {top1_rate:.3f}  "
          f"({top1_rate/base_rate:.1f}x relative risk)")

    prs_distribution_plot(G_test, y_test, beta_mpc, beta_plain, scaler)


if __name__ == "__main__":
    main()
