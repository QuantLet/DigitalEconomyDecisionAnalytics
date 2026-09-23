<div style="margin: 0; padding: 0; text-align: center; border: none;">
<a href="https://quantlet.com" target="_blank" style="text-decoration: none; border: none;">
<img src="https://github.com/StefanGam/test-repo/blob/main/quantlet_design.png?raw=true" alt="Header Image" width="100%" style="margin: 0; padding: 0; display: block; border: none;" />
</a>
</div>

```
Name of Quantlet: DEDA_MPC_GenomicML

Published in: Digital Economy and Decision Analytics

Description:
- Demonstrate Multi-Party Computation (MPC) for privacy-preserving supervised
  machine learning over genomic data
- Simulate a secure logistic regression for disease risk prediction
  (case-control setting) without revealing individual genotypes
- Illustrate Polygenic Risk Score (PRS) computation via LASSO under MPC
- Show how MPC secret-sharing enables collaborative model training across
  multiple data holders (hospitals / biobanks) without sharing raw data

Keywords:
- Python
- MPC
- Multi-Party Computation
- Secret Sharing
- Genomic Data
- Disease Risk Prediction
- Polygenic Risk Score
- PRS
- Logistic Regression
- LASSO
- Privacy-Preserving Machine Learning
- Federated Learning

Author: Pavel Shibaev
```

## Background

Genomic disease risk modeling is a **supervised prediction / classification** problem.
Most methods train models to estimate P(disease | genome) using labeled case–control
data — logistic regression, Polygenic Risk Scores (PRS), penalized regression, and
deep networks — optimizing discrimination (AUC, C-index) across the whole population.

The core privacy challenge: genomic data is highly sensitive and siloed across
institutions. **Multi-Party Computation (MPC)** allows multiple parties (hospitals,
biobanks) to jointly compute a function over their combined data **without** any party
learning the other parties' raw inputs.

### Anomaly Detection vs. Supervised Risk Modeling

| Use-case | Approach |
|---|---|
| Population disease risk (common diseases, PRS) | Supervised classification / regression |
| Rare disease / aberrant expression detection | Unsupervised anomaly detection (OUTRIDER, LOF) |
| Regulatory peak catalogues, omics QC | Unsupervised outlier flagging |

This quantlet focuses on the **supervised** path with MPC-enabled privacy.

## Files

| File | Description |
|---|---|
| `mpc_secret_sharing.py` | Additive secret sharing primitives (3-party, semi-honest) |
| `mpc_logistic_regression.py` | Secure logistic regression for case-control genomic data |
| `mpc_prs_lasso.py` | Privacy-preserving Polygenic Risk Score via coordinate LASSO |
| `DEDA_MPC_GenomicML.ipynb` | Notebook walkthrough of all three demos |

## Quick Start

```bash
pip install numpy scikit-learn matplotlib
python mpc_logistic_regression.py
python mpc_prs_lasso.py
```

## Output

The scripts produce:
- AUC curves comparing plain vs. MPC-secure logistic regression
- PRS distribution plots (cases vs. controls) computed under secret sharing
- Accuracy / privacy overhead summary table
