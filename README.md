# PortfolioVaR-ARCH

## Overview

This project implements portfolio Value-at-Risk (VaR) estimation using ARCH/GARCH models. The goal is to model and forecast the conditional volatility of financial asset returns and assess portfolio risk under changing market conditions.

## Contents

- `PortfolioVaR-ARCH.ipynb` — Jupyter notebook demonstrating the methodology, data processing, and model results.
- `var_estimation.py` — Python module implementing ARCH/GARCH model estimation and portfolio VaR calculations.
- `requirements.txt` — Python dependencies for reproducibility.
- `Theory/` — Collection of key academic papers related to time series econometrics and risk modeling.
- `README.md` — Project documentation.
- `LICENSE` — Project license.

## Features

- Estimation of conditional volatility using ARCH and GARCH models.
- Backtesting of portfolio VaR estimates.
- Use of robust econometric techniques such as stationarity tests and information criteria.
- Modular Python code suitable for extension and integration into larger risk frameworks.

## Getting Started

1. Clone the repository:

```bash
git clone https://github.com/Stef-creator/PortfolioVaR-ARCH.git
cd PortfolioVaR-ARCH
```
2. Set up a Python environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

3. Run the notebook PortfolioVaR-ARCH.ipynb to explore the data, models, and results.

Usage
- Use the var_estimation.py module to fit ARCH/GARCH models on your financial return series.
- Customize portfolio weights and parameters to simulate different risk scenarios.
- Extend the notebook or scripts with your own data or additional risk measures.

References

The Theory/ folder contains foundational papers relevant to time series econometrics and VaR modeling.

License
This project is licensed under the GNU License. See LICENSE for details.
