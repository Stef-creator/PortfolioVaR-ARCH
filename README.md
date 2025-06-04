# PortfolioVaR-ARCH
Portfolio Value-at-Risk (VaR) estimation using ARCH-family volatility models. Supports daily and weekly horizons with Normal, t-distributed, and Filtered Historical Simulation innovations. Includes model selection, diagnostic tests, and visualization for robust financial risk assessment.

This project implements Value-at-Risk (VaR) estimation for a portfolio using various ARCH-family volatility models. It supports daily and weekly horizons and employs three innovation assumptions: Normal, t-distributed, and Filtered Historical Simulation (FHS).

## Features

- Download historical price data for a custom portfolio using Yahoo Finance.
- Fit multiple ARCH-family models (GARCH, EGARCH, GJR-GARCH, APARCH) and select the best via Akaike Information Criterion (AIC).
- Estimate VaR under different assumptions with Monte Carlo simulations.
- Comprehensive model diagnostics including Ljung-Box and Jarque-Bera tests.
- Visualize simulated returns and risk measures.
- Support for reproducible results with fixed random seeds.

## Installation

```bash
pip install -r requirements.txt
