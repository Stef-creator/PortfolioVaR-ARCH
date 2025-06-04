from arch import arch_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t
from datetime import datetime, timedelta
import yfinance as yf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera
import statsmodels.api as sm



def download_data(tickers, start_date='2020-01-01', end_date=None):
    """
    Downloads historical adjusted closing price data for given stock tickers using yfinance.

    Args:
        tickers (list of str): List of ticker symbols to download data for.
        start_date (str, optional): Start date for data in 'YYYY-MM-DD' format. Defaults to '2020-01-01'.
        end_date (str, optional): End date for data in 'YYYY-MM-DD' format. Defaults to yesterday's date if not provided.

    Returns:
        pandas.DataFrame: DataFrame containing adjusted closing prices for each ticker, indexed by date.

    Raises:
        Exception: If data download fails or tickers are invalid.

    Note:
        Requires 'yfinance', 'pandas', and 'datetime' libraries.
    """
    if end_date is None:
        end_date = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    raw = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)
    data = pd.concat([raw[ticker]['Close'] for ticker in tickers], axis=1)
    data.columns = tickers
    data.dropna(inplace=True)
    return data


def stationarity_tests(series, significance=0.05):
    """
    Perform ADF and KPSS stationarity tests on a time series.

    Parameters:
        series (pd.Series or np.ndarray): The time series to test.
        significance (float): Significance level to determine stationarity (default 0.05).

    Returns:
        pd.DataFrame: Summary table with test statistics, p-values, and stationarity verdicts.
    """
    # ADF Test
    adf_result = adfuller(series, autolag='AIC')
    adf_stat = adf_result[0]
    adf_pval = adf_result[1]
    adf_stationary = adf_pval < significance

    # KPSS Test
    kpss_result = kpss(series, regression='c', nlags="auto")
    kpss_stat = kpss_result[0]
    kpss_pval = kpss_result[1]
    kpss_stationary = kpss_pval > significance

    # Prepare output
    results = {
        'Test': ['Augmented Dickey-Fuller', 'KPSS'],
        'Test Statistic': [adf_stat, kpss_stat],
        'p-value': [adf_pval, kpss_pval],
        'Stationary': [adf_stationary, kpss_stationary],
        'Null Hypothesis': ['Unit root (non-stationary)', 'Stationary']
    }

    return pd.DataFrame(results)


def run_diagnostics_two_models(model_result1, model_result2, model_name1="Model 1", model_name2="Model 2"):
    """
    Run diagnostic tests and plots for two ARCH/GARCH model results side-by-side.
    Includes:
    - Standardized residuals plot
    - Ljung-Box test for autocorrelation (residuals and squared residuals)
    - Jarque-Bera test for normality
    - Q-Q plots
    Returns a summary DataFrame with test statistics, p-values, and pass/fail for both models.
    """

    def diagnostics_for_model(res, name):
        std_resid = res.std_resid.dropna()


        lb_resid = acorr_ljungbox(std_resid, lags=[10], return_df=True).iloc[0]
        lb_sq_resid = acorr_ljungbox(std_resid**2, lags=[10], return_df=True).iloc[0]

        
        jb_stat, jb_pvalue = jarque_bera(std_resid)

        return {
            "Model": name,
            "LB Residuals Stat": lb_resid["lb_stat"],
            "LB Residuals p": lb_resid["lb_pvalue"],
            "LB Residuals Pass": lb_resid["lb_pvalue"] > 0.05,
            "LB Squared Residuals Stat": lb_sq_resid["lb_stat"],
            "LB Squared Residuals p": lb_sq_resid["lb_pvalue"],
            "LB Squared Residuals Pass": lb_sq_resid["lb_pvalue"] > 0.05,
            "JB Stat": jb_stat,
            "JB p": jb_pvalue,
            "JB Pass": jb_pvalue > 0.05,
            "Std Residuals": std_resid
        }

    diag1 = diagnostics_for_model(model_result1, model_name1)
    diag2 = diagnostics_for_model(model_result2, model_name2)

    # Plot standardized residuals side-by-side
    fig, axes = plt.subplots(2, 2, figsize=(15,10))
    axes = axes.flatten()

    # Residuals plot
    axes[0].plot(diag1["Std Residuals"])
    axes[0].set_title(f'Standardized Residuals - {model_name1}')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Std Residual')

    axes[1].plot(diag2["Std Residuals"])
    axes[1].set_title(f'Standardized Residuals - {model_name2}')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Std Residual')

    # Q-Q plots
    sm.qqplot(diag1["Std Residuals"], line='s', ax=axes[2])
    axes[2].set_title(f'Q-Q Plot - {model_name1}')

    sm.qqplot(diag2["Std Residuals"], line='s', ax=axes[3])
    axes[3].set_title(f'Q-Q Plot - {model_name2}')

    plt.tight_layout()
    plt.show()

    # Create summary table
    summary_df = pd.DataFrame([
        {
            "Model": diag1["Model"],
            "Test": "Ljung-Box Residuals",
            "Statistic": diag1["LB Residuals Stat"],
            "p-value": diag1["LB Residuals p"],
            "Pass": diag1["LB Residuals Pass"]
        },
        {
            "Model": diag1["Model"],
            "Test": "Ljung-Box Squared Residuals",
            "Statistic": diag1["LB Squared Residuals Stat"],
            "p-value": diag1["LB Squared Residuals p"],
            "Pass": diag1["LB Squared Residuals Pass"]
        },
        {
            "Model": diag1["Model"],
            "Test": "Jarque-Bera",
            "Statistic": diag1["JB Stat"],
            "p-value": diag1["JB p"],
            "Pass": diag1["JB Pass"]
        },
        {
            "Model": diag2["Model"],
            "Test": "Ljung-Box Residuals",
            "Statistic": diag2["LB Residuals Stat"],
            "p-value": diag2["LB Residuals p"],
            "Pass": diag2["LB Residuals Pass"]
        },
        {
            "Model": diag2["Model"],
            "Test": "Ljung-Box Squared Residuals",
            "Statistic": diag2["LB Squared Residuals Stat"],
            "p-value": diag2["LB Squared Residuals p"],
            "Pass": diag2["LB Squared Residuals Pass"]
        },
        {
            "Model": diag2["Model"],
            "Test": "Jarque-Bera",
            "Statistic": diag2["JB Stat"],
            "p-value": diag2["JB p"],
            "Pass": diag2["JB Pass"]
        },
    ])

    return summary_df


def fit_all_garch_models(portfolio_returns, dist='normal'):
    """
    Fits multiple GARCH-type models (GARCH, EGARCH, GJR-GARCH, APARCH) to a given return series
    using different combinations of lag orders (p, q), and returns a DataFrame of results sorted by AIC.

    Parameters
    ----------
    portfolio_returns : array-like or pandas.Series
        The time series of portfolio returns to which the GARCH models will be fitted.
    dist : str, optional (default='normal')
        The distribution to use for the model residuals. Common options include 'normal', 't', etc.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the following columns for each fitted model:
            - 'Model': The model type (e.g., 'GARCH', 'EGARCH', etc.).
            - 'p': The lag order for the ARCH term.
            - 'q': The lag order for the GARCH term.
            - 'AIC': The Akaike Information Criterion for the fitted model.
            - 'Result': The fitted model result object.

    Notes
    -----
    - The function scales the input returns by 100 before fitting.
    - For each model type, it tries all combinations of p and q in the range [1, 2].
    - If a model fails to fit, the exception is caught and a message is printed.
    """
    models = ['GARCH', 'EGARCH', 'GJR-GARCH', 'APARCH']
    model_map = {'GARCH': 'Garch', 'EGARCH': 'EGARCH', 'GJR-GARCH': 'GARCH', 'APARCH': 'APARCH'}
    power_map = {'GARCH': 2.0, 'EGARCH': 1.0, 'GJR-GARCH': 2.0, 'APARCH': 1.0}
    o_map = {'GARCH': 0, 'EGARCH': 0, 'GJR-GARCH': 1, 'APARCH': 1}

    results = []

    for model_name in models:
        for p in range(1, 3):
            for q in range(1, 3):
                try:
                    am = arch_model(
                        portfolio_returns * 100,  # scale to percent
                        vol=model_map[model_name],
                        p=p,
                        q=q,
                        o=o_map[model_name],
                        power=power_map[model_name],
                        dist=dist  # pass distribution here
                    )
                    res = am.fit(disp='off')
                    results.append({
                        'Model': model_name,
                        'p': p,
                        'q': q,
                        'AIC': res.aic,
                        'Result': res
                    })
                except Exception as e:
                    print(f"Model {model_name}({p},{q}) failed: {e}")

    return pd.DataFrame(results).sort_values('AIC')

def compute_var_arch_model_daily(
    model_result_norm,
    model_result_t,
    portfolio_returns,
    portfolio_value,
    simulations=10000,
    confidence_levels=[0.95, 0.99]
):
    """
    Simulate Value-at-Risk (VaR) using separate fitted models for normal and t-distributed innovations.

    Parameters
    ----------
    model_result_norm : fitted ARCH model result with Normal innovations
    model_result_t : fitted ARCH model result with Student's t innovations
    portfolio_returns : pd.Series or np.ndarray
        Historical portfolio returns.
    portfolio_value : float
        Portfolio value in currency units.
    simulations : int
        Number of simulations to run.
    confidence_levels : list of float
        Confidence levels for VaR.

    Returns
    -------
    pd.DataFrame
        VaR summary table with simulation method, confidence level, VaR %, VaR $, and mean simulated return.
    """
    expected_daily = portfolio_returns.mean() * 100  # in percent

    # Normal innovations simulation (using normal-fitted model)
    vol_norm = np.sqrt(model_result_norm.forecast(horizon=1).variance.values[-1, 0])
    sim_norm = np.random.normal(loc=expected_daily, scale=vol_norm, size=simulations)
    var_norm_pct = pd.Series(sim_norm).quantile(1 - np.array(confidence_levels))
    var_norm_usd = var_norm_pct / 100 * portfolio_value
    mean_norm = np.mean(sim_norm)

    # t-distribution innovations simulation (using t-fitted model)
    vol_t = np.sqrt(model_result_t.forecast(horizon=1).variance.values[-1, 0])
    dof = model_result_t.params.get('nu', 10)  # degrees of freedom from t-model
    sim_t = vol_t * t.rvs(df=dof, size=simulations) + expected_daily
    var_t_pct = pd.Series(sim_t).quantile(1 - np.array(confidence_levels))
    var_t_usd = var_t_pct / 100 * portfolio_value
    mean_t = np.mean(sim_t)

    # Filtered Historical Simulation (FHS) uses residuals from normal model
    std_resid = model_result_norm.std_resid.dropna()
    sim_fhs = vol_norm * np.random.choice(std_resid, size=simulations, replace=True) + expected_daily
    var_fhs_pct = pd.Series(sim_fhs).quantile(1 - np.array(confidence_levels))
    var_fhs_usd = var_fhs_pct / 100 * portfolio_value
    mean_fhs = np.mean(sim_fhs)

    # Plot histograms
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    sim_data = {'Normal': sim_norm, 't-distribution': sim_t, 'FHS': sim_fhs}
    var_lines = [var_norm_pct, var_t_pct, var_fhs_pct]
    colors = ['blue', 'green', 'orange']

    for ax, (title, data), var, color in zip(axes, sim_data.items(), var_lines, colors):
        ax.hist(data, bins=50, density=True, alpha=0.6, color=color, label=f'{title} Simulation')
        for cl, v in zip(confidence_levels, var):
            ax.axvline(v, color='k', linestyle='--', label=f'{int(cl*100)}% VaR')
        mean_val = np.mean(data)
        ax.axvline(mean_val, color='red', linestyle='-', label='Mean')
        ax.set_title(f'{title} Innovations')
        ax.set_xlabel('Simulated Daily Return (%)')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    # Return VaR summary
    rows = []
    for i, cl in enumerate(confidence_levels):
        rows.append({
            'Method': 'Normal',
            'Confidence Level': cl,
            'VaR (%)': var_norm_pct.iloc[i],
            'VaR ($)': var_norm_usd.iloc[i],
            'Mean (%)': mean_norm
        })
        rows.append({
            'Method': 't-distribution',
            'Confidence Level': cl,
            'VaR (%)': var_t_pct.iloc[i],
            'VaR ($)': var_t_usd.iloc[i],
            'Mean (%)': mean_t
        })
        rows.append({
            'Method': 'FHS',
            'Confidence Level': cl,
            'VaR (%)': var_fhs_pct.iloc[i],
            'VaR ($)': var_fhs_usd.iloc[i],
            'Mean (%)': mean_fhs
        })

    return pd.DataFrame(rows)

def compute_var_arch_model_weekly(
    model_result_norm,
    model_result_t,
    portfolio_returns,
    portfolio_value,
    simulations=10000,
    confidence_levels=[0.95, 0.99]
):
    """
    Simulate weekly Value-at-Risk (VaR) using two fitted ARCH-family models (Normal and t-distributed innovations).

    Parameters
    ----------
    model_result_norm : fitted ARCH model result with Normal innovations
    model_result_t : fitted ARCH model result with Student's t innovations
    portfolio_returns : pd.Series or np.ndarray
        Daily historical portfolio returns in decimal (not percent).
    portfolio_value : float
        Portfolio value in currency units.
    simulations : int
        Number of Monte Carlo simulations.
    confidence_levels : list of float
        Confidence levels for VaR calculation.

    Returns
    -------
    pd.DataFrame
        VaR summary with columns: Method, Confidence Level, VaR (%), VaR ($), Mean (%).
    """
    horizon = 5  # trading days per week
    expected_daily = portfolio_returns.mean() * 100  # daily return in %
    expected_weekly = expected_daily * horizon  # scale linearly

    # Volatility forecast for normal model (daily)
    vol_norm_daily = np.sqrt(model_result_norm.forecast(horizon=1).variance.values[-1, 0])
    vol_norm_weekly = vol_norm_daily * np.sqrt(horizon)

    # Volatility forecast for t model (daily)
    vol_t_daily = np.sqrt(model_result_t.forecast(horizon=1).variance.values[-1, 0])
    vol_t_weekly = vol_t_daily * np.sqrt(horizon)
    dof = model_result_t.params.get('nu', 10)

    # Simulate Normal innovations
    sim_norm = np.random.normal(loc=expected_weekly, scale=vol_norm_weekly, size=simulations)
    var_norm_pct = pd.Series(sim_norm).quantile(1 - np.array(confidence_levels))
    var_norm_usd = var_norm_pct / 100 * portfolio_value
    mean_norm = np.mean(sim_norm)

    # Simulate t-distributed innovations
    sim_t = vol_t_weekly * t.rvs(df=dof, size=simulations) + expected_weekly
    var_t_pct = pd.Series(sim_t).quantile(1 - np.array(confidence_levels))
    var_t_usd = var_t_pct / 100 * portfolio_value
    mean_t = np.mean(sim_t)

    # Filtered Historical Simulation (FHS) using residuals from normal model
    std_resid = model_result_norm.std_resid.dropna()
    sim_fhs = vol_norm_weekly * np.random.choice(std_resid, size=simulations, replace=True) + expected_weekly
    var_fhs_pct = pd.Series(sim_fhs).quantile(1 - np.array(confidence_levels))
    var_fhs_usd = var_fhs_pct / 100 * portfolio_value
    mean_fhs = np.mean(sim_fhs)

    # Plot histograms
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    sim_data = {'Normal': sim_norm, 't-distribution': sim_t, 'FHS': sim_fhs}
    var_lines = [var_norm_pct, var_t_pct, var_fhs_pct]
    colors = ['blue', 'green', 'orange']

    for ax, (title, data), var, color in zip(axes, sim_data.items(), var_lines, colors):
        ax.hist(data, bins=50, density=True, alpha=0.6, color=color, label=f'{title} Simulation')
        for cl, v in zip(confidence_levels, var):
            ax.axvline(v, color='k', linestyle='--', label=f'{int(cl*100)}% VaR')
        mean_val = np.mean(data)
        ax.axvline(mean_val, color='red', linestyle='-', label='Mean')
        ax.set_title(f'{title} Innovations - Weekly Returns')
        ax.set_xlabel('Simulated Weekly Return (%)')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    # Return DataFrame with results
    rows = []
    for i, cl in enumerate(confidence_levels):
        rows.append({
            "Method": "Normal",
            "Confidence Level": cl,
            "VaR (%)": var_norm_pct.iloc[i],
            "VaR ($)": var_norm_usd.iloc[i],
            "Mean (%)": mean_norm
        })
        rows.append({
            "Method": "t-distribution",
            "Confidence Level": cl,
            "VaR (%)": var_t_pct.iloc[i],
            "VaR ($)": var_t_usd.iloc[i],
            "Mean (%)": mean_t
        })
        rows.append({
            "Method": "FHS",
            "Confidence Level": cl,
            "VaR (%)": var_fhs_pct.iloc[i],
            "VaR ($)": var_fhs_usd.iloc[i],
            "Mean (%)": mean_fhs
        })

    return pd.DataFrame(rows)
