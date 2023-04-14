import numpy as np
import pandas as pd
import statsmodels.api as sm


def linear_regression(data, x, y):
    # Select the predictor variables and the response variable
    data.dropna(inplace=True)
    X = data[x]
    y = data[y]

    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    y_std = (y - np.mean(y, axis=0)) / np.std(y, axis=0)

    # Add a constant term to the predictor variables (required for statsmodels)
    X = sm.add_constant(X)
    X_std = sm.add_constant(X_std)

    # Fit the multiple linear regression model
    model = sm.OLS(y, X).fit()
    std_model = sm.OLS(y_std, X_std).fit()

    # Table 1: Regression Coefficients Table
    coefficients_table = pd.read_html(
        model.summary().tables[1].as_html(), header=0, index_col=0)[0]

    # Table 1b: Standardised Regression Coefficients Table
    std_coefficients_table = pd.read_html(
        std_model.summary().tables[1].as_html(), header=0, index_col=0)[0]

    # Add standardised coefficients to the coefficicnets table
    coefficients_table.insert(2, 'Std Coef(β)', std_coefficients_table['coef'])

    # Rename the 'coef' column to 'unstd coef' to avoid confusion
    coefficients_table = coefficients_table.rename(columns={
        'coef': 'Unstd Coef(b)',
        'std err': 'SE',
        'P>|t|': 'p',
        '[0.025': 'CI Lower',
        '0.975]': 'CI Upper'
    })

    # Table 2: Regression Summary Table
    summary_table = pd.read_html(
        model.summary().tables[0].as_html(), header=None, index_col=None)[0]
    summary_table.columns = ['Description', 'Value', 'Description', 'Value']

    # Table 3: Regression Diagnostics Table
    diagnostics_table = pd.read_html(
        model.summary().tables[2].as_html(), header=None, index_col=None)[0]
    diagnostics_table.columns = ['Statistic', 'Value', 'Statistic', 'Value']

    # Table 4: overall model fit
    fvalue = model.fvalue.round(3)
    pvalue = model.f_pvalue.round(3)
    df_model = int(model.df_model)
    df_resid = int(model.df_resid)

    r_squared = model.rsquared.round(3)
    r = np.sqrt(r_squared).round(3)
    r_squared_adj = model.rsquared_adj.round(3)

    overall_model_fit = [['R', 'R-squared', 'Adj. R-squared', 'F-statistic', 'p-value', 'df-model', 'df-resid'],
                         [r, r_squared, r_squared_adj, fvalue, pvalue, df_model, df_resid]]
    overall_model_fit = pd.DataFrame(
        overall_model_fit[1:], columns=overall_model_fit[0])

    results = {}
    results['overall_model_fit'] = overall_model_fit
    results['summary_table'] = summary_table.round(3)
    results['coefficients_table'] = coefficients_table.round(3)
    results['diagnostics_table'] = diagnostics_table.round(3)
    return results
