import statsmodels.api as sm
import pandas as pd
import numpy as np


def multi_logistic_regression(data, x, y):
    # Define dependent and independent variables
    y_var = data[y]
    x_vars = data[x]
    x_vars = sm.add_constant(x_vars)

    # Fit multinomial logistic regression model
    model = sm.MNLogit(y_var, x_vars).fit()

    # Create Summary Table
    summary_table = pd.read_html(
        model.summary().tables[0].as_html(), header=None, index_col=None)[0]
    summary_table.columns = ['Description', 'Value', 'Description', 'Value']

    # Create Coefficients table
    coefficients_table = pd.read_html(
        model.summary().tables[1].as_html(), header=None, index_col=None)[0]
    coefficients_table.columns = [
        'variable', 'coef', 'std err', 'z', 'P>|z|', '0.025', '0.975']

    coefficients_table = coefficients_table[~coefficients_table['coef'].astype(
        str).str.contains('coef')]

    coefficients_tables = []
    temp = pd.DataFrame()
    for i, row in coefficients_table.iterrows():
        if row['variable'] == 'const':
            if not temp.empty:
                coefficients_tables.append(temp)
                temp = pd.DataFrame()
            temp = pd.concat([temp, row.to_frame().T])
        else:
            temp = pd.concat([temp, row.to_frame().T])
    coefficients_tables.append(temp)

    for i in coefficients_tables:
        i.set_index(["variable"], inplace=True)

    dv_levels = np.sort(y_var.unique())[1:]
    dv_levels = pd.Series(dv_levels)

    coefficients_table = pd.concat(
        [df for df in coefficients_tables], keys=dv_levels)

    return {'summary_table': summary_table, 'coefficients_table': coefficients_table}
