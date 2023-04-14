import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel


def ordinal_regression(data, x, y, distr='logit'):
    if not pd.api.types.is_categorical_dtype(data[y]):
        raise ValueError(
            f"Endogenous variable {y} is not categorical. Please convert the dtype of data['{y}'] to categorical. Please also ensure that the order of levels is correct.")
    mod = OrderedModel(data[y], data[x], distr=distr)
    method = 'bfgs'
    res = mod.fit(method=method)

    # create an empty dictionary
    results = {}

    # extract the first table and save it to the dictionary
    summary_table = res.summary().tables[0]
    results['summary_table'] = summary_table

    # extract the second table and save it to the dictionary
    coefficients_table = pd.read_html(
        res.summary().tables[1].as_html(), header=0, index_col=0)[0]

    # Extract the coefficients for the independent variables
    num_ivs = mod.exog.shape[1]
    model_coefficients = coefficients_table.head(num_ivs)

    results['model_coefficients'] = model_coefficients

    # Extract the coefficients for the dependent variable levels
    num_of_thresholds = len(data[y].unique()) - 1
    model_thresholds = coefficients_table.tail(num_of_thresholds)

    # model_thresholds is the last rows

    # Get actual threshold values
    actual_thresholds = mod.transform_threshold_params(
        res.params[-num_of_thresholds:])
    actual_thresholds = actual_thresholds[1:-1]
    actual_thresholds = pd.Series(actual_thresholds)
    actual_thresholds = list(actual_thresholds)

    # Add actual threshold estimates to model_thresholds
    model_thresholds = model_thresholds.copy()
    model_thresholds.loc[:, 'Actual Estimates'] = actual_thresholds
    results['model_thresholds'] = model_thresholds

    return results
