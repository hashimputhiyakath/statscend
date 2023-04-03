# StatScend

The **statscend** package is a Python library designed to provide a convenient and user-friendly interface for performing statistical analysis. Currently, the library includes a set of functions for fitting and evaluating regression models. More functions will be added soon.

The package is designed to be user-friendly, with intuitive syntax and clear output. It is intended for use by researchers, data scientists.
Under the hood, **statscend** utiize the `statsmodels` and `sklearn` libraries to provide a range of data analysis and modeling functions. Essentially, this library provides a high-level interface that abstracts away many of the complexities of working with statistical models, making it easy to get started with data analysis in Python.

## Installation

You can install this package using pip. Open a terminal or command prompt and type:

`pip install regression_analysis`

---

## Dependencies

- numpy
- pandas
- statsmodels
- sklearn

---

## License

This package is licensed under the MIT License.

---

## Linear Regression

Paragraph

`linear_regression(data, x, y)`

This function performs a linear regression analysis on the input dataset.

#### Parameters

| Parameter | Data type                 | Description                                                       |
| --------- | ------------------------- | ----------------------------------------------------------------- |
| `data`    | pandas DataFrame          | The input dataset.                                                |
| `x`       | string or list of strings | The name(s) of the column(s) to be used as predictor variable(s). |
| `y`       | string                    | The name of the column to be used as the response variable.       |

#### Returns

This function returns a dictionary of tables that summarize the results of a linear regression analysis. The tables included are

- **coefficients table:** shows the estimated coefficients and standard errors for each predictor variable,
- **summary table:** provides an overview of the regression results,
- **diagnostics table:** includes various diagnostic measures such as the residuals and leverage values.

These tables can be used to evaluate the fit of the regression model and to identify any potential issues such as outliers or multicollinearity.

### Exmaple

---

## Logistic Regression

Paragraph

`logistic_regression(data, x, y)`

This function performs a logistic regression analysis on the input dataset.

#### Parameters

| Parameter | Data type                 | Description                                                       |
| --------- | ------------------------- | ----------------------------------------------------------------- |
| `data`    | pandas DataFrame          | The input dataset.                                                |
| `x`       | string or list of strings | The name(s) of the column(s) to be used as predictor variable(s). |
| `y`       | string                    | The name of the column to be used as the response variable.       |

#### Returns

This function returns a dictionary of tables that summarize the results of a linear regression analysis. The tables included are

- **summary_table:** Logistic Regression Summary Table. This table provides information on the model's goodness of fit, including the number of observations, the model's Log-Likelihood value, and the Wald Chi-Square test statistic.
- **coefficients_table:** provides information on the regression coefficients of the predictors used in the logistic regression model. This table includes columns for the predictor variable name, unstandardized coefficient estimates, standardized coefficient estimates, standard error, z-value, p-value, confidence interval lower bound, confidence interval upper bound, and odds ratio.

- **predictive_measures_table:** provides information on the predictive measures of the logistic regression model. This table includes columns for the accuracy, specificity, and sensitivity of the model.
- **classification_table:** provides information on the classification accuracy of the logistic regression model. This table includes the number of true positives, true negatives, false positives, and false negatives, as well as the percentage of correct classifications for each category.
