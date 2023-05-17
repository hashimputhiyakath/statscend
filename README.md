# StatScend

The **statscend** package is a Python library designed to provide a convenient and user-friendly interface for performing statistical analysis. Currently, the library includes a set of functions for fitting and evaluating regression models. More functions will be added soon.

The package is designed to be user-friendly, with intuitive syntax and clear output. It is intended for use by researchers, data scientists.
Under the hood, **statscend** utiize the `statsmodels` and `sklearn` libraries to provide a range of data analysis and modeling functions. Essentially, this library provides a high-level interface that abstracts away many of the complexities of working with statistical models, making it easy to get started with data analysis in Python.

## Installation

You can install Statscent using pip. First, make sure you have Python 3.x and pip installed on your system. Then, open a terminal window and run the following command:

`pip install statscend`

This will download and install the latest version of the Statscent package and its dependencies. If you prefer to install a specific version of the package, you can specify the version number using pip:

`pip install statscent==1.0.0`

Once you've installed Statscent, you can import it in your Python code using the following statement:

`import statscent`

However, since the package name is quite long, you may want to use a shorter alias for convenience. We suggest using st as the alias, since this follows the same naming convention used by other popular data analysis libraries like pandas (pd) and numpy (np).

Here's an example of how to import Statscent with the `ss` alias:

`import statscent as ss`

This will allow you to use the Statscent package in your code using the shorter st alias, like so:

`ss.linear_regression(data, x y)`

While you are free to use any alias you prefer, we believe that using `ss` will help to ensure a smooth and consistent experience for all Statscent users.

---

## Dependencies

- numpy
- pandas
- statsmodels
- sklearn
- seaborn

---

## License

This package is licensed under the MIT License.

---

## Functions

<details>
<summary>linear_regression()</summary>
The `linear_regression()` function computes a linear regression on one or more predictor variables against a response variable in a given dataset. The function uses the statsmodels package to perform the linear regression analysis and returns a dictionary of four dataframes containing overall model fit measures, regression coefficients, regression summary, and regression diagnostics.

#### Parameters:

- data : pandas DataFrame<br/>
  The data on which to perform the regression.

- x : str or list of str<br/>
  The name(s) of the predictor variable(s) in the dataset. If performing multiple regression, x should be a list of variable names.

- y : str<br/>
  The name of the response variable in the dataset.

#### Returns

A dictionary with four pandas DataFrames containing the results of the regression:

- overall_model_fit: Pandas DataFrame<br/>
  A table that includes the overall model fit statistics such as R, R-squared, Adj. R-squared, F-statistic, p-value, df-model and df-resid.

- coefficients_table : Pandas DataFrame<br/>
  A table of the regression coefficients, including the unstandardized coefficients, their standard errors, the standardized coefficients, and the p-values.

- diagnostics_table : Pandas DataFrame<br/>
  A table of diagnostic statistics for the regression, including the Omnibus test, the Durbin-Watson statistic, the Jarque-Bera test, and the condition number.
- residuals_table : Pandas DataFrame <br/>
  A table containing the fitted values and residuals of the regression model. The "Fitted Values" column represents the predicted values obtained from the regression model for the corresponding input variables. The "Residuals" column represents the difference between the actual values and the predicted values, indicating the deviation or error in the model's predictions.

#### Examples

Simple Linear Regression

`result = linear_regression(data=penguins, x='bill_depth_mm', y='body_mass_g')`

In this example, the function is used to perform a simple linear regression, with `bill_depth_mm` as the predictor variable and `body_mass_g` as the response variable. The results are stored in the `result` dictionary.

Multiple Linear Regression

`result = linear_regression(data=penguins, x=['bill_depth_mm', 'flipper_length_mm'], y='body_mass_g')`

In this example, the function is used to perform a multiple linear regression, with both `bill_depth_mm` and `flipper_length_mm` as predictor variables and `body_mass_g` as the response variable. The results are stored in the `result` dictionary.

#### Notes

- This function requires the following packages to be installed: numpy, pandas, and statsmodels. If these packages are not already installed, you can install them using pip: <br/>

  `pip install numpy pandas statsmodels`

- This function uses the ordinary least squares (OLS) method to estimate the regression coefficients.

- This function assumes that the predictor variables are not correlated with each other.

- This function removes any rows from the dataset that contain missing values before performing the regression analysis.

</details>

<details>
<summary>bi_logistic_regression()</summary>

### bi_logistic_regression(data, x, y, y_dummy=False)

The `bi_logistic_regression()` function is designed to perform binomial logistic regression with on one or more predictor variables against a response variable in a given dataset. The function uses the statsmodels package to perform the linear regression analysis and returns a dictionary of four dataframes: overall model fit measures, coefficients table, predictive measures table, and classification table, and summary table

#### Parameters:

- data: Pandas dataframe<br/>
  A Pandas DataFrame containing the predictor and outcome variables
- x: str or list of str<br/>
  A column name or list of column names in the DataFrame that correspond to the predictor variables.
- y: str <br/>
  The name of the column in the DataFrame that represents the binary outcome variable. The values in the column should be binary and contain only two unique values. There is no need for the column to be dummy-coded. If the column is already dummy-coded, you can set the `y_dummy` parameter to True to indicate that the values in the column are dummy-coded.

- y_dummy: [optional] boolean <br/>
  It indicates whether the outcome variable is already dummy-coded. If y is already dummy-coded, set `y_dummy` parameter to `True` (default is `False`).

#### Returns

- overall_model_test: a DataFrame that includes different goodness-of-fit measures such as the deviance, degrees of freedom, and p-value for the overall model.
- coefficients_table: a DataFrame that displays the estimated coefficients and corresponding odds ratios for each predictor variable.
- predictive_measures_table: a DataFrame that shows the accuracy, specificity, and sensitivity of the model, along with other predictive measures such as the area under the receiver operating characteristic curve (AUC-ROC).
- classification_table: a DataFrame that provides the confusion matrix and percentage of correct predictions for the model, as well as other performance metrics such as the positive predictive value (PPV) and negative predictive value (NPV).
- summary_table: a DataFrame that summarizes the distribution of the predictor and outcome variables, including the count, mean, standard deviation, minimum, and maximum values.

The output of the function can be saved to a variable. This variable will contain a dictionary with four tables.

`results = bn_logistic_regression(data=adelie_chinstrap, x='bill_depth_mm', y='species', y_dummy=False)`

To access the dictionary keys, you can use the keys() method.
`print(results.keys())`

In Jupyter Notebook, you don't need to use print()
`(results.keys()`

Once you have the dictionary keys, you can display each table by using the key to index into the dictionary. Here's an example:

`results['overall_model_test']`

`results['coefficients_table']`

`results['predictive_measures_table']`

`results['classification_table']`

`results['summary_table']`

#### Examples

```

x = ['bill_length_mm', 'bill_depth_mm]
y = 'species'

result = bn_logistic_regression(data=penguins, x=x, y=y, y_dummy=False)


print(result['overall_model_test'])


```

#### Notes

</details>

<details>

<summary>ordinal_regression()</summary>

### ordinal_regression(data, x, y, distr='logit')

The `bi_logistic_regression()` function perform ordinal logistic regression on the specified data.

#### Parameters:

- data : pandas DataFrame<br/>
  The dataset to use for the regression. It must contain the dependent variable (y) and at least one independent variable (x).
- x : str<br/>
  The name of the column in the `data` DataFrame containing the independent variable(s).
- y : str<br/>
  The name of the column in the `data` DataFrame containing the dependent variable. The `y` variable must be a pandas categorical variable with an ordered category. If the y variable is not a categorical variable or if the category is not ordered, a ValueError will be raised.
- distr : str, optional<br/>
  The distribution to use for the model. Supported values are `probit` or `logit`. Default is `logit`.

#### Returns

The ordinal_logistic_regression function returns a dictionary containing three keys:

- `summary_table`: This key contains a summary table for the regression. The summary table provides information such as the coefficients for the independent variables, the coefficients for the thresholds (i.e., the values that separate the different levels of the dependent variable), the standard errors for the coefficients, the z-scores, and the p-values.

- `model_coefficients`: This key contains the coefficients for the independent variables in the regression. These coefficients represent the estimated effect of each independent variable on the dependent variable, while controlling for the other variables in the model.

-`model_thresholds`: This key contains the coefficients for the dependent variable levels, along with their standard errors, z-scores, and p-values. These coefficients represent the values that separate the different levels of the dependent variable, and are specific to the distribution that was used in the model (`probit` or `logit`).

The output of the function can be saved to a variable. This variable will contain a dictionary with three tables.

`result = ss.ordinal_regression(data=penguins, x='bill_length_mm', y='species')`

To access the dictionary keys, you can use the keys() method.
`print(results.keys())`

In Jupyter Notebook, you don't need to use print()
`(results.keys()`

Once you have the dictionary keys, you can display each table by using the key to index into the dictionary. Here's an example:

`print(result['summary_table'])`

`print(result['model_coefficients'])`

`print(result['model_thresholds'])`

#### Examples

```

species_order = ['Adelie', 'Chinstrap', 'Gentoo']

penguins['species'] = pd.Categorical(penguins['species'], categories=species_order, ordered=True)

x = ['bill_length_mm', 'bill_depth_mm]
y = 'species'

result = ordinal_regression(data=penguins, x=x, y=y)

print(result['model_coefficients'])

```

</details>

<details>
<summary>mn_logistic_regression()</summary>

### mn_logistic_regression(data, x, y)

The `mn_logistic_regression` function performs multinomial logistic regression analysis on a given dataset.

#### Parameters:

- data: Pandas DataFrame <br/>
  A Pandas DataFrame containing the data to be analyzed. This data should include both the independent variable(s) and the dependent variable.
- x: str or list of str<br/>
  The x argument specifies the column name or names of the independent variable(s) in the data DataFrame. It can be a single column name or a list of column names representing multiple independent variables.
- y: str <br/>
  The column name of the dependent variable in the data DataFrame. The y argument is required and must be specified.

#### Returns

The mn_logistic_regression function returns a dictionary containing two keys:

- summary_table: a Pandas DataFrame containing the summary statistics for the analysis. The table includes information such as the number of observations, the model used, and the log-likelihood of the model.
- model_coefficients: a Pandas DataFrame containing the coefficients for the analysis. The table includes information such as the variable name, the coefficient value, standard error, z-score, p-value, and confidence intervals.

The output of the function can be saved to a variable. This variable will contain a dictionary with two tables.

`result = ss.mn_logistic_regression(data=penguins, x='bill_length_mm', y='species')`

To access the dictionary keys, you can use the keys() method.
`print(results.keys())`

In Jupyter Notebook, you don't need to use print()
`(results.keys()`

Once you have the dictionary keys, you can display each table by using the key to index into the dictionary. Here's an example:

`print(result['summary_table'])`

`print(result['model_coefficients'])`

#### Examples

```

x = ['bill_length_mm', 'bill_depth_mm]
y = 'species'

result = mn_logistic_regression(data=penguins, x=x, y=y)

print(result['model_coefficients'])

```

</details>
