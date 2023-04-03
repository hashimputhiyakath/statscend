import numpy as np
import statsmodels.api as sm
import pandas as pd
from sklearn.metrics import confusion_matrix


def logistic_regression(data, x, y, y_dummy=False):

    data.dropna(inplace=True)
    x = data[x]
    x = sm.add_constant(x)

    if y_dummy == False:
        y = pd.get_dummies(data[y], drop_first=True,  prefix=y)
        # pd.concat
        pd.concat([data, y], axis=1)
    else:
        y = data[y]

    # Fit logistic regression model
    model = sm.Logit(y, x).fit()

    summary_table = pd.read_html(
        model.summary().tables[0].as_html(), header=None, index_col=None)[0]
    summary_table.columns = ['Description', 'Value', 'Description', 'Value']

    coefficients_table = pd.read_html(
        model.summary().tables[1].as_html(), header=0, index_col=None)[0]
    coefficients_table.columns = [
        'Predictors', 'Unstd Coef', 'SE', 'Z', 'P', 'CI Lower', 'CI Upper']
    coefficients_table['Odds Ratio'] = np.exp(
        coefficients_table['Unstd Coef']).round(2)

    # X_test = sm.add_constant(X)  # add constant term to test data
    x_test = x  # add constant term to test data
    y_pred = model.predict(x_test)

    # Make predictions on test data to creat classification table
    y_pred_binary = (y_pred > 0.5).astype(int)

    cm = confusion_matrix(y, y_pred_binary)

    # Make classification table
    cmdf = pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'], columns=[
                        'Predicted Negative', 'Predicted Positive'])

    cmdf['%Correct'] = ""
    cmdf.iloc[0, 2] = (
        (cmdf.iloc[0, 0]/(cmdf.iloc[0, 0] + cmdf.iloc[0, 1])) * 100).round(3)
    cmdf.iloc[1, 2] = (
        (cmdf.iloc[1, 1]/(cmdf.iloc[1, 0] + cmdf.iloc[1, 1])) * 100).round(3)

    # Calculate predictive measures from classification table
    accuracy = ((cmdf['%Correct'].mean())/100).round(3)
    specificity = ((cmdf['%Correct'][0])/100).round(3)
    sensitivity = ((cmdf['%Correct'][1])/100).round(3)

    predictive_measures = {
        'Accuracy': [accuracy],
        'Specificity': [specificity],
        'Sensitivity': [sensitivity]
    }

    predictive_measures_table = pd.DataFrame(predictive_measures)

    results_dict = {
        "summary_table": summary_table,
        "coefficients_table": coefficients_table,
        "predictive_measures_table": predictive_measures_table,
        "classification_table": cmdf
    }

    return results_dict
