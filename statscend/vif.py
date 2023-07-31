import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def vif(df):
    '''
    This function takes a dataframe with numeric predictor variables and returns a new dataframe with
    a Variance Inflation Factor (VIF) for each variable. 
    '''
    # Create a new DataFrame for VIF
    vif_data = pd.DataFrame()

    # Add the column names to the new DataFrame
    vif_data["feature"] = df.columns

    # Calculate VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(
        df.values, i) for i in range(len(df.columns))]

    return vif_data
