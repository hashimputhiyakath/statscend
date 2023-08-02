from statsmodels.multivariate.manova import MANOVA
import pandas as pd


def manova(df, dvs, iv):

    # Subset the DataFrame with only the variables we need
    df = df[[iv] + dvs]

    df.dropna(inplace=True)

    # Run MANOVA
    maov = MANOVA.from_formula(f'{"+".join(dvs)} ~ {iv}', data=df)
    manova_result = maov.mv_test()

    # Create dataframes from the result
    intercept_result = manova_result['Intercept']['stat']
    iv_result = manova_result[iv]['stat']

    manova_table_intercept = pd.DataFrame(intercept_result)
    manova_table_iv = pd.DataFrame(iv_result)

    # Return as dictionary
    return {'intercept_summary': intercept_result, 'iv_summary': iv_result}
