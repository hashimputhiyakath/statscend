import seaborn as sns


def data(name):
    """
    Load a common open-source dataset.

    Parameters
    ----------
    name : str
        Name of the dataset to load. Currently supported datasets are 'penguins' and 'tips'.

    Returns
    -------
    pandas.DataFrame
        The loaded dataset as a Pandas dataframe.

    Raises
    ------
    ValueError
        If the provided dataset name is not supported.
    """
    if name == 'penguins':
        penguins = sns.load_dataset('penguins')
        penguins['body_mass_g'] = penguins['body_mass_g']/1000
        penguins.rename(columns={
            'bill_length_mm': 'bill_length',
            'bill_depth_mm': 'bill_depth',
            'flipper_length_mm': 'flipper_length',
            'body_mass_g': 'body_mass'
        }, inplace=True)
        return penguins

    elif name == 'tips':
        return sns.load_dataset('tips')
    else:
        raise ValueError(
            f"Unsupported dataset name: {name}. Currently supported datasets are 'penguins' and 'tips'.")
