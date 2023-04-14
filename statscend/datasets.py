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
        return sns.load_dataset('penguins')
    elif name == 'tips':
        return sns.load_dataset('tips')
    elif name == 'iris':
        return sns.load_dataset('iris')
    elif name == 'titanic':
        return sns.load_dataset('titanic')
    elif name == 'anscombe':
        return sns.load_dataset('anscombe')
    else:
        raise ValueError(
            f"Unsupported dataset name: {name}. Currently supported datasets are 'penguins', 'tips, 'iris', titanic, and 'anscombe'.")
