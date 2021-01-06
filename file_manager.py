import pandas as pd
import config


def read_dataset():
    return pd.read_csv(config.file_name, delimiter=config.file_delimiter)


def write_to_csv(dataframe, filename):
    dataframe.to_csv(filename)


def remove_instances_with_missing_values(dataset):
    return dataset.drop(dataset[dataset.eq(config.missing_value).any(1)].index)
