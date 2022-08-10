"""
Author:         Victor Loveday
Date:           01/08/2022  
"""

# Imports
import logging
from json import load

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_kddcup99

from src.config import RANDOM_STATE
from src.logger_config import setup_logger

logger = setup_logger(logging.getLogger(__name__))


class LabelManager:
    """
    Helper class to read in discrete and continuous identifiers for data.
    """

    def __init__(self, config_file="data.json"):
        """
        Class constructor.

        :param config_file: The file path to read the data configuration file form.
        """
        self._data_info = self._read_data_info(config_file=config_file)
        self._X_column_names = None
        self._y_column_name = None

    @staticmethod
    def _read_data_info(config_file) -> dict:
        """
        Helper method to read the contents of the config file.

        :param config_file: The configuration file path.
        :return: The json data within the file as a dictionary.
        """
        with open(config_file) as fh:
            data_info = load(fp=fh)

        return data_info

    @property
    def info(self):
        """
        :return: The configuration dictionary.
        """
        return self._data_info

    def _get_column_names(self, key: str) -> list:
        """
        Get the names of all features within the dataset X or y as a list.

        :param key: Feature set names to get, options: "X" or "y"
        :return: A list of all the names in that feature set.
        """
        return [item.get("name") for item in self.info.get(key)]

    @property
    def X_column_names(self):
        """
        :return: All the feature names for the X feature set.
        """
        # Lazy Init
        if self._X_column_names is None:
            self._X_column_names = self._get_column_names(key="X")

        return self._X_column_names

    @property
    def y_column_name(self):
        """
        :return: All the feature names for the y feature set.
        """
        # Lazy Init
        if self._y_column_name is None:
            self._y_column_name = self._get_column_names(key="y")

        return self._y_column_name

    @property
    def X_y_column_names(self) -> tuple:
        """
        Helper function for acquiring the dataset column names from data.json.

        :return: The X columns names and the y column name as a tuple.
        """
        return self.X_column_names, self.y_column_name

    def get_variable_on_dtype(self, key: str, dtype: str) -> list:
        """
        Return a list of feature names depending if the dtype is continuous or categorical.

        :param key: The data set to search in. options: "X" or "y".
        :param dtype: The dtype to search for. options: "discrete" or "continuous".
        :return: A list feature names that contain data of the dtype passed.
        """
        return [item.get("name") for item in self.info.get(key) if item.get("dtype") == dtype]

    @property
    def X_discrete(self):
        """
        :return: Returns feature names from X that have the "discrete" dtype.
        """
        return self.get_variable_on_dtype(key="X", dtype="discrete")

    @property
    def X_continuous(self):
        """
        :return: Returns feature names from X that have the "continuous" dtype.
        """
        return self.get_variable_on_dtype(key="X", dtype="continuous")


class DataRetriever:

    def __init__(self, label_manager: LabelManager):
        self.label_manager = label_manager
        self._X = None
        self._y = None

    def _remove_duplicate_rows(self):
        """
        Helper function to remove duplicates for the dataset.
        """
        # Merge (X, y) before reduction.
        dataset = pd.concat([self.X, self.y], axis=1, join="outer")
        orig_size = dataset.shape[0]
        logger.info(f"Step  - Original dataset record count: {orig_size}")

        # Reduce the merge dataset by removing duplicates.
        dataset.drop_duplicates(inplace=True)
        logger.info(f"Step  - Dataset record count with duplicates removed: {dataset.shape[0]}")
        logger.info(
            f"Step  - Dataset records reduced by {round(100 - ((dataset.shape[0] / orig_size) * 100), 2)}%")

        # Reassign X and y with the reduced dataset.
        self._X = pd.DataFrame(data=dataset.iloc[:, :-1].values, columns=self.label_manager.X_column_names)

        self._y = pd.DataFrame(data=dataset.iloc[:, -1].values.reshape(-1, 1),
                               columns=self.label_manager.y_column_name)

    def X_y_dataset(self, remove_duplicates: bool = False, full_dataset: bool = True, force: bool = False) -> np.array:
        """
        Helper function to create the dataset, including the dependant "target" variable.

        :param remove_duplicates: Flag to decide whether duplicates should be reduced using Dataframe.drop_duplicates
        :param full_dataset: Flag to decide if full dataset or only 10% should be retrieved.
        :param force: Flag to force re-retrieval of X and y from source or used locally stored (X, y) from previous call.
        :return: The dataset as (X, y).
        """
        # Lazy init
        if self._X is None or self._y is None or force is True:

            logger.info(f"Step  - Only 10% of Dataset: {(not full_dataset)}")
            data, target = fetch_kddcup99(return_X_y=True, percent10=(not full_dataset), random_state=RANDOM_STATE)

            target = np.array(target).reshape(-1, 1)

            self._X = pd.DataFrame(data=data, columns=self.label_manager.X_column_names)
            self._y = pd.DataFrame(data=target, columns=self.label_manager.y_column_name)

            if remove_duplicates:
                self._remove_duplicate_rows()

        return self._X, self._y

    @property
    def X(self):
        """
        Returns X features set. Builds it using lazy initialisation if it has not already been assigned.
        :return: Feature set X
        """
        # Lazy init
        if self._X is None:
            self.X_y_dataset()

        return self._X

    @property
    def y(self):
        """
        Returns y features set. Builds it using lazy initialisation if it has not already been assigned.
        :return: Feature set y
        """
        # Lazy init
        if self._y is None:
            self.X_y_dataset()

        return self._y
