"""
Author:         Victor Loveday
Date:           01/08/20220   
"""

import numpy as np
import pandas as pd

from src.data import LabelManager
from src.pipeline import PreprocessPipelineFactory
from src.utils import refactor_names, refactor_byte_name


class Preprocess:

    def __init__(self):
        self._signature_keys = None

    def X_pre_process(self, X, pipeline_factory: PreprocessPipelineFactory, **kwargs):
        """
        Perform pre-processing on X.

        :param X: The set of input features.
        :param pipeline_factory:  The pipeline factory to obtain the X pre-processing pipeline from.
        :param kwargs: The kwargs for the pipeline to use.
        :return: The processed dataset X.
        """
        X_preprocess_pipeline = pipeline_factory.X_preprocess_pipeline(**kwargs)

        _X = X_preprocess_pipeline.fit_transform(X)

        names = X_preprocess_pipeline.get_feature_names_from_ohe_step()
        feature_names = refactor_names(names, kwargs["category_variables"])
        feature_names = np.append(feature_names, kwargs["numeric_variables"])

        _X = self._convert_to_array(_X)

        X = pd.DataFrame(data=_X, columns=feature_names)

        return X

    def y_pre_process(self, y, pipeline_factory: PreprocessPipelineFactory):
        """
        Perform pre-processing on y.

        :param X: The set of output labels.
        :param pipeline_factory:  The pipeline factory to obtain the y pre-processing pipeline from.
        :param kwargs: The kwargs for the pipeline to use.
        :return: The processed dataset X.
        """
        y_preprocess_pipeline = pipeline_factory.y_preprocess_pipeline()
        y = y_preprocess_pipeline.fit_transform(y)
        y = self._convert_to_array(y)
        y = y.ravel()
        y = pd.DataFrame(data=y, columns=["signature"])
        self._signature_keys = y_preprocess_pipeline.named_transformers_['lep'].named_steps["le"].classes_

        return y

    def X_y_pre_process(self, X, y, label_manager: LabelManager) -> tuple:
        """
        Helper method that manages the preprocessing of both X and y datasets and returns the post processed data.
        :param X: The input dataset of features.
        :param y: The output dataset of labels.
        :param label_manager: The label manager object from main.
        :return: (X, y) datasets after being processed.
        """
        pipeline_factory = PreprocessPipelineFactory()

        X = self.X_pre_process(X, pipeline_factory,
                               category_variables=label_manager.X_discrete,
                               numeric_variables=label_manager.X_continuous)
        y = self.y_pre_process(y, pipeline_factory)

        return X, y

    @staticmethod
    def _convert_to_array(dataset: pd.Series):
        """
        Attempts to convert a pandas Series object into a numpy array.

        :param dataset: A Series object to transform
        :return: The converted dataset.
        """
        if type(dataset) is not np.ndarray:
            dataset = dataset.toarray()

        return dataset

    @property
    def y_classes(self):
        """
        The y class names.
        :return:
        """
        return {key: refactor_byte_name(value) for key, value in enumerate(self._signature_keys)}
