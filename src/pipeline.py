"""
Author:         David Walshe
Date:           08/04/2020   
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImblearnPipeline
from imblearn.under_sampling import NearMiss
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

from src.config import RANDOM_STATE

if TYPE_CHECKING:
    import numpy as np
    from pandas import DataFrame


class CustomColumnTransformer(ColumnTransformer):
    """
    Custom ColumnTransformer to allow easy feature extraction
    """

    def get_feature_names_from_ohe_step(self) -> np.ndarray:
        """
        Helper method to access internal step feature names.

        :return: The feature names after the OneHotEncoder step of the Pipeline.
        """
        return self.named_transformers_['cp'].named_steps["ohe"].get_feature_names()


class PipelineLabelEncoder(TransformerMixin):
    """
    Custom LabelEncoder to allow for passing of X and y datasets.

    Default LabelEncoder only accepts one dataset by default and is not
    suitable for Pipeline usage.
    """

    def __init__(self):
        """
        Class Constructor
        """
        self.encoder = LabelEncoder()

    def fit(self, X: DataFrame, y: DataFrame = None):
        """
        Fit the dataset X to the encoder.

        :param X: Passed dataset to be encoded.
        :param y: Dummy variable included to allow for Pipeline usage.
        :return: This instance.
        """
        self.encoder.fit(X)
        return self

    def transform(self, X: DataFrame, y: DataFrame = None) -> np.ndarray:
        """
        Apply the LabelEncoder transformation to the dataset X.

        :param X: The dataset to encode.
        :param y: Dummy variable included to allow for Pipeline usage.
        :return: A numpy ndarray of the applied transformation.
        """
        return self.encoder.transform(X).reshape(-1, 1)

    @property
    def classes_(self):
        return self.encoder.classes_


class PreprocessPipelineFactory:
    """
    Method Factory Class to help with the creation of various pre-processing Pipelines.
    """

    def X_preprocess_pipeline(self, category_variables: list, numeric_variables: list) -> CustomColumnTransformer:
        """
        Creates a pre-processing pipeline targeted at the X segment for the KDD cup99 dataset.

        :param category_variables: The categorical variable names of the dataset X.
        :param numeric_variables: The numerical variable names of the dataset X.
        :return: A CustomColumnTransformer instance to pre-process the X dataset.
        """
        return CustomColumnTransformer(
            transformers=[
                ("cp", self._category_step, category_variables),
                ("sp", self._scaler_step, numeric_variables)
            ],
            remainder="drop",
            n_jobs=-1
        )

    def y_preprocess_pipeline(self, variables: tuple = (0,)):
        """
        Creates a pre-processing pipeline targeted at the y segment for the KDD cup99 dataset.

        :param variables: Optional argument to pass in column indexes to use in the pipeline. Default= (0, )
        :return: A CustomColumnTransformer instance to pre-process the X dataset.
        """
        return ColumnTransformer(
            transformers=[
                ("lep", self._label_encoder_step, variables)
            ],
            remainder="drop",
            n_jobs=-1
        )

    @property
    def _category_step(self) -> SklearnPipeline:
        """
        Property to get the category step for use in a Pipeline.

        :return: Pipeline with an OneHotEncoder internal step.
        """
        return SklearnPipeline([
            ("ohe", OneHotEncoder())
        ])

    @property
    def _scaler_step(self) -> SklearnPipeline:
        """
        Property to get the scaler step for use in a Pipeline.

        :return: Pipeline with an StandardScaler internal step.
        """
        return SklearnPipeline([
            ("ss", StandardScaler())
        ])

    @property
    def _label_encoder_step(self) -> SklearnPipeline:
        """
        Property to get the encoder step for use in a Pipeline.

        :return: Pipeline with a LabelEncoder internal step.
        """
        return SklearnPipeline([
            ("le", PipelineLabelEncoder())
        ])


class SamplingPipelineFactory:

    def __init__(self, y, max_sample_limit=1000, k_neighbors=5):
        """
        Class Constructor.
        
        :param y: The y dataset of output labels. 
        :param max_sample_limit: The upper limit of samples.
        :param k_neighbors: The number of neigbors to use in SMOTE and near miss.
        """
        self.k_neighbors = k_neighbors
        self.under_sampling_strategy = {key: max_sample_limit for key, value in Counter(y["signature"]).items() if
                                        value > max_sample_limit}
        self.ros_sampling_strategy = {key: k_neighbors * 20 for key, value in Counter(y["signature"]).items() if
                                      value <= k_neighbors}

    def sampling_pipeline(self) -> ImblearnPipeline:
        """
        Creates and returns a sampling Pipeline.
        :return: A constructed sampling pipeline.
        """
        return ImblearnPipeline(
            steps=[
                ("ros", self.random_over_sampling_step),
                ("nm", self.near_miss_step),
                ("smt", self.smote_pipeline_step),
            ]
        )

    @property
    def random_over_sampling_step(self) -> RandomOverSampler:
        """
        Creates a RandomOverSammpling Step for a sampling pipeline.
        
        :return: A RandomOverSampler for use in a sampling pipeline. 
        """
        return RandomOverSampler(sampling_strategy=self.ros_sampling_strategy,
                                 random_state=RANDOM_STATE)

    @property
    def near_miss_step(self) -> NearMiss:
        """
        Creates a NearMiss Version 1 Step for Undersampling the majority classes for a sampling pipeline.
        
        :return: A NearMiss1 object for usin a sampling pipeline.
        """
        return NearMiss(sampling_strategy=self.under_sampling_strategy,
                        n_neighbors=self.k_neighbors,
                        n_jobs=-1)

    @property
    def smote_pipeline_step(self) -> SMOTE:
        """
        Creates a SMOTE Step for sample synthesis of the minority classes for a sampling pipeline.

        :return: A SMOTE object for usin a sampling pipeline.
        """
        return SMOTE(k_neighbors=self.k_neighbors,
                     random_state=RANDOM_STATE,
                     n_jobs=-1)
