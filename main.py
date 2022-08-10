"""
Author:     David Walshe
Date:       03/04/2020
"""
import logging
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Pre-processing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import src.plotting as myPlt
from src.config import RANDOM_STATE, FULL_DATA_SET, TUNING
from src.data import DataRetriever, LabelManager
from src.evaluate import ModelEvaluator, ModelTuner
from src.logger_config import setup_logger
from src.pipeline import SamplingPipelineFactory
from src.preprocess import Preprocess
from src.timer import Timer

mpl.rcParams['figure.dpi'] = 200

logger = setup_logger(logging.getLogger(__name__))


def get_y_distribution(y):
    distribution = pd.DataFrame(columns=["label", "class", "count", "percentage"])
    counter = Counter(y["signature"])
    for index, (k, v) in enumerate(counter.items()):
        per = v / len(y) * 100
        distribution.loc[index] = {"label": k, "class": preprocess.y_classes.get(k, k), "count": v, "percentage": per}

    return distribution.sort_values("percentage", ascending=False)


def print_optimiser_results(model_tuner: ModelTuner):
    print(model_tuner.best_parameters)
    print(model_tuner.best_score)
    print(model_tuner.cv_results)


if __name__ == '__main__':
    logger.info("Start")

    # ==============================================================================================================
    # ==============================================================================================================
    # Setup
    # ==============================================================================================================
    # ==============================================================================================================
    timer = Timer()
    logger.info("Stage - Data Retrieval BEGIN")
    label_manager = LabelManager(config_file="data.json")
    data_retriever = DataRetriever(label_manager=label_manager)

    # ==============================================================================================================
    # ==============================================================================================================
    # Get Raw Data.
    # ==============================================================================================================
    # ==============================================================================================================
    logger.info("Stage - BEGIN")
    raw_X, raw_y = data_retriever.X_y_dataset(remove_duplicates=False, full_dataset=FULL_DATA_SET)
    logger.info(f"Stage - Data Retrieval END {timer.time_stage('Data Retrieval')}")

    myPlt.plot_value_counts(raw_y, title="y Distribution (Raw)")

    # ==============================================================================================================
    # ==============================================================================================================
    # Get Raw Data with duplicates removed
    # ==============================================================================================================
    # ==============================================================================================================
    logger.info("Stage - BEGIN")
    X, y = data_retriever.X_y_dataset(remove_duplicates=True, full_dataset=FULL_DATA_SET, force=True)
    logger.info(f"Stage - Data Retrieval END {timer.time_stage('Data Retrieval')}")

    myPlt.plot_value_counts(y, title="y Distribution (Duplicates removed)")

    myPlt.plot_value_counts_compare(y1=raw_y, y2=y)
    myPlt.plot_value_counts_compare(y1=raw_y, y2=y, level="Max")
    myPlt.plot_value_counts_compare(y1=raw_y, y2=y, level="Mid")
    myPlt.plot_value_counts_compare(y1=raw_y, y2=y, level="Min")

    # ==============================================================================================================
    # ==============================================================================================================
    # Preprocess raw data
    # ==============================================================================================================
    # ==============================================================================================================
    logger.info("Stage - Preprocess BEGIN")
    preprocess = Preprocess()
    X, y = preprocess.X_y_pre_process(X, y, label_manager)
    logger.info(f"Stage - Preprocess END {timer.time_stage('Preprocessing')}")

    # ==============================================================================================================
    # ==============================================================================================================
    # Principle Component Analysis
    # ==============================================================================================================
    # ==============================================================================================================
    logger.info("Stage - PCA BEGIN")
    X_backup = X

    # ==============================================================================================================
    # ==============================================================================================================
    # PCA with 3 Components
    # ==============================================================================================================
    # ==============================================================================================================
    logger.info("Stage - PCA 3 Component BEGIN")
    X = X_backup
    pca = PCA(n_components=3)
    X = pca.fit_transform(X)
    explained_variance_3d_df = pd.DataFrame(pca.explained_variance_ratio_.reshape(1, -1),
                                            columns=["X-axis", "Y-axis", "Z-axis"])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y["signature"], marker='o', cmap=plt.cm.get_cmap('tab20', 23))

    ax.set_title("PCA Analysis (3 Components)")
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')

    plt.show()
    logger.info(f"Step  - PCA 3 Component END {timer.time_stage('PCA 3C')}")

    # ==============================================================================================================
    # ==============================================================================================================
    # PCA with 2 Components
    # ==============================================================================================================
    # ==============================================================================================================
    logger.info("Stage - PCA 2 Component BEGIN")
    X = X_backup
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)
    explained_variance_2d_df = pd.DataFrame(pca.explained_variance_ratio_.reshape(1, -1), columns=["X-axis", "Y-axis"])

    myPlt.show_pca_plot(X, y, title="PCA Analysis (2 Components)")
    logger.info(f"Step  - PCA 2 Component END {timer.time_stage('PCA 2C')}")

    # ==============================================================================================================
    # ==============================================================================================================
    # PCA N-Component Selection
    # ==============================================================================================================
    # ==============================================================================================================
    logger.info("Step  - PCA Variance Graph BEGIN")
    X = X_backup
    pca = PCA().fit(X)
    pca_selection = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(pca_selection)
    plt.title("PCA Variance Graph")
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.show()
    logger.info(f"Step  - PCA Variance Graph {timer.time_stage('PCA Graph')}")

    # ==============================================================================================================
    # ==============================================================================================================
    # PCA with 20 Components
    # ==============================================================================================================
    # ==============================================================================================================
    logger.info("Stage - PCA 2 Component BEGIN")
    X = X_backup
    pca = PCA(n_components=20)
    X = pca.fit_transform(X)
    logger.info(f"Step  - PCA 2 Component END {timer.time_stage('PCA 2C')}")

    logger.info("Stage - PCA END")

    # ==============================================================================================================
    # ==============================================================================================================
    # Test/Train Split
    # ==============================================================================================================
    # ==============================================================================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.20, random_state=RANDOM_STATE
    )
    timer.time_stage("Train Test Split")

    # ==============================================================================================================
    # ==============================================================================================================
    # Sampling
    # ==============================================================================================================
    # ==============================================================================================================
    logger.info("Stage - Sampling BEGIN")
    if FULL_DATA_SET:
        max_sample_limit = 100_000
    else:
        max_sample_limit = 10_000
    sampling_pipeline = SamplingPipelineFactory(y_train, max_sample_limit=max_sample_limit).sampling_pipeline()

    prior_y_train = y_train
    X_train, y_train = sampling_pipeline.fit_resample(X_train, y_train)

    # Plot sampling vs original
    myPlt.plot_value_counts_compare(y1=prior_y_train, y2=y_train, title="Original Vs Sampled Training Sets (y)",
                                    label1="Original Data", label2="Sampled Data")

    logger.info(f"Stage - Sampling END {timer.time_stage('Sampling')}")

    if TUNING:
        # ==============================================================================================================
        # ==============================================================================================================
        # Hyper-parameter Tuning
        # ==============================================================================================================
        # ==============================================================================================================

        # Random Forest.
        # ===
        rf_params_estimators = [
            {
                "n_estimators": [100, 250, 500],
            }
        ]
        model_tuner_estimator = ModelTuner(y_classes=preprocess.y_classes, tuning_params=rf_params_estimators,
                                           model_type="RF")
        model_tuner_estimator.run_model_optimisation(X_train=X_train, y_train=y_train)
        print_optimiser_results(model_tuner_estimator)
        # --------------------------------------------------------------------------------------------------------------

        rf_params_criterion = [
            {
                "criterion": ["gini", "entropy"],
                "n_estimators": [model_tuner_estimator.best_parameters["n_estimators"]]
            }
        ]
        model_tuner_criterion = ModelTuner(y_classes=preprocess.y_classes, tuning_params=rf_params_criterion,
                                           model_type="RF")
        model_tuner_criterion.run_model_optimisation(X_train=X_train, y_train=y_train)
        print_optimiser_results(model_tuner_criterion)
        # --------------------------------------------------------------------------------------------------------------

        rf_params_max_features = [
            {
                "max_features": ["auto", "sqrt", "log2", None],
                "criterion": [model_tuner_criterion.best_parameters["criterion"]],
                "n_estimators": [model_tuner_criterion.best_parameters["n_estimators"]]
            }
        ]

        model_tuner_max_features = ModelTuner(y_classes=preprocess.y_classes, tuning_params=rf_params_max_features,
                                              model_type="RF")
        model_tuner_max_features.run_model_optimisation(X_train=X_train, y_train=y_train)
        print_optimiser_results(model_tuner_max_features)

        # --------------------------------------------------------------------------------------------------------------

        rf_params_oob = [
            {
                "oob_score": [True, False],
                "max_features": [model_tuner_max_features.best_parameters["max_features"]],
                "criterion": [model_tuner_max_features.best_parameters["criterion"]],
                "n_estimators": [model_tuner_max_features.best_parameters["n_estimators"]]
            }
        ]

        model_tuner_oob = ModelTuner(y_classes=preprocess.y_classes, tuning_params=rf_params_oob,
                                     model_type="RF")
        model_tuner_oob.run_model_optimisation(X_train=X_train, y_train=y_train)
        print_optimiser_results(model_tuner_oob)
        # --------------------------------------------------------------------------------------------------------------

        # KNN
        # ===
        knn_params_neigbors = [
            {
                "n_neighbors": [3, 5, 7, 10],
            }
        ]
        model_tuner_neighors = ModelTuner(y_classes=preprocess.y_classes, tuning_params=knn_params_neigbors,
                                          model_type="KNN")
        model_tuner_neighors.run_model_optimisation(X_train=X_train, y_train=y_train)
        print_optimiser_results(model_tuner_neighors)
        # --------------------------------------------------------------------------------------------------------------

        knn_params_p = [
            {
                "p": [1, 2],
                "n_neighbors": [model_tuner_neighors.best_parameters["n_neighbors"]]
            }
        ]
        model_tuner_p = ModelTuner(y_classes=preprocess.y_classes, tuning_params=knn_params_p, model_type="KNN")
        model_tuner_p.run_model_optimisation(X_train=X_train, y_train=y_train)
        print_optimiser_results(model_tuner_p)
        # --------------------------------------------------------------------------------------------------------------

        knn_params_weights = [
            {
                "weights": ["uniform", "distance"],
                "p": [model_tuner_p.best_parameters["p"]],
                "n_neighbors": [model_tuner_p.best_parameters["n_neighbors"]]
            }
        ]
        model_tuner_weights = ModelTuner(y_classes=preprocess.y_classes, tuning_params=knn_params_weights,
                                         model_type="KNN")
        model_tuner_weights.run_model_optimisation(X_train=X_train, y_train=y_train)
        print_optimiser_results(model_tuner_weights)
        # --------------------------------------------------------------------------------------------------------------

        knn_params_algorithm = [
            {
                "algorithm": ["ball_tree", "kd_tree", "brute"],
                "leaf_size": [20, 30, 40, 50],
                "weights": [model_tuner_weights.best_parameters["weights"]],
                "p": [model_tuner_weights.best_parameters["p"]],
                "n_neighbors": [model_tuner_weights.best_parameters["n_neighbors"]]
            }
        ]
        model_tuner_algorithm = ModelTuner(y_classes=preprocess.y_classes, tuning_params=knn_params_algorithm,
                                           model_type="KNN")
        model_tuner_algorithm.run_model_optimisation(X_train=X_train, y_train=y_train)
        print_optimiser_results(model_tuner_algorithm)
        # --------------------------------------------------------------------------------------------------------------

    else:
        # ==============================================================================================================
        # ==============================================================================================================
        # Testing Model Performance
        # ==============================================================================================================
        # ==============================================================================================================
        model_evaluator = ModelEvaluator(y_classes=preprocess.y_classes)

        model_evaluator.run_model_evaluation(X_train, y_train, X_test, y_test)
        model_evaluator.show_confusion_matrices()
        model_evaluator.plot_results()
        model_evaluator.save_results()

    timer.time_script()
    stages, times = timer.plot_data
    myPlt.plot_model_build_time(stages=stages, times=times)

    logger.info("Complete")
