"""
Author:         Victor Loveday
Date:           01/08/2022 
"""

import logging
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer

from src.config import RANDOM_STATE
from src.logger_config import setup_logger
from src.plotting import compare_average_results_plots, compare_class_results
from src.timer import Timer
from src.utils import ravel_y

logger = setup_logger(logging.getLogger(__name__))
timer = Timer()


class ConfusionMatrix:
    """
    Helper class for dealing with Confusion matrix data.
    """

    def __init__(self, name, y_test, y_pred, class_names):
        """
        Class constructor.

        :param name: The name of the model containing this confusion matrix.
        :param y_test: The y_test dataset.
        :param y_pred: The model predictions on y
        :param class_names: The class names of the y labels.
        """
        self._name = name
        self._class_names = class_names
        self.matrix = confusion_matrix(y_test, y_pred, labels=[x for x in range(23)])
        self.scores = []
        self.calculate_results(class_names)

    def calculate_results(self, names):
        """
        Calculate the metric results of the confusion matrix.

        :param names: The names of the features.
        """
        for row, col in zip(range(0, self.matrix.shape[0]), range(0, self.matrix.shape[1])):
            TP = self.matrix[row, col]
            TN = np.trace(self.matrix) - TP
            FP = np.sum(self.matrix[:, col]) - TP
            FN = np.sum(self.matrix[row, :]) - TP

            self.scores.append({
                "Class": names[row],
                "TP": TP,
                "TN": TN,
                "FP": FP,
                "FN": FN,
                "Accuracy": self.calculate_accuracy(TP=TP, TN=TN, FP=FP, FN=FN),
                "Precision": self.calculate_precision(TP=TP, FP=FP),
                "Sensitivity": self.calculate_recall(TP=TP, FN=FN),
                "Specificity": self.calculate_specificity(TN=TN, FP=FP),
                "F-Score": self.calculate_f_score(TP=TP, FP=FP, FN=FN),
                "TPR": self.calculate_recall(TP=TP, FN=FN) / 100,
                "TNR": self.calculate_specificity(TN=TN, FP=FP) / 100,
                "FPR": self.calculate_fpr(TN=TN, FP=FP),
                "FNR": self.calculate_fnr(TP=TP, FN=FN)
            })

    @staticmethod
    def calculate_accuracy(TP, TN, FP, FN):
        """
        Helper method to generate the accuracy metric for a given class.

        :param TP: True Positives (Good)
        :param TN: True Negatives (Good)
        :param FP: False Positives (Bad)
        :param FN: False Positives (Bad)
        :return: The accuracy metric for the given class described by the passed params.
        """
        return round(((TP + TN) / (TP + FP + TN + FN)) * 100, 2)

    # TPR / Sensitivity
    @staticmethod
    def calculate_recall(TP, FN):
        """
        Helper method to generate the recall(a.k.a sensitivity) metric for a given class.

        :param TP: True Positives (Good)
        :param FN: False Positives (Bad)
        :return: The recall(a.k.a sensitivity) metric for the given class described by the passed params.
        """
        return round((TP / (TP + FN)) * 100, 2)

    @staticmethod
    def calculate_specificity(TN, FP):
        """
        Helper method to generate the specificity metric for a given class.

        :param TN: True Negatives (Good)
        :param FP: False Positives (Bad)
        :return: The specificity metric for the given class described by the passed params.
        """
        return round((TN / (FP + TN)) * 100, 2)

    @staticmethod
    def calculate_precision(TP, FP):
        """
        Helper method to generate the precision metric for a given class.

        :param TP: True Positives (Good)
        :param FP: False Positives (Bad)
        :return: The precision metric for the given class described by the passed params.
        """
        return round((TP / (TP + FP)) * 100, 2)

    def calculate_f_score(self, TP, FP, FN):
        """
        Helper method to generate the F-Score metric for a given class.

        :param TP: True Positives (Good)
        :param FP: False Positives (Bad)
        :param FN: False Positives (Bad)
        :return: The F-Score metric for the given class described by the passed params.
        """
        r = self.calculate_recall(TP, FN)
        p = self.calculate_precision(TP, FP)

        return round(((2 * r * p) / (r + p)) / 100, 2)

    def calculate_fpr(self, TN, FP):
        """
        Helper method to generate the FPR (False Positive Rate) metric for a given class.

        :param TN: True Negatives (Good)
        :param FP: False Positives (Bad)
        :return: The FPR metric for the given class described by the passed params.
        """
        return round((100 - self.calculate_specificity(TN=TN, FP=FP)) / 100, 2)

    def calculate_fnr(self, TP, FN):
        """
        Helper method to generate the FNR (False Negative Rate) metric for a given class.

        :param TP: True Positives (Good)
        :param FN: False Positives (Bad)
        :return: The FNR metric for the given class described by the passed params.
        """
        return round((100 - self.calculate_recall(TP=TP, FN=FN)) / 100, 2)

    @property
    def class_names(self) -> list:
        return self._class_names

    @property
    def name(self) -> str:
        return self._name

    @property
    def data(self) -> pd.DataFrame:
        """
        Retrieves the confusion matrix metric scores as a dataframe with NaN filled with 0's.

        :return: The confusion matrix data as a Dataframe with NaN's filled with 0's.
        """
        return pd.DataFrame(self.scores).fillna(0)

    @property
    def matrix_normalised(self) -> np.ndarray:
        """
        Creates a normalised version of the confusion matrix from 0 -> 1.

        :return: A numpy array containing the normalised values of the confusion matrix.
        """
        division_matrix = np.sum(self.matrix, axis=1)
        division_matrix_inverse = division_matrix.reshape(-1, 1)
        confusion_matrix_normalised = (self.matrix / division_matrix_inverse[None, :])
        confusion_matrix_normalised = np.nan_to_num(confusion_matrix_normalised, posinf=0.0, neginf=0.0)

        return confusion_matrix_normalised

    def show(self):
        """
        Helper method to show the confusion matrix as a heatmap plot.
        """
        cm_norm = self.matrix_normalised

        plt.subplots(figsize=(20, 10))
        ax: plt.Axes = sns.heatmap(cm_norm[0], cmap="YlOrRd", linewidths=0.5, annot=True,
                                   xticklabels=self.class_names,
                                   yticklabels=self.class_names,
                                   fmt=".2f", cbar=False)
        ax.set_title(f"Confusion matrix ({self.name})")
        ax.set_xlabel("Predictions")
        ax.set_ylabel("Real Values")
        plt.show()

    @property
    def f_scores(self) -> dict:
        df = self.data[["Class", "F-Score"]]

        return dict(zip(df['Class'], df['F-Score']))

    @property
    def sensitivities(self) -> dict:
        df = self.data[["Class", "Sensitivity"]]

        return dict(zip(df['Class'], df['Sensitivity']))

    @property
    def precisions(self) -> dict:
        df = self.data[["Class", "Precision"]]

        return dict(zip(df['Class'], df['Precision']))


class Model:

    def __init__(self, name, model, class_names):
        """
        Class constructor.

        :param name: The textual name of this model (RF/KNN).
        :param model: The model to use.
        :param class_names: The y labels corresponding class name.
        """
        self.name = name
        self._model = model
        self._class_names = class_names
        self._y_pred = None

        self._confusion_matrix = None

    def fit(self, X, y):
        """
        Fit the (X, y) datasets to the model.
        :param X: The X dataset of features.
        :param y: The y dataset of output labels.
        """
        logger.info(f"Step  - Fitting {self.name} model BEGIN")
        self._model.fit(X, ravel_y(y))
        logger.info(f"Step  - Fitting {self.name} model END {timer.time_stage(f'Fitting Model {self.name}')}")

    def predict(self, X_test, force=False):
        """
        Predicts y values for the set of features found in X_test.

        :param X_test: The dataset of input features.
        :param force: Flag to force re-computation of y_pred if previously calculated.
        :return: The y predictions for X_test input features.
        """
        logger.info(f"Step  - Prediction BEGIN")
        if self._y_pred is None or force is True:
            self._y_pred = self._model.predict(X_test)
        else:
            logger.info(f"Step  - Prediction already evaluated")
        logger.info(f"Step  - Prediction END {timer.time_stage(f'y Prediction {self.name}')}")

        return self._y_pred

    def confusion_matrix(self, y_test):
        """
        Creates a confusion matrix object.

        :param y_test: The y output labels to use to construct the confusion matrix.
        :return: The newly constructed confusion matrix.
        """
        if self._confusion_matrix is None:
            if self._y_pred is None:
                logger.info(f"y_pred has not been evaluated yet. Call 'predict(X_test)'")
            else:
                self._confusion_matrix = ConfusionMatrix(name=self.name,
                                                         y_test=y_test,
                                                         y_pred=self._y_pred,
                                                         class_names=self.class_names)

        return self._confusion_matrix

    def multi_class_roc_auc_score(self, y_test, average="macro"):
        """
        Returns the roc auc score for the multiclass problem.

        :param y_test: The y output labels to use.
        :param average: The averaging metric to use.
        :return: The ROC Area Under the Curve score.
        """
        lb = LabelBinarizer()

        lb.fit(y_test)
        y_test = lb.transform(y_test)
        y_pred = lb.transform(self._y_pred)

        score = roc_auc_score(y_test, y_pred, average=average)

        logger.info(f"Step  - {self.name} ROC AUC Score: {score}")

        return score

    def plot_multi_class_roc_curve(self, y_test: pd.DataFrame):
        """
        Creates a plot of all output features ROC curves. Experimental and not verified the correct data is being
        output.

        :param y_test: The test set of output labels.
        :return: A tuple of FPR, TPR and ROC_AUC.
        """
        lb = LabelBinarizer()
        lb.fit(y_test)

        y_test = lb.transform(y_test)
        class_names = [self.get_class_for_label(label) for label in lb.classes_]

        y_pred = lb.transform(self._y_pred)

        fpr = {}
        tpr = {}
        roc_auc = {}
        for i, _ in enumerate(class_names):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        self._plot_roc_curve(fpr, tpr, roc_auc, class_names)

        return fpr, tpr, roc_auc

    @staticmethod
    def _plot_roc_curve(fpr, tpr, roc_auc, class_names):
        """
        Creates a ROC Plot for a given output feature.

        :param fpr: False positive rate.
        :param tpr: True postive rate.
        :param roc_auc: ROC Area under the curve scoring.
        :param class_names: The output label's class names.
        """
        fig, axs = plt.subplots(5, 5)

        for (p1, p2), i in zip(product([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]), range(len(class_names))):
            print(f"{p1}, {p2}, {i}, {class_names[i]}")
            lw = 2
            axs[p1, p2].plot(fpr[i], tpr[i], color='darkorange',
                             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[i])
            axs[p1, p2].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            axs[p1, p2].set_title(class_names[i])

        plt.title("ROC Curve Analysis")
        plt.xlabel("TPR")
        plt.ylabel("FPR")
        plt.show()

    @property
    def model(self):
        return self._model

    def get_class_for_label(self, label) -> str:
        """
        Gets a class name from it's numeric label.

        :param label: The numeric label to match.
        :return: The class name matching the label passed.
        """
        return self._class_names.get(label)

    @property
    def class_names(self) -> list:
        """
        List of all class names.

        :return: A list of all class names.
        """
        return [self._class_names.get(idx) for idx, _ in enumerate(self._class_names)]

    @property
    def get_classes_size(self) -> int:
        """
        The amount of classes.

        :return: The amount of classes.
        """
        return len(self.class_names)

    @property
    def matrix(self) -> ConfusionMatrix:
        """
        Getter accessor for confusion matrix.

        :return: The confusion matrix fo this model.
        """
        return self._confusion_matrix


class ModelEvaluator:
    """
    Class to manage evaluation of the models passed.
    """

    def __init__(self, y_classes):
        """
        Class constructor.

        :param y_classes: The class names for y output labels.
        """
        self.y_classes = y_classes
        self.y_test = None

        self.rf_model = Model("RF", RandomForestClassifier(n_estimators=100,
                                                           criterion="gini",
                                                           max_features="auto",
                                                           random_state=RANDOM_STATE,
                                                           n_jobs=-1), y_classes)
        self.knn_model = Model("KNN", KNeighborsClassifier(n_neighbors=10,
                                                           p=1,
                                                           weights="distance",
                                                           algorithm="kd_tree",
                                                           n_jobs=-1), y_classes)

    @staticmethod
    def _eval_model(model: Model, X, y) -> None:
        """
        Evaluates the passed model with (X, y) datasets.

        :param model: The model to validate (RF/KNN).
        :param X: The input feature dataset.
        :param y: The output labels to predict.
        """
        logger.info(f"Step  - Evaluating {model.name} model BEGIN")
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
        cross_val_score(model.model, X, ravel_y(y), scoring='f1_macro', cv=cv, verbose=10, n_jobs=-1)
        logger.info(f"Step  - Evaluating {model.name} model {timer.time_stage(f'{model.name} Evaluation')} END")

    def run_model_evaluation(self, X_train, y_train, X_test, y_test):
        """
        Fits and evaluates all internal models.

        :param X_train: The training set of input features.
        :param y_train: The training set of output labels.
        :param X_test: The test set of input features.
        :param y_test: The test set of output labels.
        """
        self.y_test = y_test
        logger.info("Stage - Model Analysis BEGIN")
        for classifier in (self.knn_model, self.rf_model):
            classifier.fit(X_train, y_train)
            self._eval_model(classifier, X=X_test, y=y_test)
            classifier.predict(X_test=X_test)
        logger.info("Stage - Model Analysis END")

    @property
    def rf(self):
        return self.rf_model

    @property
    def knn(self):
        return self.knn_model

    def show_confusion_matrices(self):
        """
        Helper method to show both models confusion matrices.
        """
        self.rf.confusion_matrix(y_test=self.y_test).show()
        self.knn.confusion_matrix(y_test=self.y_test).show()

    def show_roc_curves(self):
        """
        Helper method to show both ROC curve collections for both models. Experimental.
        """
        self.rf.plot_multi_class_roc_curve(y_test=self.y_test)
        self.knn.plot_multi_class_roc_curve(y_test=self.y_test)

    def roc_auc_scores(self):
        """
        Helper method to show ROC AUC metric. Experimental.
        """
        self.rf.multi_class_roc_auc_score(y_test=self.y_test)
        self.knn.multi_class_roc_auc_score(y_test=self.y_test)

    def save_results(self):
        """
        Saves the metric data for both models to a csv file for import to Excel.
        """
        self.rf.matrix.data.to_csv("rf_results_averages.csv", index=False)
        pd.DataFrame(self.rf.matrix.matrix).to_csv("rf_confusion_matrix.csv", index=False)
        pd.DataFrame(self.rf.matrix.matrix_normalised[0, :, :]).to_csv("rf_confusion_matrix_normalised.csv",
                                                                       index=False)
        self.knn.matrix.data.to_csv("knn_results_averages.csv", index=False)
        pd.DataFrame(self.rf.matrix.matrix).to_csv("knn_confusion_matrix.csv", index=False)
        pd.DataFrame(self.rf.matrix.matrix_normalised[0, :, :]).to_csv("knn_confusion_matrix_normalised.csv",
                                                                       index=False)

    def show_result_overview(self):
        """
        Helper method to show pretty print format of metric results to the console for both models.

        :return: The metric results of the model.
        """
        from pprint import pprint
        knn_result = {key: round(value, 2) for key, value in self._get_result_overview(self.knn).items()}
        rf_result = {key: round(value, 2) for key, value in self._get_result_overview(self.rf).items()}
        results = {
            "KNN": knn_result,
            "RF": rf_result
        }

        pprint(results)

        return results

    @staticmethod
    def _get_result_overview(model: Model):
        """
        Returns a dict with the passed models result metric data.

        :param model: The model to extract data from (RF/KNN)
        :return: A dict with all the metric data contained.
        """
        return {
            "Avg F-Score": model.matrix.data["F-Score"].mean(),
            "Avg Accuracy(a)": model.matrix.data["Accuracy"].mean(),
            "Avg Precision(p)": model.matrix.data["Precision"].mean(),
            "Avg Sensitivity(r)": model.matrix.data["Sensitivity"].mean(),
            "Avg Specificity": model.matrix.data["Specificity"].mean(),
            "Avg TPR": model.matrix.data["TPR"].mean(),
            "Avg TNR": model.matrix.data["TNR"].mean(),
            "Avg FPR": model.matrix.data["FPR"].mean(),
            "Avg FNR": model.matrix.data["FNR"].mean(),
        }

    def plot_results(self):
        compare_class_results(self.rf.matrix.sensitivities, self.knn.matrix.sensitivities,
                              "Random Forest Vs KNN Sensitivity Comparision")
        compare_class_results(self.rf.matrix.precisions, self.knn.matrix.precisions,
                              "Random Forest Vs KNN Precision Comparision")
        compare_class_results(self.rf.matrix.f_scores, self.knn.matrix.f_scores,
                              "Random Forest Vs KNN F-Score Comparision")

        compare_average_results_plots(self.show_result_overview())


class ModelTuner(ModelEvaluator):
    """
    Helper class for performing GridSearch hyper-parameter analysis.
    """

    def __init__(self, y_classes, tuning_params: list, model_type: str):
        """
        Class constructor.

        :param y_classes: The class names of the y numeric labels.
        :param tuning_params: The tuning parameters to use.
        :param model_type: The model type (RF/KNN)
        """
        super().__init__(y_classes=y_classes)
        self.tuning_params = tuning_params
        self.model_type = model_type
        self._grid_search = None
        self._cv_results = None
        self._best_score = None
        self._best_parameters = None

    @property
    def grid_search(self):
        return self._grid_search

    @property
    def best_score(self):
        return self._best_score

    @property
    def best_parameters(self):
        return self._best_parameters

    @property
    def cv_results(self):
        return self._cv_results

    @property
    def classifier(self):
        return {
            "RF": self.rf_model.model,
            "KNN": self.knn_model.model,
        }.get(self.model_type.upper(), None)

    def run_model_optimisation(self, X_train, y_train):
        """
        Fits and runs grid search using X_train and y_train.

        :param X_train: The training input set of features.
        :param y_train: The training output set of labels.
        """
        logger.info("Stage - Model Optimisation BEGIN")
        y_train = ravel_y(y_train)
        self.classifier.fit(X_train, y_train)
        self._run_grid_search(self.classifier, X=X_train, y=y_train)
        logger.info("Stage - Model Optimisation END")

    def _run_grid_search(self, model, X, y):
        """
        Private helper method to run GridSearch and record results.

        :param model: The model to run through GridSearch.
        :param X: The X training set of features.
        :param y: The y training set of output labels.
        """
        grid_search = GridSearchCV(estimator=model,
                                   param_grid=self.tuning_params,
                                   scoring="f1_macro",
                                   cv=4,
                                   verbose=10,
                                   n_jobs=-1)
        self._grid_search = grid_search.fit(X, y)
        self._cv_results = grid_search.cv_results_
        self._best_score = grid_search.best_score_
        self._best_parameters = grid_search.best_params_
