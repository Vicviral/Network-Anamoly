"""
Author:         David Walshe
Date:           09/04/2020   
"""
from collections import Counter
from itertools import zip_longest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import change_label_to_class


def get_colors():
    import json
    with open("colors.json") as fh:
        colors = json.load(fh)

    return list(colors.values())


def plot_2d_space(X, y, label='Classes'):
    colors = get_colors()
    for label, colors in zip_longest(np.unique(y), colors):
        plt.scatter(
            X[y == label, 0],
            X[y == label, 1],
            c=colors, label=label, marker="o"
        )
    plt.title(label)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def plot_value_counts(y, title="Count (target)"):
    if type(y) is not pd.DataFrame:
        y = pd.DataFrame(y, columns=["signature"])

    target_count = y["signature"].value_counts()
    target_count.plot.bar(title=title)
    plt.xlabel("Classes")
    plt.ylabel("Instance Count")
    plt.show()


def plot_model_build_time(stages, times):
    import math
    fig, ax = plt.subplots()
    ax.bar(stages, times)
    plt.xticks(stages, stages)
    max_time = math.ceil(max(times))
    tick_scale = math.ceil(max_time / 20)
    max_time += tick_scale
    plt.yticks([i for i in range(0, max_time, tick_scale)],
               [i if max_time < 60 else f"{int(i / 60)}:{i % 60}" for idx, i in
                enumerate(range(0, max_time, tick_scale))])
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    total_time = sum(times)
    if max_time > 60:
        total_time = f"{round(total_time / 60)}m {round(total_time % 60)}s"
        plt.ylabel("Minutes")
    else:
        plt.ylabel("Seconds")
    plt.xlabel("Stages")

    textstr = f"Total Time: {total_time}"

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    plt.show()


def show_pca_plot(X, y, title="PCA Component Plot"):
    plt.scatter(X[:, 0], X[:, 1],
                c=y["signature"], edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('tab20', 23))
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title(title)
    plt.colorbar()
    plt.show()


def plot_value_counts_compare(y1, y2, level="All", title="Raw Vs Reduced Datasets (All)",
                              label1="Raw Data", label2="Reduced Data"):
    y1_count = Counter(y1["signature"])
    y2_count = Counter(y2["signature"])

    ks = []
    mx = 10000
    mn = 900
    for key, value in y1_count.items():
        if level == "Max":
            title = f"Raw Vs Reduced Datasets (High Occurrence)"
            if value <= mx:
                ks.append(key)
        elif level == "Mid":
            title = f"Raw Vs Reduced Datasets (Mid Occurrence)"
            if value > mx or value <= mn:
                ks.append(key)
        elif level == "Min":
            title = f"Raw Vs Reduced Datasets (Low Occurrence)"
            if value > mn:
                ks.append(key)
        else:
            break

    if len(ks) > 0:
        for k in ks:
            y1_count.pop(k)

    keys = list(y1_count.keys())

    y1_group = [y1_count[key] for key in keys]
    y2_group = [y2_count[key] for key in keys]

    if type(keys[0]) is int:
        keys = [change_label_to_class(key) for key in keys]

    x = np.arange(len(keys))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, y1_group, width, label=label1)
    ax.bar(x + width / 2, y2_group, width, label=label2)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Instance Occurrences')
    ax.set_xlabel('Classes')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.legend()

    fig.tight_layout()

    plt.xticks(rotation=90)
    plt.show()


def compare_average_results_plots(result_metrics: dict):
    rf_results = result_metrics["RF"]
    knn_results = result_metrics["KNN"]

    percentage_result_keys = ['Avg Accuracy(a)', 'Avg Precision(p)', 'Avg Sensitivity(r)', 'Avg Specificity']
    x = np.arange(len(percentage_result_keys))  # the label locations

    rf_per_data = [rf_results[label] for label in percentage_result_keys]
    knn_per_data = [knn_results[label] for label in percentage_result_keys]

    results_plotter(rf_per_data, knn_per_data, keys=percentage_result_keys, x=x)

    float_result_keys = ['Avg F-Score', 'Avg TPR', 'Avg TNR', 'Avg FPR', 'Avg FNR']
    x = np.arange(len(float_result_keys))  # the label locations

    rf_float_data = [rf_results[label] for label in float_result_keys]
    knn_float_data = [knn_results[label] for label in float_result_keys]

    results_plotter(rf_float_data, knn_float_data, keys=float_result_keys, x=x, scale="0-1")


def compare_class_results(rf_result_metric: dict, knn_result_metric, title="No Title"):
    classes = list(rf_result_metric.keys())

    x = np.arange(len(classes))  # the label locations

    rf_per_data = [rf_result_metric[label] for label in classes]
    knn_per_data = [knn_result_metric[label] for label in classes]

    results_plotter(rf_per_data, knn_per_data, keys=classes, x=x, title=title, rotate=True)


def results_plotter(rf_data, knn_data, keys, x, scale="%",
                    title="Average Metric Comparison Between KNN and Random Forests", rotate=False):
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, rf_data, width, label="KNN")
    ax.bar(x + width / 2, knn_data, width, label="RF")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(f'Output Value ({scale})')
    ax.set_xlabel('Metrics')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    fig.tight_layout()
    if rotate:
        plt.xticks(rotation=90)
    plt.show()
