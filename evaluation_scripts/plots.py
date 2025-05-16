import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import ListedColormap


def plot_significance_matrices(matrices, figsize=(10, 3)):
    """
    Args:
        matrices (dict): Dictionary where keys are metric names and values are lists of matrices.
        figsize (tuple): Figure size.
    """
    num_rows = len(next(iter(matrices.values())))  # Number of matrices per metric
    num_cols = len(matrices)  # Number of metrics

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    if num_rows == 1:
        axes = np.expand_dims(axes, 0)

    colors = sns.color_palette("muted")
    muted_red = colors[3]
    muted_blue = colors[0]

    cmap = ListedColormap([muted_red, muted_blue])
    for row in range(num_rows):
        for col, (metric_name, matrix_list) in enumerate(matrices.items()):
            ax = axes[row, col]
            matrix = matrix_list[row]
            sns.heatmap(
                matrix,
                mask=np.isnan(matrix),    # Mask NaNs
                ax=ax,
                cmap=cmap,                # Custom colormap
                cbar=False,
                #linewidths=0.5,
                #linecolor='black',
                xticklabels=False,
                yticklabels=False,
                vmin=0, vmax=1,            # Map 0 -> red, 1 -> blue
                square=True
            )

            if row == 0:
                visible_metric_name = metric_name
                if metric_name == "ndcg":
                    visible_metric_name = "nDCG"
                elif metric_name == "map":
                    visible_metric_name = "MAP"
                elif metric_name == "mrr":
                    visible_metric_name = "MRR"
                elif metric_name == "reo":
                    visible_metric_name = r"$REO_{Overall}$"
                ax.set_title(visible_metric_name, fontsize=12)
                # ax.set_title(metric_name, fontsize=12)

    plt.tight_layout()
    plt.show()

def plot_significance_matrices_with_groups(matrices, groups, figsize=(10, 10)):
    """
    Args:
        matrices (dict): Dictionary where keys are metric names and values are lists of matrices per group.
        groups (list): List of group names.
        figsize (tuple): Figure size.
    """
    fig, axes = plt.subplots(len(groups), len(matrices), figsize=figsize)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    colors = sns.color_palette("muted")

    muted_red = colors[3]
    muted_blue = colors[0]

    cmap = ListedColormap([muted_red, muted_blue])
    for row, group in enumerate(groups):
        for col, (metric_name, matrix_list) in enumerate(matrices.items()):
            ax = axes[row, col]
            matrix = matrix_list[row]

            sns.heatmap(
                matrix,
                mask=np.isnan(matrix),  # Mask NaNs
                ax=ax,
                cmap=cmap,  # Custom colormap
                cbar=False,
                # linewidths=0.5,
                # linecolor='black',
                xticklabels=False,
                yticklabels=False,
                vmin=0, vmax=1,  # Map 0 -> red, 1 -> blue
                square=True
            )

            if row == 0:
                visible_metric_name = metric_name
                if metric_name == "disp_exp":
                    visible_metric_name = "DE"
                elif metric_name == "lrd":
                    visible_metric_name = "LRD"
                elif metric_name == "reo":
                    visible_metric_name = "REO"
                ax.set_title(visible_metric_name, fontsize=12)
            if col == 0:
                ax.set_ylabel(group, fontsize=12)

    plt.tight_layout()
    plt.show()


def boxplot_comparison(df_a, df_b, df_c, x, y, y_name, lower_lim=0):
    combined_df = pd.concat([df_a, df_b, df_c], ignore_index=True)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=combined_df, x=x, y=y, palette='muted')

    plt.ylabel(y_name)
    plt.ylim(lower_lim, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

def violin_comparison(merged_df):
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        data=merged_df,
        x='Group',
        y='value',
        hue='Type',
        split=True,
        inner='quartile',
        palette='muted'
    )
    plt.ylabel("Value")
    plt.tight_layout()
    plt.show()
