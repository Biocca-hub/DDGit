from scatter_plot import scatter_plot

import pandas as pd

import numpy as np

def pearson_corr(y_hat, y):
    vx = y_hat - y_hat.mean()
    vy = y - y.mean()

    return (vx * vy).sum() / (
        np.sqrt((vx**2).sum()) *
        np.sqrt((vy**2).sum()) +
        1e-8
    )

import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def single_scatter(ax, targets, predictions, title):

    df_plot = pd.DataFrame({
        "targets": targets,
        "predictions": predictions
    })

    sns.scatterplot(
        data=df_plot,
        x="targets",
        y="predictions",
        color="navy",
        s=15,
        edgecolor="black",
        linewidth=0.1,
        alpha=0.1,
        legend=False,
        ax=ax
    )

    min_val = min(min(targets), min(predictions))
    max_val = max(max(targets), max(predictions))
    pad = 0.05 * (max_val - min_val)

    ax.set_xlim(min_val - pad, max_val + pad)
    ax.set_ylim(min_val - pad, max_val + pad)

    # y = x
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="blue",
        linestyle="--",
        linewidth=1
    )

    # x = 0 e y = 0
    ax.axvline(0, color="blue", linestyle="--", linewidth=1)
    ax.axhline(0, color="blue", linestyle="--", linewidth=1)

    # medie
    mean_target = df_plot["targets"].mean()
    mean_prediction = df_plot["predictions"].mean()

    mean_diag = (mean_target + mean_prediction) / 2

    ax.scatter(
        mean_diag,
        mean_diag,
        marker="*",
        s=120,
        color="yellow",
        edgecolor="black",
        linewidth=1.2,
        zorder=10
    )

    ax.scatter(
        mean_target,
        mean_prediction,
        marker="*",
        s=120,
        color="red",
        edgecolor="black",
        linewidth=0.5,
        zorder=20
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Target")
    ax.set_ylabel("Prediction")
    ax.set_title(title)


def create_run_scatter_plots(df, output_dir):

    sns.set_theme(style="whitegrid", palette="viridis")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 12
    })

    for i in range(len(df)):

        train_targets = ast.literal_eval(df.loc[i, "train_targets"])
        train_predictions = ast.literal_eval(df.loc[i, "train_predictions"])

        val_targets = ast.literal_eval(df.loc[i, "val_targets"])
        val_predictions = ast.literal_eval(df.loc[i, "val_predictions"])

        test_targets = ast.literal_eval(df.loc[i, "test_targets"])
        test_predictions = ast.literal_eval(df.loc[i, "test_predictions"])

        fig, axes = plt.subplots(
            1, 3,
            figsize=(18, 6)
        )

        single_scatter(
            axes[0],
            train_targets,
            train_predictions,
            "Training"
        )

        single_scatter(
            axes[1],
            val_targets,
            val_predictions,
            "Validation"
        )

        single_scatter(
            axes[2],
            test_targets,
            test_predictions,
            "Testing"
        )

        run_number = df.loc[i, "run_number"]

        fig.suptitle(
            f"Run {run_number}",
            fontsize=18
        )

        plt.tight_layout()

        plt.savefig(
            f"{output_dir}/scatter_run_{run_number}.png",
            dpi=600,
            bbox_inches="tight"
        )

        plt.close()

df = pd.read_csv('../../4_b_mlp/output/best_combination/good_split_all_predictions/scat.txt', sep = '\t')

#create_run_scatter_plots(df,"../output/")

import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_targets = []
train_predictions = []

val_targets = []
val_predictions = []

test_targets = []
test_predictions = []

for i in range(len(df)):

    train_targets.extend(ast.literal_eval(df.loc[i, "train_targets"]))
    train_predictions.extend(ast.literal_eval(df.loc[i, "train_predictions"]))

    val_targets.extend(ast.literal_eval(df.loc[i, "val_targets"]))
    val_predictions.extend(ast.literal_eval(df.loc[i, "val_predictions"]))

    test_targets.extend(ast.literal_eval(df.loc[i, "test_targets"]))
    test_predictions.extend(ast.literal_eval(df.loc[i, "test_predictions"]))

"""
fig, ax = plt.subplots(figsize=(8, 8))

sns.scatterplot(
    x=train_targets,
    y=train_predictions,
    color="green",
    alpha=0.3,
    s=5,
    label="Training",
    ax=ax
)

sns.scatterplot(
    x=val_targets,
    y=val_predictions,
    color="blue",
    alpha=0.3,
    s=5,
    label="Validation",
    ax=ax
)

sns.scatterplot(
    x=test_targets,
    y=test_predictions,
    color="red",
    alpha=0.3,
    s=5,
    label="Test",
    ax=ax
)

all_values = (
    train_targets + train_predictions +
    val_targets + val_predictions +
    test_targets + test_predictions
)

min_val = min(all_values)
max_val = max(all_values)

pad = 0.05 * (max_val - min_val)

ax.set_xlim(min_val - pad, max_val + pad)
ax.set_ylim(min_val - pad, max_val + pad)

ax.plot(
    [min_val, max_val],
    [min_val, max_val],
    'k--',
    linewidth=1,
    label='y = x'
)

ax.axvline(0, color='black', linestyle=':', linewidth=1)
ax.axhline(0, color='black', linestyle=':', linewidth=1)

ax.set_aspect('equal')

ax.set_xlabel("Target")
ax.set_ylabel("Prediction")
ax.set_title("All runs")

ax.legend()

plt.tight_layout()
plt.savefig("all_runs_scatter.png", dpi=600)
plt.close()
"""
fig, axes = plt.subplots(
    1, 3,
    figsize=(18, 6),
    sharex=True,
    sharey=True
)

# TRAIN
sns.scatterplot(
    x=train_targets,
    y=train_predictions,
    color="green",
    alpha=0.05,
    s=5,
    ax=axes[0]
)

# VALIDATION
sns.scatterplot(
    x=val_targets,
    y=val_predictions,
    color="blue",
    alpha=0.05,
    s=5,
    ax=axes[1]
)

# TEST
sns.scatterplot(
    x=test_targets,
    y=test_predictions,
    color="red",
    alpha=0.05,
    s=5,
    ax=axes[2]
)

all_values = (
    train_targets + train_predictions +
    val_targets + val_predictions +
    test_targets + test_predictions
)

min_val = min(all_values)
max_val = max(all_values)

pad = 0.05 * (max_val - min_val)

for ax in axes:

    ax.set_xlim(min_val - pad, max_val + pad)
    ax.set_ylim(min_val - pad, max_val + pad)

    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        'k--',
        linewidth=1
    )

    ax.axvline(0, color='black', linestyle=':', linewidth=1)
    ax.axhline(0, color='black', linestyle=':', linewidth=1)

    #ax.set_aspect('equal')

axes[0].set_title("Training")
axes[1].set_title("Validation")
axes[2].set_title("Test")

fig.suptitle(
    "Predictions vs Targets (20 runs)",
    fontsize=18
)

plt.tight_layout()

plt.savefig(
    "../output/all_runs_scat1.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close()