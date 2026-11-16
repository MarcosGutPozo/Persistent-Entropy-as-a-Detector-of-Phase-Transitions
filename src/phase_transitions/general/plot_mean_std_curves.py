import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Global plotting style
# ---------------------------------------------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 1.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


def plot_mean_std_curves(
    x,
    data_left,
    data_right,
    params,
    left_ylabel,
    right_ylabel,
    param_label,
    xlabel="Time",
    filename="comparison.pdf"
):

    n = len(params)

    ncols_block = 2
    nrows = int(np.ceil(n / ncols_block))

    fig, axes = plt.subplots(
        nrows,
        4,
        figsize=(16, 2.2 * nrows),
        sharex=True
    )

    axes = np.asarray(axes).reshape(nrows, 4)

    for i, param in enumerate(params):

        row = i // 2
        col = i % 2

        ax = axes[row, col]

        mat = data_left[float(param)]

        mean = mat.mean(axis=0)
        std = mat.std(axis=0)

        ax.plot(
            x,
            mean,
            color="#1f77b4"
        )

        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color="#1f77b4",
            alpha=0.2
        )

        ax.set_title(
            rf"${param_label} = {param:g}$",
            fontsize=10
        )

        ax.set_ylabel(
            left_ylabel,
            fontsize=10
        )

        ax.tick_params(
            axis="both",
            labelsize=10
        )

        ax.grid(
            True,
            alpha=0.25
        )

        ax = axes[row, col + 2]

        mat = data_right[float(param)]

        mean = mat.mean(axis=0)
        std = mat.std(axis=0)

        ax.plot(
            x,
            mean,
            color="#1f77b4"
        )

        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color="#1f77b4",
            alpha=0.2
        )

        ax.set_title(
            rf"${param_label} = {param:g}$",
            fontsize=10
        )

        ax.set_ylabel(
            right_ylabel,
            fontsize=10
        )

        ax.tick_params(
            axis="both",
            labelsize=10
        )

        ax.grid(
            True,
            alpha=0.25
        )

    for i in range(n, nrows * 2):

        row = i // 2
        col = i % 2

        axes[row, col].axis("off")
        axes[row, col + 2].axis("off")

    for j in range(4):

        if axes[-1, j].axison:
            axes[-1, j].set_xlabel(
                xlabel,
                fontsize=10
            )

    for j in [0, 1]:
        axes[:, j].sharey(
            axes[0, 0]
        )

    for j in [2, 3]:
        axes[:, j].sharey(
            axes[0, 2]
        )

    fig.subplots_adjust(
        left=0.06,
        right=0.98,
        bottom=0.10,
        top=0.96,
        hspace=0.35,
        wspace=0.20
    )

    plt.savefig(
        filename,
        format="pdf",
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)
