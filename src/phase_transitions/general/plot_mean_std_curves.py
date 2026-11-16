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


# ---------------------------------------------------------
# Mean and standard deviation comparison plots
# ---------------------------------------------------------
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
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig = plt.figure(
        figsize=(16, 2.2 * nrows)
    )

    subfigs = fig.subfigures(
        1,
        2,
        wspace=0.03
    )

    for subfig, curves_by_param, ylabel in zip(
        subfigs,
        [data_left, data_right],
        [left_ylabel, right_ylabel]
    ):

        axes = subfig.subplots(
            nrows,
            ncols,
            sharex=True,
            sharey=True
        )

        axes = np.asarray(axes).reshape(-1)

        for i, (ax, param) in enumerate(zip(axes, params)):

            mat = curves_by_param[float(param)]

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
                ylabel,
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

            # Only show x-axis label in the last row
            if i // ncols == nrows - 1:
                ax.set_xlabel(
                    xlabel,
                    fontsize=10
                )

        # Hide unused axes
        for ax in axes[n:]:
            ax.axis("off")

        subfig.subplots_adjust(
            left=0.12,
            right=0.98,
            bottom=0.10,
            top=0.96,
            hspace=0.35,
            wspace=0.20
        )

    fig.savefig(
        filename,
        format="pdf",
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)
