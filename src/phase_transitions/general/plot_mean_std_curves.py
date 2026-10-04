import numpy as np
import matplotlib.pyplot as plt

def plot_mean_std_curves(
    x,
    curves_by_param,
    params,
    title,
    ylabel
):

    n = len(params)

    ncols = 2
    nrows = int(
        np.ceil(n / ncols)
    )

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(12, 3.5 * nrows),
        sharex=True,
        sharey=True
    )

    axes = np.asarray(
        axes
    ).reshape(-1)


    for ax, eta in zip(
        axes,
        params
    ):

        mat = curves_by_param[
            float(eta)
        ]

        mean = mat.mean(
            axis=0
        )

        std = mat.std(
            axis=0
        )


        # Mean
        ax.plot(
            x,
            mean,
            linewidth=2
        )


        # Standard deviation
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.2
        )


        ax.set_title(
            f"eta = {eta:g}"
        )

        ax.set_ylabel(
            ylabel
        )

        ax.grid(
            True,
            alpha=0.3
        )


    # Hide unused axes
    for ax in axes[n:]:
        ax.axis("off")


    for ax in axes[:n]:
        ax.set_xlabel("time")


    fig.suptitle(
        title,
        fontsize=14
    )

    plt.tight_layout()

    plt.show()