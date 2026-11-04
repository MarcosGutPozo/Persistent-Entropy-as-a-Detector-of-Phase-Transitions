import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.phase_transitions.kuramoto.kuramoto import Kuramoto
from src.phase_transitions.general.persistent_entropy import (
    pe_from_distance_matrix,
    plot_barcode
)
from src.phase_transitions.general.phase_detection_algorithm import (
    detect_solutions,
    plot_solution_probabilities,
    plot_phase_detection
)
from src.phase_transitions.general.plot_mean_std_curves import (
    plot_mean_std_curves
)


np.random.seed(42)


# ============================================================
# PARAMETERS
# ============================================================

N = 50
t0 = 0.0
t1 = 10.0
steps = 500
K_values = np.arange(0.0, 15.0 + 1e-9, 0.1)
K_selected = [0.0, 1.0, 3.0, 5.0, 10.0, 15.0]
R_K_train = 100
R_K_test = 100
delta = 0.3
alpha = 0.05


# ============================================================
# MODEL
# ============================================================

model = Kuramoto(
    N=N,
    t0=t0,
    t1=t1,
    steps=steps,
    K_values=K_values,
    R_K_train=R_K_train,
    R_K_test=R_K_test,
    K_selected=K_selected
)

kuramoto_time = model.t0 + np.arange(model.steps) * model.dt


# Indices of selected K values within K_values
selected_indices = [
    np.where(np.isclose(model.K_values, K))[0][0]
    for K in model.K_selected
]


# ============================================================
# 1. TRAINING SIMULATIONS
# ============================================================

# Store temporal curves only for selected K values.
kuramoto_r = {
    float(k): np.empty((R_K_train, steps), dtype=np.float32)
    for k in K_selected
}

kuramoto_pe = {
    float(k): np.empty((R_K_train, steps), dtype=np.float32)
    for k in K_selected
}

# Store theta for all K values.
theta_values = np.empty(
    (R_K_train, len(model.K_values)),
    dtype=object
)


for r in range(R_K_train):

    G = model.random_adjacency()
    omega0, theta0 = model.sample_kuramoto_initial()

    for j, k in enumerate(
        tqdm(model.K_values, desc=f"Training {r+1}/{R_K_train}")
    ):

        theta_ts, r_ts = model.simulate_kuramoto(
            G=G,
            omega0=omega0,
            theta0=theta0,
            K=float(k)
        )

        theta_values[r, j] = theta_ts

        # Calculate temporal quantities only for selected K.
        if j in selected_indices:

            kuramoto_r[float(k)][r, :] = r_ts

            pe0, _ = model.kuramoto_pe_h0_series(
                theta_ts,
                normalize=False
            )

            kuramoto_pe[float(k)][r, :] = pe0


# Temporal plots
plot_mean_std_curves(
    x=kuramoto_time,
    data_left=kuramoto_r,
    data_right=kuramoto_pe,
    params=model.K_selected,
    left_ylabel=r"$r(t)$",
    right_ylabel=r"PE($H_0$)",
    param_label="K",
    xlabel="Time",
    filename="kuramoto_r_pe.pdf"
)


# ============================================================
# 2. TRAINING WEIGHTS
# ============================================================

all_weights_training = np.empty(
    (R_K_train, len(model.K_values)),
    dtype=object
)

for r in range(R_K_train):

    for j in range(len(model.K_values)):

        theta = theta_values[r, j][-1]

        D = model.synchrony_distance_from_theta(theta)

        _, weight, _ = pe_from_distance_matrix(
            D,
            dim=0,
            min_persistence=0.01
        )

        all_weights_training[r, j] = weight


# ============================================================
# 3. TEST SIMULATIONS AND WEIGHTS
# ============================================================

theta_values_test = np.empty(
    (R_K_test, len(model.K_values)),
    dtype=object
)

all_weights_test = np.empty(
    (R_K_test, len(model.K_values)),
    dtype=object
)

entropy_test = np.empty(
    (R_K_test, len(model.K_values)),
    dtype=object
)


for r in range(R_K_test):

    G = model.random_adjacency()
    omega0, theta0 = model.sample_kuramoto_initial()

    for j, k in enumerate(
        tqdm(model.K_values, desc=f"Test {r+1}/{R_K_test}")
    ):

        theta_ts, _ = model.simulate_kuramoto(
            G=G,
            omega0=omega0,
            theta0=theta0,
            K=float(k)
        )

        theta_values_test[r, j] = theta_ts

        D = model.synchrony_distance_from_theta(theta_ts[-1])

        _, weight, pe = pe_from_distance_matrix(
            D,
            dim=0,
            min_persistence=0.01
        )

        all_weights_test[r, j] = weight
        entropy_test[r, j] = pe


# ============================================================
# 4. PHASE DETECTION
# ============================================================

best_solutions = detect_solutions(
    all_weights_training,
    delta=delta,
    alpha=alpha,
    kmax=2
)


plot_solution_probabilities(
    all_weights_test,
    best_solutions,
    delta=delta,
    alpha=alpha,
    param_values=model.K_values,
    param_symbol="K",
    filename="kuramoto_robustness"
)


plot_phase_detection(
    all_weights_test,
    best_solutions[0],
    delta=delta,
    alpha=alpha,
    entropies=entropy_test,
    param_values=model.K_values,
    param_symbol="K",
    delta_values=(0.35, 0.40, 0.45),
    ylabel=r"PE($H_0$)",
    filename="kuramoto_phase_detection"
)


# ============================================================
# 5. BARCODE VISUALIZATION
# ============================================================

K_barcode = [0.0, 5.0, 15.0]

rep_barcode = 0
t_barcode = -1

fig, axes = plt.subplots(
    3,
    2,
    figsize=(10.2, 6.0),
    sharex=False
)

for row, K in enumerate(K_barcode):

    j = np.where(np.isclose(model.K_values, K))[0][0]

    theta = theta_values[rep_barcode, j][t_barcode]

    D = model.synchrony_distance_from_theta(theta)

    dgms, _, _ = pe_from_distance_matrix(D, dim=1)

    dgms[0] = np.asarray(dgms[0], dtype=float).copy()
    dgms[0][np.isinf(dgms[0][:, 1]), 1] = np.max(D)

    plot_barcode(
        dgms[0],
        ax=axes[row, 0],
        title=rf"$K = {K:g}$",
        show_xlabel=(row == 2),
        show_ylabel=True,
        xlim=(0, 0.03)
    )

    plot_barcode(
        dgms[1],
        ax=axes[row, 1],
        title=rf"$K = {K:g}$",
        show_xlabel=(row == 2),
        show_ylabel=False,
        xlim=(0, 0.5)
    )

fig.text(
    0.285, 0.94,
    r"$H_0$ barcode",
    ha="center",
    va="bottom",
    fontsize=10
)

fig.text(
    0.765, 0.94,
    r"$H_1$ barcode",
    ha="center",
    va="bottom",
    fontsize=10
)

fig.subplots_adjust(
    left=0.09,
    right=0.98,
    bottom=0.09,
    top=0.88,
    hspace=0.55,
    wspace=0.25
)

plt.savefig(
    "kuramoto_barcodes.pdf",
    format="pdf",
    bbox_inches="tight"
)

plt.show()
plt.close(fig)