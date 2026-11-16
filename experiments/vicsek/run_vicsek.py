import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.phase_transitions.vicsek.vicsek import Vicsek
from src.phase_transitions.general.persistent_entropy import pe_from_distance_matrix, plot_barcode
from src.phase_transitions.general.phase_detection_algorithm import detect_solutions, plot_solution_probabilities, plot_phase_detection
from src.phase_transitions.general.plot_mean_std_curves import plot_mean_std_curves


np.random.seed(42)


# ============================================================
# Paramaters
# ============================================================

L = 5.0
v0 = 1.0
dt = 0.1
steps = 300
num_particles = 300

eta_values = np.concatenate([
    np.arange(0.01, 1.00, 0.01),
    np.arange(1.00, 5.01, 0.05)
])

eta_selected = [0.05, 0.1, 0.5, 1, 3, 5]
eta_barcode = [0.1, 1, 5]

R_eta_train = 100
R_eta_test = 100

r_int = 0.5

delta = 0.1
alpha = 0.05

rep_barcode = 0
t_barcode = -1


# ============================================================
# Model Vicsek
# ============================================================

model = Vicsek(
    L=L,
    v0=v0,
    steps=steps,
    dt=dt,
    num_particles=num_particles,
    eta_values=eta_values,
    R_eta_train=R_eta_train,
    R_eta_test=R_eta_test,
    r_int=r_int,
    eta_selected=eta_selected
)

vicsek_time = np.arange(model.steps) * model.dt


# Indices of selected eta values within eta_values
selected_indices = [np.where(np.isclose(model.eta_values, eta))[0][0] for eta in model.eta_selected]

# ============================================================
# 1. TRAINING SIMULATIONS
# ============================================================

# Store temporal curves only for selected eta values.
vicsek_psi = {float(eta): np.empty((model.R_eta_train, model.steps), dtype=np.float32) for eta in model.eta_selected}

vicsek_pe = {float(eta): np.empty((model.R_eta_train, model.steps), dtype=np.float32) for eta in model.eta_selected}

# Store theta for all eta values.
theta_values = np.empty((model.R_eta_train, len(model.eta_values)), dtype=object)

for r in range(model.R_eta_train):

    for j, eta in enumerate(tqdm(model.eta_values, desc=f"Training {r+1}/{model.R_eta_train}")):

        pos_ts, theta_ts, psi_ts = model.simulate_vicsek(float(eta))

        theta_values[r, j] = theta_ts

        # Calculate temporal quantities only for selected eta.
        if j in selected_indices:

            vicsek_psi[float(eta)][r, :] = psi_ts

            pe0, _ = model.vicsek_pe_h0_series(theta_ts, min_persistence=0.01)

            vicsek_pe[float(eta)][r, :] = pe0


plot_mean_std_curves(
    x=vicsek_time,
    data_left=vicsek_psi,
    data_right=vicsek_pe,
    params=model.eta_selected,
    left_ylabel=r"$\psi(t)$",
    right_ylabel=r"PE($H_0$)",
    param_label=r"\eta",
    xlabel="Time",
    filename="vicsek_psi_pe.pdf"
)


# ============================================================
# Training weights
# ============================================================

all_weights_training = np.empty((model.R_eta_train, len(model.eta_values)), dtype=object)

for r in range(model.R_eta_train):

    for j in range(len(model.eta_values)):

        theta = theta_values[r, j][-1]

        D = model.distance_velocity(theta)

        _, weight, _ = pe_from_distance_matrix(D, dim=0, min_persistence=0.01)

        all_weights_training[r, j] = weight


# ============================================================
# Test weights
# ============================================================

theta_values_test = np.empty((model.R_eta_test, len(model.eta_values)), dtype=object)

all_weights_test = np.empty((model.R_eta_test, len(model.eta_values)), dtype=object)

entropy_test = np.empty((model.R_eta_test, len(model.eta_values)))

for r in range(model.R_eta_test):

    for j, eta in enumerate(tqdm(model.eta_values, desc=f"Test {r+1}/{model.R_eta_test}")):

        pos_ts, theta_ts, psi_ts = model.simulate_vicsek(float(eta))

        theta_values_test[r, j] = theta_ts

        D = model.distance_velocity(theta_ts[-1])

        _, weight, pe = pe_from_distance_matrix(D, dim=0, min_persistence=0.01)

        all_weights_test[r, j] = weight

        entropy_test[r, j] = pe


# ============================================================
# Phase detection
# ============================================================

best_solutions = detect_solutions(
    all_weights_training,
    delta=delta,
    alpha=alpha,
    kmax=3
)


plot_solution_probabilities(
    all_weights_test,
    best_solutions,
    delta=delta,
    alpha=alpha,
    param_values=model.eta_values,
    param_symbol=r"\eta",
    filename="vicsek_robustness"
)


plot_phase_detection(
    all_weights_test,
    best_solutions[0],
    delta=delta,
    alpha=alpha,
    entropies=entropy_test,
    param_values=model.eta_values,
    param_symbol=r"\eta",
    delta_values=(0.15, 0.20, 0.30),
    ylabel=r"PE($H_0$)",
    filename="vicsek_phase_detection"
)


# ============================================================
# Barcodes
# ============================================================

fig, axes = plt.subplots(
    3,
    2,
    figsize=(10.2, 6.0),
    sharex=False
)

for row, eta in enumerate(eta_barcode):

    j = np.where(np.isclose(model.eta_values, eta))[0][0]

    theta = theta_values[rep_barcode, j][t_barcode]

    D = model.distance_velocity(theta)

    dgms, _, _ = pe_from_distance_matrix(D, dim=1)

    dgms[0] = np.asarray(dgms[0], dtype=float).copy()
    dgms[0][np.isinf(dgms[0][:, 1]), 1] = np.max(D)

    plot_barcode(
        dgms[0],
        ax=axes[row, 0],
        title=rf"$\eta = {eta:g}$",
        show_xlabel=(row == 2),
        show_ylabel=True,
        xlim=(0, 0.03)
    )

    plot_barcode(
        dgms[1],
        ax=axes[row, 1],
        title=rf"$\eta = {eta:g}$",
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
    "vicsek_barcodes.pdf",
    format="pdf",
    bbox_inches="tight"
)

plt.show()
plt.close(fig)
