import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.phase_transitions.vicsek.vicsek import Vicsek
from src.phase_transitions.general.persistent_entropy import pe_from_distance_matrix, plot_barcode
from src.phase_transitions.general.phase_detection_algorithm import phase_detection, plot_phases

# Fix the random seed to ensure reproducibility of the simulations.
np.random.seed(42)

# ============================================================
# PARAMETERS
# ============================================================

# Size of the two-dimensional simulation domain.
L = 5.0

# Constant particle speed.
v0 = 1.0

# Time step used in the numerical integration.
dt_v = 0.1

# Number of time steps in each simulation.
steps = 500

# Number of self-propelled particles in the system.
num_particles = 300

# Range of noise strengths explored in the experiment.
eta_values = [
    0.05,
    0.1,
    0.5,
    1,
    3,
    5
]

# Number of independent realizations performed for each noise strength.
R_eta = 5

# Interaction radius used to determine neighboring particles.
r_int = 0.5

# Selected noise strength used for the detailed topological analysis
# and phase detection.
eta_selected = 0.05

# ============================================================
# MODEL
# ============================================================

# Initialize the Vicsek model using the parameters defined above.
model = Vicsek(
    L=L,
    v0=v0,
    steps=steps,
    dt_v=dt_v,
    num_particles=num_particles,
    eta_values=eta_values,
    R_eta=R_eta,
    r_int=r_int,
    eta_selected=eta_selected
)


# Display the main parameters of the Vicsek model.
print("Vicsek model")
print("----------------")
print(f"L       = {model.L}")
print(f"v0      = {model.v0}")
print(f"steps   = {model.steps}")
print(f"dt_v    = {model.dt_v}")
print(f"num_particles = {model.num_particles}")
print(f"r_int   = {model.r_int}")
print(f"R_eta   = {model.R_eta}")
print(f"eta values = {model.eta_values}")

# ============================================================
# SIMULATIONS
# ============================================================

# Dictionaries used to store the global polarization parameter
# psi(t) and the persistence entropy PE(H0)(t) for each noise strength.
vicsek_psi = {}
vicsek_pe = {}

# Construct the time vector corresponding to the simulation steps.
vicsek_time = (
    np.arange(model.steps)
    * model.dt_v
)


# Perform a parameter sweep over the different noise strengths eta.
for eta in tqdm(
    model.eta_values,
    desc="Vicsek sweep"
):

    all_psi = []
    all_pe = []


    # Perform multiple independent realizations for each noise strength.
    for realization in range(model.R_eta):

        # ----------------------------------------------------
        # Simulate Vicsek
        # ----------------------------------------------------

        # Simulate the temporal evolution of the particle positions,
        # particle orientations, and global polarization.
        pos_ts, theta_ts, psi_ts = model.simulate_vicsek(
            float(eta)
        )


        # ----------------------------------------------------
        # Store psi(t)
        # ----------------------------------------------------

        # Store the global polarization parameter for this realization.
        all_psi.append(
            psi_ts
        )


        # ----------------------------------------------------
        # Compute PE(H0)(t)
        # ----------------------------------------------------

        # Compute the temporal evolution of the H0 persistence entropy
        # associated with the particle orientation configuration.
        pe0, _ = model.vicsek_pe_h0_series(
            theta_ts,
            normalize=False
        )

        all_pe.append(
            pe0
        )


    # --------------------------------------------------------
    # Store all realizations for this eta
    # --------------------------------------------------------

    # Store all independent realizations for the current noise strength.
    vicsek_psi[float(eta)] = np.stack(
        all_psi,
        axis=0
    )

    vicsek_pe[float(eta)] = np.stack(
        all_pe,
        axis=0
    )

# ============================================================
# SELECT K VALUES FOR PLOTS
# ============================================================

eta_subset = [float(eta) for eta in eta_values]


# ============================================================
# PLOT PE(H0)
# ============================================================

eta_subset = [float(eta) for eta in eta_values]

plot_mean_std_curves(
    vicsek_time,
    vicsek_pe,
    eta_subset,
    title=(
        "Vicsek: mean PE(H0)(t) "
        "± std across realizations"
    ),
    ylabel="PE(H0)"
)


# ============================================================
# PLOT psi(t)
# ============================================================

plot_mean_std_curves(
    vicsek_time,
    vicsek_psi,
    eta_subset,
    title=r"Vicsek: mean $\psi(t) \pm$ std across realizations",
    ylabel=r"$\psi(t)$")

# APPLY ALGORITHM

pos_ts, theta_ts, psi_ts = model.simulate_vicsek(float(eta_selected))
pe0, weight0 = model.vicsek_pe_h0_series(theta_ts, normalize= False, min_persistence=0.01)

pe = np.array(pe0)

plt.plot(vicsek_time, pe)
plt.title(fr"Vicsek: Temporal evolution of entropy for a realization with $eta={eta_selected}$")
plt.xlabel("Time")
plt.ylabel(r"$PE(H_0)(t)$")
plt.show()

fig, axes = plt.subplots(4, 2, figsize=(12, 12))
t_list=[0,10,50,200]
for row, ti in enumerate(t_list):
    theta_i = theta_ts[ti]
    D = model.distance_velocity(theta_i)
    dgms_H0, _, _ = pe_from_distance_matrix(D, dim=0)
    dgms_H1, _, _ = pe_from_distance_matrix(D, dim=1)

    ax0 = axes[row, 0]

    plot_barcode(
        dgms_H0,
        ax=ax0,
        title=f"Vicsek H0 barcode (t={ti*model.dt_v})"
    )

    ax0.set_xlim(0, 0.03)
    ax0.set_xticks(np.linspace(0, 0.03, 6))

    # =========================
    # H1
    # =========================

    ax1 = axes[row, 1]

    plot_barcode(
        dgms_H1,
        ax=ax1,
        title=f"Vicsek H1 barcode (t={ti*model.dt_v})"
    )

    ax1.set_xlim(0, 0.5)
    ax1.set_xticks(np.linspace(0, 0.5, 6))

plt.tight_layout()
plt.show()

best_c, best_m, best_k, _, _, _, results_distrib, results_m, results_M = phase_detection(weight0, k_max=1, percent_disp=0.1, percent_cond=0)

plot_phases(best_c, best_k, best_m, results_distrib, results_m, results_M, pe, vicsek_time, experiment = 'Vicsek', dim='0')
 