import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.phase_transitions.kuramoto.kuramoto import Kuramoto
from src.phase_transitions.general.persistent_entropy import pe_from_distance_matrix, plot_barcode
from src.phase_transitions.general.phase_detection_algorithm import phase_detection, plot_phases
from src.phase_transitions.general.plot_mean_std_curves import plot_mean_std_curves

# Fix the random seed to ensure reproducibility of the simulations.
np.random.seed(42)


# ============================================================
# PARAMETERS
# ============================================================

# Number of oscillators in the Kuramoto network.
N = 50

# Initial and final simulation times.
t0 = 0.0
t1 = 10.0

# Number of time steps used to discretize the simulation.
steps = 500

# Range of coupling strengths K explored in the experiment.
# The coupling strength controls the interaction between oscillators
# and therefore the degree of synchronization in the system.
K_values = np.arange(
    0.0,
    16.0 + 1e-9,
    0.5
)

# Selected coupling strengths used for visualization.
requested_K = [
    0.0,
    1.0,
    3.0,
    5.0,
    10.0,
    15.0
]

# Number of independent realizations performed for each value of K.
R_K = 10

# Coupling strength used for the detailed topological analysis
# and phase detection.
K_selected = 10

# ============================================================
# MODEL
# ============================================================

# Initialize the Kuramoto model with the parameters defined above.
model = Kuramoto(
    N=N,
    t0=t0,
    t1=t1,
    steps=steps,
    K_values=K_values,
    R_k=R_K,
    K_selected=K_selected
)


# Display the main parameters of the Kuramoto model.
print("Kuramoto model")
print("----------------")
print(f"N       = {model.N}")
print(f"t0      = {model.t0}")
print(f"t1      = {model.t1}")
print(f"steps   = {model.steps}")
print(f"dt      = {model.dt}")
print(f"R_K     = {model.R_k}")
print(f"K values = {model.K_values}")

# ============================================================
# SIMULATIONS
# ============================================================

# Dictionaries used to store the synchronization parameter r(t)
# and the persistence entropy PE(H0)(t) for each coupling strength K.
kuramoto_r = {}
kuramoto_pe = {}

# Construct the time vector corresponding to the simulation steps.
kuramoto_time = (
    model.t0
    + np.arange(model.steps) * model.dt
)


# Perform a parameter sweep over the different coupling strengths K.
# For each K, several independent realizations are simulated in order
# to characterize the average behavior and its variability.
for K in tqdm(
    model.K_values,
    desc="Kuramoto sweep"
):

    all_r = []
    all_pe = []

    # Perform multiple independent realizations for each value of K.
    for realization in range(model.R_k):

        # ----------------------------------------------------
        # 1. Generate network
        # ----------------------------------------------------

        # Generate a random adjacency matrix defining the interaction
        # network between the oscillators.
        G = model.random_adjacency()


        # ----------------------------------------------------
        # 2. Initial conditions
        # ----------------------------------------------------

        # Sample the initial natural frequencies and phases
        # of the oscillators.
        omega0, theta0 = model.sample_kuramoto_initial()


        # ----------------------------------------------------
        # 3. Simulate Kuramoto
        # ----------------------------------------------------

        # Simulate the temporal evolution of the oscillator phases
        # and compute the synchronization order parameter r(t).
        theta_ts, r_ts = model.simulate_kuramoto(
            G=G,
            omega0=omega0,
            theta0=theta0,
            K=float(K)
        )


        # ----------------------------------------------------
        # 4. Store r(t)
        # ----------------------------------------------------

        # Store the synchronization parameter for this realization.
        all_r.append(r_ts)


        # ----------------------------------------------------
        # 5. Compute PE(H0)(t)
        # ----------------------------------------------------

        # Compute the temporal evolution of the persistence entropy associated with H0.
        pe0, _ = model.kuramoto_pe_h0_series(
            theta_ts,
            normalize=False
        )

        all_pe.append(pe0)


    # --------------------------------------------------------
    # Store all realizations for this K
    # --------------------------------------------------------

    kuramoto_r[float(K)] = np.stack(
        all_r,
        axis=0
    )

    kuramoto_pe[float(K)] = np.stack(
        all_pe,
        axis=0
    ) 

# ============================================================
# PLOT PE(H0)
# ============================================================

K_subset = [float(K) for K in requested_K if float(K) in kuramoto_pe]

plot_mean_std_curves(
    kuramoto_time,
    kuramoto_pe,
    K_subset,
    title=(
        "Kuramoto: mean PE(H0)(t) "
        "± std across realizations"
    ),
    ylabel="PE(H0)"
)


# ============================================================
# PLOT r(t)
# ============================================================

K_subset = [float(K) for K in requested_K if float(K) in kuramoto_r]

plot_mean_std_curves(
    kuramoto_time,
    kuramoto_r,
    K_subset,
    title=(
        "Kuramoto: mean r(t) "
        "± std across realizations"
    ),
    ylabel="r(t)"
)

# ============================================================
# DETAILED TOPOLOGICAL ANALYSIS
# ============================================================

# Generate one independent realization of the Kuramoto network
# at the selected coupling strength K_selected.

G = model.random_adjacency()
omega0, theta0 = model.sample_kuramoto_initial()

# Simulate the selected realization to obtain the temporal evolution
# of the oscillator phases and the synchronization parameter.
theta_ts, r_ts = model.simulate_kuramoto(G, omega0, theta0, float(K_selected))

# Compute the temporal evolution of the H0 persistence entropy
# and the corresponding persistence weights.
#
# A minimum persistence threshold of 0.01 is used to remove
# short-lived topological features and focus on the most relevant
# structures.
pe0, weight0 = model.kuramoto_pe_h0_series(theta_ts, normalize=False, min_persistence=0.01)

pe = np.array(pe0)

# Plot the temporal evolution of persistence entropy
# for the selected realization.
plt.plot(kuramoto_time, pe)
plt.title(fr"Kuramoto: Temporal evolution of entropy for a realization with $K={K_selected}$")
plt.xlabel("Time")
plt.ylabel(r"$PE(H_0)(t)$")
plt.xlim(model.t0, model.t1)
plt.show()

# Select representative time points at which the H0 and H1
# persistence barcodes will be visualized.
fig, axes = plt.subplots(4, 2, figsize=(12, 12))
t_list=[0,25,60,200]
for row, ti in enumerate(t_list):
    theta_i = theta_ts[ti]
    D = model.synchrony_distance_from_theta(theta_i)
    dgms_H0, _ , _ = pe_from_distance_matrix(D, dim=0)
    dgms_H1, _ , _ = pe_from_distance_matrix(D, dim=1)
    
    ax0 = axes[row, 0]

    plot_barcode(
        dgms_H0,
        ax=ax0,
        title=f"Kuramoto H0 barcode (t={ti*model.dt})"
    )

    ax0.set_xlim(0, 0.03)
    ax0.set_xticks(np.linspace(0, 0.03, 6))

    ax1 = axes[row, 1]

    plot_barcode(
        dgms_H1,
        ax=ax1,
        title=f"Kuramoto H1 barcode (t={ti*model.dt})"
    )

    ax1.set_xlim(0, 0.5)
    ax1.set_xticks(np.linspace(0, 0.5, 6))

plt.tight_layout()
plt.show()

# ============================================================
# PHASE DETECTION
# ============================================================

# Apply the phase detection algorithm to the persistence weights
# in order to identify the different dynamical phases of the
# Kuramoto system and determine the optimal parameters for their
# separation.
best_c, best_m, best_k, _, _, _, results_distrib, results_m, results_M = phase_detection(weight0, k_max=1, percent_disp=0.1, percent_cond=0)

# Plot the persistence entropy together with the phases
# detected by the phase detection algorithm.
plot_phases(best_c, best_k, best_m, results_distrib, results_m, results_M, pe, kuramoto_time, experiment = 'Kuramoto', dim='0')


