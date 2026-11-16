import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from src.phase_transitions.general.cnn import set_seed, make_loaders, run_experiment
from src.phase_transitions.general.density_filtration import density_filtration
from src.phase_transitions.general.persistent_entropy import persistence_weights, persistent_entropy, pe_from_point_cloud, plot_barcode
from src.phase_transitions.general.phase_detection_algorithm import plot_solution_probabilities, plot_phase_detection, detect_solutions

from src.phase_transitions.mnet.mnet import MNet


device = "cuda" if torch.cuda.is_available() else "cpu"

# ==================================================
# Parameters
# ==================================================

N_REALIZATIONS = 40
N_NETWORKS = 100
N_ITERATIONS = 2000
SAVE_EVERY = 100

BASE_SEED = 0

dataset = "mnist"

train_loader, test_loader, input_size, num_classes = make_loaders(dataset, batch_size=128, num_workers=4)

criterion = nn.CrossEntropyLoss()

# ==================================================
# Train independent 40 realizations
# ==================================================

all_realizations = []

for realization in range(N_REALIZATIONS):

    print(f"\n========== REALIZATION {realization + 1}/{N_REALIZATIONS} ==========")

    # Store snapshots of the 100 networks
    all_snapshots = {}

    for network in range(N_NETWORKS):

        seed = BASE_SEED + realization * N_NETWORKS + network

        set_seed(seed)

        model = MNet(
            in_channels=input_size[0],
            input_size=input_size[1],
            X=64,
            Y=32,
            Z=64,
            num_classes=num_classes
        )

        optimizer = optim.Adam(
            model.parameters(),
            lr=1e-3,
            weight_decay=0.0
        )

        history = run_experiment(
            model,
            train_loader,
            test_loader,
            optimizer,
            criterion,
            device,
            save_every=SAVE_EVERY,
            iterations=N_ITERATIONS
        )

        for i, it in enumerate(history["iteration"]):

            if it not in all_snapshots:
                all_snapshots[it] = []

            all_snapshots[it].append({
                "seed": seed,
                "filters_layer1": history["filters_layer1"][i],
                "filters_layer2": history["filters_layer2"][i],
                "loss": history["loss"][i],
                "accuracy": history["val_acc"][i]
            })

        print(
            f"Realization {realization + 1} | "
            f"Network {network + 1}/{N_NETWORKS}"
        )

    # --------------------------------------------------
    # Build point clouds joining all the networks per realization
    # --------------------------------------------------

    point_cloud_layer1 = {}
    point_cloud_layer2 = {}

    for it in all_snapshots:

        point_cloud_layer1[it] = torch.cat(
            [s["filters_layer1"] for s in all_snapshots[it]],
            dim=0
        )

        point_cloud_layer2[it] = torch.cat(
            [s["filters_layer2"] for s in all_snapshots[it]],
            dim=0
        )

    # --------------------------------------------------
    # Compute persistence entropy
    # --------------------------------------------------

    results_layer1 = {}

    for it, pc in point_cloud_layer1.items():

        pc = pc.numpy()
        pc = density_filtration(pc, k=200, p=0.3)

        dgms, pes = pe_from_point_cloud(
            pc, dim=1, improve_time=False
        )

        results_layer1[it] = {
            "n_points": len(pc),
            "H1_PE": pes,
            "dgms_H1": dgms,
        }

    results_layer2 = {}

    for it, pc in point_cloud_layer2.items():

        pc = pc.numpy()
        pc = density_filtration(pc, k=10, p=0.1)

        dgms, pes = pe_from_point_cloud(
            pc, dim=1, improve_time=True
        )

        results_layer2[it] = {
            "n_points": len(pc),
            "H1_PE": pes,
            "dgms_H1": dgms,
        }

    # --------------------------------------------------
    # Store all the realizations
    # --------------------------------------------------

    all_realizations.append({

        "realization": realization,

        "snapshots": all_snapshots,

        "point_cloud_layer1": point_cloud_layer1,
        "point_cloud_layer2": point_cloud_layer2,

        "results_layer1": results_layer1,
        "results_layer2": results_layer2,

    })

weights_layer1 = []
entropy_layer1 = []

weights_layer2 = []
entropy_layer2 = []

for realization in all_realizations:

    w1 = {
        it: persistence_weights(
            realization["results_layer1"][it]["dgms_H1"],
            min_persistence=0.15
        )
        for it in realization["results_layer1"]
    }

    e1 = {
        it: persistent_entropy(
            realization["results_layer1"][it]["dgms_H1"],
            min_persistence=0.15
        )
        for it in realization["results_layer1"]
    }

    weights_layer1.append(w1)
    entropy_layer1.append(e1)

    w2 = {
        it: persistence_weights(
            realization["results_layer2"][it]["dgms_H1"],
            min_persistence=0.15
        )
        for it in realization["results_layer2"]
    }

    e2 = {
        it: persistent_entropy(
            realization["results_layer2"][it]["dgms_H1"],
            min_persistence=0.15
        )
        for it in realization["results_layer2"]
    }

    weights_layer2.append(w2)
    entropy_layer2.append(e2)


iterations = np.array(sorted(weights_layer1[0].keys()))

R = len(all_realizations)
N = len(iterations)

all_weights_layer1 = np.empty((R, N), dtype=object)
all_weights_layer2 = np.empty((R, N), dtype=object)

all_entropy_layer1 = np.zeros((R, N))
all_entropy_layer2 = np.zeros((R, N))

for r in range(R):
    for j, it in enumerate(iterations):

        all_weights_layer1[r, j] = weights_layer1[r][it]
        all_weights_layer2[r, j] = weights_layer2[r][it]

        all_entropy_layer1[r, j] = entropy_layer1[r][it]
        all_entropy_layer2[r, j] = entropy_layer2[r][it]

# ==================================================
# Phase detection method
# ==================================================

best_solutions_layer1 = detect_solutions(
    all_weights_layer1,
    delta=0.1,
    alpha=0.05,
    kmax=2
)

best_solutions_layer2 = detect_solutions(
    all_weights_layer2,
    delta=0.15,
    alpha=0.05,
    kmax=2
)

plot_solution_probabilities(
    all_weights_layer1,
    best_solutions_layer1,
    delta=0.1,
    alpha=0.05,
    param_values=iterations,
    param_symbol="I",
    filename="mnist_layer1_robustness"
)

plot_solution_probabilities(
    all_weights_layer2,
    best_solutions_layer2,
    delta=0.15,
    alpha=0.05,
    param_values=iterations,
    param_symbol="I",
    filename="mnist_layer2_robustness"
)


plot_phase_detection(
    all_weights_layer1,
    best_solutions_layer1[0],
    delta=0.1,
    alpha=0.05,
    entropies=all_entropy_layer1,
    param_values=iterations,
    param_symbol="I",
    delta_values=(0.15, 0.20, 0.30),
    ylabel=r"PE($H_1$)",
    filename="mnist_layer1_phase_detection"
)

plot_phase_detection(
    all_weights_layer2,
    best_solutions_layer2[0],
    delta=0.15,
    alpha=0.05,
    entropies=all_entropy_layer2,
    param_values=iterations,
    param_symbol="I",
    delta_values=(0.20, 0.25, 0.30),
    ylabel=r"PE($H_1$)",
    filename="mnist_layer2_phase_detection"
)

# --------------------------------------------------
# Barcodes
# --------------------------------------------------

REALIZATION = 0
BARCODE_ITERATIONS = [100, 500, 2000]

fig, axes = plt.subplots(
    len(BARCODE_ITERATIONS), 2,
    figsize=(11, 10)
)

for i, it in enumerate(BARCODE_ITERATIONS):

    dgm1 = all_realizations[REALIZATION]["results_layer1"][it]["dgms_H1"][1]

    plot_barcode(
        dgm1,
        axes[i, 0],
        title=f"Layer 1 - H1 (iteration {it})",
        show_xlabel=(i == len(BARCODE_ITERATIONS) - 1),
        show_ylabel=True
    )

    dgm2 = all_realizations[REALIZATION]["results_layer2"][it]["dgms_H1"][1]

    plot_barcode(
        dgm2,
        axes[i, 1],
        title=f"Layer 2 - H1 (iteration {it})",
        show_xlabel=(i == len(BARCODE_ITERATIONS) - 1),
        show_ylabel=False
    )

plt.tight_layout()

plt.savefig(
    "mnist_barcodes.pdf",
    bbox_inches="tight"
)

plt.show()
