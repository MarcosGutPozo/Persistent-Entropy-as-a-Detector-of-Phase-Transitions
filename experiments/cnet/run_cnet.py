import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from src.phase_transitions.general.cnn import set_seed, make_loaders, run_experiment
from src.phase_transitions.general.density_filtration import density_filtration
from src.phase_transitions.general.persistent_entropy import persistence_weights, persistent_entropy, pe_from_point_cloud, plot_barcode
from src.phase_transitions.general.phase_detection_algorithm import phase_detection, plot_phases

from src.phase_transitions.cnet.cnet import CNet

# Train the CNet neural network on the CIFAR-10 dataset
# and study the emergence of dispersed and condensed phases during the training process.

device = "cuda" if torch.cuda.is_available() else "cpu"

# Define 60 different random seeds.
# Each seed is used to initialize and train an independent CNet model.
SEEDS = range(60)
all_snapshots = {}

# Select the CIFAR-10 dataset and create the training and test data loaders.
# The training data are divided into batches of 124 samples.
# The test loader is used to evaluate the network during training.
dataset='cifar10'
train_loader, test_loader, input_size, num_classes = make_loaders(dataset, batch_size=124, num_workers=4)
criterion = nn.CrossEntropyLoss()

# Run the neural network with the CNet architecture for each random seed
for seed in SEEDS:

    set_seed(seed)

    # Initialize the CNet architecture.
    # X=64, Y=32, and Z=64 define the dimensions of the CNet layers.
    model = CNet(in_channels=input_size[0], input_size=input_size[1], X=64, Y=32, Z=64, num_classes=num_classes)

    # Use stochastic gradient descent (SGD) to optimize the network
    # parameters during training.
    # lr=0.01:
    #     Learning rate controlling the size of each parameter update.
    # momentum=0.9:
    #     Momentum accelerates learning in consistent directions
    #     and can help the optimization process converge more efficiently.
    # weight_decay=1e-4:
    #     L2 regularization that penalizes large weights and helps
    #     reduce overfitting.
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

     # Train the network and store the filters, loss, and accuracy at the specified iterations
    history = run_experiment(model, train_loader, test_loader, optimizer,
        criterion,
        device,
        save_every= 100,
        iterations= 20000)

    for i, it in enumerate(history["iteration"]):

        filters_layer1 = history["filters_layer1"][i]
        filters_layer2 = history["filters_layer2"][i]
        loss = history["loss"][i]           
        accuracy = history["val_acc"][i]

        if it not in all_snapshots:
            all_snapshots[it] = []

        # Store all relevant information for this seed and iteration
        all_snapshots[it].append({
            "seed": seed,
            "filters_layer1": filters_layer1,
            "filters_layer2": filters_layer2,
            "loss": loss,
            "accuracy": accuracy
        })

# Create a point cloud for each layer by combining the filters from all neural networks at each training iteration
point_cloud_layer1 = {}

for it in all_snapshots:
    filters_layer1 = [snapshot["filters_layer1"] for snapshot in all_snapshots[it]]
    
    point_cloud_layer1[it] = torch.cat(filters_layer1, dim=0)

point_cloud_layer2 = {}

for it in all_snapshots:
    filters_layer2 = [snapshot["filters_layer2"] for snapshot in all_snapshots[it]]
    
    point_cloud_layer2[it] = torch.cat(filters_layer2, dim=0)

# Compute persistence entropy and persistence diagrams from the point cloud at each training iteration.
# Apply density filtration as a preprocessing step to retain the points
# that are most relevant for capturing the underlying geometry of the space.
results_layer1 = {}

for it, pc in point_cloud_layer1.items():
    pc = pc.numpy()

    pc = density_filtration(pc, k=200, p=0.14)

    dgms, pes = pe_from_point_cloud(pc, dim=1, improve_time=False)

    results_layer1[it] = {
        "n_points": len(pc),
        "H1_PE": pes,
        "dgms_H1": dgms,
    }

results_layer2 = {}

for it, pc in point_cloud_layer2.items():
    pc = pc.numpy()

    pc = density_filtration(pc, k=15, p=0.1)

    dgms, pes = pe_from_point_cloud(pc, dim=1, improve_time=True)

    results_layer2[it] = {
        "n_points": len(pc),
        "H1_PE": pes,
        "dgms_H1": dgms,
    }

# Compute persistence weights and persistence entropy at each iteration 
# using a minimum persistence threshold of 0.13 for both layers.

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

weights_layer1 = {it: persistence_weights(results_layer1[it]['dgms_H1'], min_persistence=0.13) for it in results_layer1}
entropy_layer1 = {it: persistent_entropy(results_layer1[it]['dgms_H1'], min_persistence=0.13) for it in results_layer1} 
iterations1 = list(entropy_layer1.keys())
values1 = list(entropy_layer1.values())

axes[0].plot(iterations1, values1, marker=None, color='royalblue')
axes[0].set_title(r"CIFAR-10 C(64,32,64) - Layer 1: Evolution of entropy with a 0.13 lifetime threshold")
axes[0].set_xlabel("Iteration (I)")
axes[0].set_ylabel("PE(H1)(I)")
axes[0].grid(False)

weights_layer2 = {it: persistence_weights(results_layer2[it]['dgms_H1'], min_persistence=0.13) for it in results_layer2}
entropy_layer2 = {it: persistent_entropy(results_layer2[it]['dgms_H1'], min_persistence=0.13) for it in results_layer2} 
iterations2 = list(entropy_layer2.keys())
values2 = list(entropy_layer2.values())

axes[1].plot(iterations2, values2, marker=None, color='royalblue')
axes[1].set_title(r"CIFAR-10 C(64,32,64) - Layer 2: Evolution of entropy with a 0.13 lifetime threshold")
axes[1].set_xlabel("Iteration (I)")
axes[1].set_ylabel("PE(H1)(I)")
axes[1].grid(False)

plt.tight_layout()
plt.show()

# Plot the persistence barcodes for selected training iterations for both layers
fig, axes = plt.subplots(8, 2, figsize=(16, 12))

lista = [100, 500, 1000, 3000, 5000, 10000, 15000, 20000]

for row, idx in enumerate(lista):
    
    plot_barcode(
        results_layer1[idx]['dgms_H1'],
        ax=axes[row, 0],
        title=f'CIFAR-10 C(64,32,64) Layer 1: H1 barcode (Iteration={idx})'
    )
    
    plot_barcode(
        results_layer2[idx]['dgms_H1'],
        ax=axes[row, 1],
        title=f'CIFAR-10 C(64,32,64) Layer 2: H1 barcode (Iteration={idx})'
    )
    
    for col in [0, 1]:
        axes[row, col].set_xlim(0, 2)
        axes[row, col].set_xticks(np.linspace(0, 2, 5))
        # CAMBIO AQUÍ: Ahora dice "bar_index" en el eje vertical
        axes[row, col].set_ylabel("bar_index")

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()

# Apply the phase detection algorithm to identify the dispersed and condensed
# phases and determine their optimal parameters.
best_c1, best_m1, best_k1, _, _, _, results_distrib1, results_m1, results_M1 = phase_detection(experiment='cnn', k_max=3, percent_disp=0.02, percent_cond=0, point_cloud=results_layer1, min_persistence=0.13)
best_c2, best_m2, best_k2, _, _, _, results_distrib2, results_m2, results_M2 = phase_detection(experiment='cnn', k_max=3, percent_disp=0.01, percent_cond=0, point_cloud=results_layer2, min_persistence=0.13)

items1 = sorted(entropy_layer1.items())
iterations1 = np.array([k for k, _ in items1])
values1 = np.array([v for _, v in items1])

items2 = sorted(entropy_layer2.items())
iterations2 = np.array([k for k, _ in items2])
values2 = np.array([v for _, v in items2])

# Plot the detected training phases for both layers.
plot_phases(best_c1, best_k1, best_m1, results_distrib1, results_m1, results_M1, values1, iterations1, experiment = 'CIFAR-10 C(64,32,64) - Layer 1', dim='1')
plot_phases(best_c2, best_k2, best_m2, results_distrib2, results_m2, results_M2, values2, iterations2, experiment = 'CIFAR-10 C(64,32,64) - Layer 2', dim='1')