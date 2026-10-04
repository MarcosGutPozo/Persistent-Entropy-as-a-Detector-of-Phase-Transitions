## Persistent Entropy as a Detector of Phase Transitions

Code accompanying the paper "Persistent Entropy as a Detector of Phase Transitions".

Abstract: Persistent entropy is a scalar summary of persistence barcodes widely used to detect regime changes, yet there is no account of when a structural change in a barcode must produce a detectable change in entropy. We establish a model-agnostic theorem supplying such conditions. Treating persistence diagrams as random objects indexed by a control parameter, we identify a dispersion–condensation mechanism in the normalized persistence weights and derive an explicit lower bound on the entropy difference between the two regimes, valid with high probability at finite sample size and insensitive to the absolute scale of bar lifetimes. We also give a procedure for verifying the hypotheses on empirical barcodes. Applied to convolutional networks, the
criterion shows that the circular organization of learned filters reported by Gabrielsson and Carlsson emerges through a sharp topological phase transition, and locates its onset: within a few hundred iterations on MNIST, but an order of magnitude later on CIFAR-10. The same criterion detects the Kuramoto synchronization and Vicsek order–disorder transitions.

The repository implements the dispersion--condensation detection procedure and contains the code used for the numerical experiments in the paper. The criterion is applied to convolutional neural networks, the Kuramoto model, and the Vicsek model.

## Installation

The code requires **Python 3.9 or later**.

First, clone this repository and navigate to the project directory:

```bash
git clone <repository-url>
cd <repository-name>
```

Install the required dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```
`requirements.txt` contains all the Python packages required to run the experiments and reproduce the results presented in the paper.

## Repository structure

```text
phase_transitions/
├── README.md
├── requirements.txt
├── src/
│   └── phase_transitions/
│       ├── cnet.py
│       ├── mnet.py
│       ├── kuramoto.py
│       ├── vicsek.py
│       └── general/
│           ├── cnn.py
│           ├── density_filtration.py
│           ├── persistent_entropy.py
│           ├── phase_detection_algorithm.py
│           └── plot_mean_std_curves.py
└── experiments/
    ├── run_cnet.py
    ├── run_mnet.py
    ├── run_kuramoto.py
    └── run_vicsek.py
```

## Source Code

The `src/phase_transitions/` directory contains the implementations used throughout the experiments.

* **`cnet.py`** — Convolutional network architecture used for the MNIST experiment.
* **`mnet.py`** — Convolutional network architecture used for the CIFAR-10 experiment.
* **`kuramoto.py`** — Implementation of the Kuramoto model and related simulation utilities.
* **`vicsek.py`** — Implementation of the Vicsek model and related simulation utilities.

The `general/` directory contains common tools used across the experiments:

* **`cnn.py`** — Utilities for training and topological analysis of convolutional networks.
* **`density_filtration.py`** — Construction of density-based filtrations.
* **`persistent_entropy.py`** — Computation of persistence weights and persistent entropy.
* **`phase_detection_algorithm.py`** — Implementation of the dispersion--condensation detection procedure.
* **`plot_mean_std_curves.py`** — Utilities for plotting means and standard deviations across realizations.

## Experiments

The `experiments/` directory contains the scripts used to run the numerical experiments reported in the paper.

* **`run_mnet.py`** — MNIST/CNN experiment.
* **`run_cnet.py`** — CIFAR-10/CNN experiment.
* **`run_kuramoto.py`** — Kuramoto model experiment.
* **`run_vicsek.py`** — Vicsek model experiment.

## Citation

If you use this code, please cite the following paper:

```bibtex
@article{GutiérrezDelPozo2026,
  title   = {Persistent Entropy as a Detector of Phase Transitions},
  author  = {Gutiérrez-del-Pozo, M. and Ruco, M. and Paluzo-Hidalgo, E.},
  year    = {2026}
}
```
