## Persistent Entropy as a Detector of Phase Transitions

Code accompanying the paper "Persistent Entropy as a Detector of Phase Transitions".

Abstract: Persistent entropy is a scalar summary of persistence barcodes widely used to detect regime changes, yet there is no account of when a structural change in a barcode must
produce a detectable change in entropy. We establish a model-agnostic theorem supplying such conditions, based on the distinction between dispersed diagrams, where
persistence mass is distributed across several features, and condensed diagrams, where it is concentrated in a few dominant features. For any such pair, we derive an explicit
lower bound on their entropy difference, depending only on the normalized persistence weights. Since diagrams depend on random initial conditions, we formulate the
framework probabilistically: if each regime holds with probability at least 1 − δ across independent realizations, the gap holds with probability at least 1−2δ for a fixed num-
ber of points, and it persists in the large-sample limit under a mild regularity condition. We also give a procedure that identifies both regimes on empirical barcodes and certi-
fies them across realizations. Applied to convolutional networks, the criterion shows that the dominant H1 class of the first-layer filters, which Gabrielsson and Carlsson
interpret as a circular organization, emerges through a dispersion–condensation transition, and locates it: the condensed regime is certified from about one thousand training
iterations on MNIST but from about thirteen thousand on CIFAR-10, once test accuracy has largely saturated. The same criterion detects the Kuramoto synchronization
and Vicsek order-disorder transitions

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
