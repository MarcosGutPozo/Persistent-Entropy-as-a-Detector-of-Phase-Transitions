import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser


def persistence_weights(diagram, min_persistence=None, truncate=None):

    # Converts persistence lifetimes into normalized weights.
    dgm = np.asarray(diagram).copy()

    # Truncate infinite bars if requested
    if truncate is not None:
        inf_mask = np.isinf(dgm[:, 1])
        dgm[inf_mask, 1] = truncate

    lifetimes = dgm[:, 1] - dgm[:, 0]

    # Remove non-positive lifetimes
    lifetimes = lifetimes[lifetimes > 0]

    # Optional persistence threshold
    if min_persistence is not None:
        lifetimes = lifetimes[lifetimes > min_persistence]

    total = lifetimes.sum()

    if total <= 0:
        return np.array([])

    return lifetimes / total


def persistent_entropy(diagram, normalize=False, min_persistence=None, truncate=None):

    # Calculate persistent entropy from persistence lifetimes
    
    weights=persistence_weights(diagram, min_persistence=min_persistence, truncate=truncate)
    pe = -(weights * np.log(weights)).sum()

    if normalize:
        pe = pe / np.log(len(weights))

    return float(pe)

def plot_barcode(dgm, ax, title):
    # This function plots a barcode where each bar represents the lifetime of a topological feature.
    if dgm is None or len(dgm) == 0:
        ax.set_title(title + " (empty)")
        ax.set_xlabel("filtration value")
        ax.set_ylabel("bar index")
        return
    dgm = np.array(dgm)
    for i, (b, d) in enumerate(dgm):
        ax.plot([b, d], [i, i], lw=2, color='#1f77b4')
    ax.set_title(title)
    ax.set_xlabel("filtration value")
    ax.set_ylabel("bar index")
    ax.grid(True, alpha=0.3)  

def pe_from_point_cloud(point_cloud, dim=1, improve_time=False, n_perm=2000, normalize=False, min_persistence=None):
    # This function computes persistence diagrams and persistent entropy from a point cloud.
    if improve_time:
        res = ripser(point_cloud, maxdim=dim, n_perm=n_perm)
    else:
        res = ripser(point_cloud, maxdim=dim)

    dgms = res["dgms"]
    pe = persistent_entropy(dgms[dim], normalize=normalize, min_persistence=min_persistence) 
    return dgms, pe

def pe_from_distance_matrix(D, dim=0, normalize=False, min_persistence=None):
    # This function computes persistence diagrams, normalized weights and persistent entropy from distance matrix.
    res = ripser(D, distance_matrix=True, maxdim=dim)
    trunc = float(np.max(D)) if np.size(D) > 0 else 0.0
    dgms = res["dgms"]
    weights = persistence_weights(dgms[dim], min_persistence=min_persistence, truncate=trunc)
    pe = persistent_entropy(dgms[dim], normalize=normalize, min_persistence=min_persistence, truncate=trunc)
    return dgms, weights, pe  