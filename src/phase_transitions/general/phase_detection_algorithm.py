import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import beta


# =====================================================================
# 1. Entropy bounds
# =====================================================================
def g(x):
    return -x * math.log(x)


def Hdisp(c, m):
    """Entropy lower bound in the dispersed regime."""
    a = 1 - (m - 1) * c
    return -a * math.log(a) + (m - 1) * c * math.log(1 / c)


def Hcond(c, k, S):
    """Entropy upper bound in the condensed regime (three cases of Eq. 4)."""
    kappa = k * (1 + np.exp((-g(c) - g(1 - c)) / c))
    if S < k:
        return math.log(S)
    if S <= kappa:
        return math.log(k)
    return -(1 - c) * math.log((1 - c) / k) - c * math.log(c / (S - k))


# =====================================================================
# 2. Statistics across realizations
# =====================================================================
def j_star(R, delta, alpha=0.05):
    """Smallest j such that the alpha-quantile of Beta(j, R-j+1) is >= 1-delta."""
    for j in range(1, R + 1):
        if beta.ppf(alpha, j, R - j + 1) >= 1 - delta:
            return j
    raise ValueError(f"No j* exists for R={R}, delta={delta}, alpha={alpha}")


# =====================================================================
# 3. Observables on the weights (weights: object array of shape (R, N))
# =====================================================================
def compute_pm(m, weights):
    """m-th largest normalized weight p_m for each realization and index."""
    R, N = weights.shape
    pm = np.zeros((R, N))
    for r in range(R):
        for j in range(N):
            w = weights[r, j]
            if len(w) >= m:
                pm[r, j] = w[m - 1]
    return pm


def compute_Tk(k, weights):
    """Residual mass T_k = 1 - M_k outside the k dominant features."""
    R, N = weights.shape
    Tk = np.zeros((R, N))
    for r in range(R):
        for j in range(N):
            w = weights[r, j]
            if len(w) > k:
                Tk[r, j] = max(0.0, 1.0 - np.sum(w[:k]))
    Tk[Tk < 1e-12] = 0.0  # remove floating-point noise
    return Tk


# =====================================================================
# 4. Solution search
# =====================================================================
def detect_solutions(weights, delta, alpha, kmax=5, n_best=100):
    """Return the n_best parameter sets (k, m, c_disp, c_cond, S) ranked by Gamma."""
    R, N = weights.shape
    jstar = j_star(R, delta, alpha)
    S_i = np.array([[len(w) for w in row] for row in weights])  # diagram sizes
    Smax = int(S_i.max())
    solutions = []

    for k in range(1, kmax + 1):
        Tk = compute_Tk(k, weights)

        # --- Condensation candidates: (c_cond, S, Hcond, coverage) ---
        cond_values = []
        for S in np.unique(S_i):
            # Per-index statistic: j*-th smallest T_k among diagrams with size <= S
            Tstar = np.full(N, np.inf)
            for j in range(N):
                Tk_ok = Tk[S_i[:, j] <= S, j]
                if len(Tk_ok) >= jstar:
                    Tstar[j] = np.sort(Tk_ok)[jstar - 1]

            c_cands = np.unique(np.round(Tstar[np.isfinite(Tstar) & (Tstar > 0)], 3))
            for c_cond in c_cands[(c_cands > 0) & (c_cands <= 1 / (k + 1))]:
                I_plus = Tstar <= c_cond
                if I_plus.any():
                    cond_values.append([c_cond, S, Hcond(c_cond, k, S), I_plus.mean()])

        if not cond_values:
            continue
        cond = np.array(cond_values, dtype=float)

        # --- Dispersion candidates for each m ---
        for m in range(k + 1, Smax + 1):
            # Per-index statistic: j*-th largest p_m across realizations
            pstar = np.sort(compute_pm(m, weights), axis=0)[::-1][jstar - 1, :]
            c_cands = np.unique(np.round(pstar[pstar > 0], 3))

            disp_values = []
            for c_disp in c_cands[(c_cands > 0) & (c_cands <= 1 / m)]:
                I_minus = pstar >= c_disp
                if I_minus.any():
                    disp_values.append([c_disp, Hdisp(c_disp, m), I_minus.mean()])

            if not disp_values:
                continue
            disp = np.array(disp_values, dtype=float)

            # --- Gap and score Gamma for every (c_disp, c_cond) pair ---
            Delta = disp[:, 1][:, None] - cond[:, 2][None, :]
            cov = disp[:, 2][:, None] * cond[:, 3][None, :]
            Gamma = np.where(Delta > 0, Delta / np.log(Smax) * cov, -np.inf)

            for i, j in zip(*np.where(np.isfinite(Gamma))):
                solutions.append({
                    "k": k, "m": m,
                    "c_disp": disp[i, 0], "c_cond": cond[j, 0],
                    "S": int(cond[j, 1]),
                    "Delta": Delta[i, j], "Gamma": Gamma[i, j],
                })

    return sorted(solutions, key=lambda s: s["Gamma"], reverse=True)[:n_best]


# =====================================================================
# 5. Phase classification
# =====================================================================
def distrib_from_solution(weights, sol):
    """Label (-1 dispersed, 0 undefined, +1 condensed) of one realization."""
    k, m, S = sol["k"], sol["m"], sol["S"]
    labels = []
    for w in weights:
        p_m = w[m - 1] if len(w) >= m else 0.0   # m-th weight
        T_k = max(0.0, 1 - w[:k].sum())          # mass outside the k largest
        if p_m >= sol["c_disp"]:
            labels.append(-1)                    # dispersed
        elif T_k <= sol["c_cond"] and len(w) <= S:
            labels.append(1)                     # condensed
        else:
            labels.append(0)                     # transition / undefined
    return np.array(labels)


def probabilities_from_solution(weights, sol):
    """Fraction of dispersed / condensed / undefined realizations at each index."""
    labels = np.array([distrib_from_solution(weights[r], sol)
                       for r in range(weights.shape[0])])
    return (labels == -1).mean(0), (labels == 1).mean(0), (labels == 0).mean(0)


def classify_phases(p_disp, p_cond, R, delta, alpha):
    """Phase at each index: requires at least j* realizations in that phase."""
    j = j_star(R, delta, alpha)
    phase = np.zeros(len(p_disp), dtype=int)
    phase[np.rint(R * p_disp).astype(int) >= j] = -1
    phase[np.rint(R * p_cond).astype(int) >= j] = 1
    return phase


# =====================================================================
# 6. Plot style and helpers
# =====================================================================
STYLE = {
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "lines.linewidth": 1.5, "pdf.fonttype": 42, "ps.fonttype": 42,
}
COLORS = {-1: "forestgreen", 0: "lightgray", 1: "firebrick"}
MAX_TICKS = 6


def _frac_labels(symbol):
    """Legend labels F_disp, F_cond, F_ind for the given parameter symbol."""
    return {name: rf"$F_{{\mathrm{{{name}}}}}({symbol})$" for name in ("disp", "cond", "ind")}


def _set_xticks(ax, values):
    """At most MAX_TICKS ticks, labelled with the parameter values."""
    idx = np.unique(np.linspace(0, len(values) - 1, min(MAX_TICKS, len(values)), dtype=int))
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{values[i]:g}" for i in idx])
    ax.set_xlim(-0.5, len(values) - 0.5)


def _finish(fig, filename):
    """Save (format inferred from the extension) and show the figure."""
    if filename is not None:
        fig.savefig(filename, bbox_inches="tight")
    plt.show()


# =====================================================================
# 7. Figure 1: robustness across the best solutions
# =====================================================================
def plot_solution_probabilities(all_weights_test, best_solutions, delta, alpha,
                                param_values, param_symbol="K", filename=None):
    R = all_weights_test.shape[0]
    P = np.asarray(param_values)
    x = np.arange(len(P))

    # Phase assigned by each solution at each index
    phases = np.array([
        classify_phases(*probabilities_from_solution(all_weights_test, s)[:2], R, delta, alpha)
        for s in best_solutions
    ])
    # Fraction of solutions assigning each phase
    p_disp, p_ind, p_cond = (phases == -1).mean(0), (phases == 0).mean(0), (phases == 1).mean(0)

    lab = _frac_labels(param_symbol)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.2, 2.2))
        ax.scatter(x, p_disp, color=COLORS[-1], s=25, zorder=3, label=lab["disp"])
        ax.scatter(x, p_cond, color=COLORS[1], s=25, zorder=3, label=lab["cond"])
        ax.scatter(x, p_ind, color=COLORS[0], s=25, zorder=2, label=lab["ind"])

        _set_xticks(ax, P)
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel("Fraction")
        ax.set_xlabel(rf"${param_symbol}$")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.35), ncol=3,
                  frameon=False, markerscale=1.3, columnspacing=1.2, handletextpad=0.4)
        fig.subplots_adjust(left=0.16, right=0.98, bottom=0.30, top=0.78)
        _finish(fig, filename)

    return p_disp, p_cond, p_ind


# =====================================================================
# 8. Figure 2: phase detection with the optimal solution
# =====================================================================
def plot_phase_detection(all_weights_test, solution, delta, alpha, entropies,
                         param_values, param_symbol="K",
                         delta_values=(0.35, 0.40, 0.45),
                         ylabel=r"PE($H_0$)", filename=None):
    R = all_weights_test.shape[0]
    P = np.asarray(param_values)
    x = np.arange(len(P))

    p_disp, p_cond, p_ind = probabilities_from_solution(all_weights_test, solution)
    phase = classify_phases(p_disp, p_cond, R, delta, alpha)
    # One row per delta value, to show sensitivity of the classification
    phase_matrix = np.array([classify_phases(p_disp, p_cond, R, d, alpha) for d in delta_values])

    mean_pe, std_pe = entropies.mean(0), entropies.std(0)
    lab = _frac_labels(param_symbol)
    cmap = ListedColormap([COLORS[-1], COLORS[0], COLORS[1]])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

    with plt.rc_context(STYLE):
        fig, (ax_f, ax_e, ax_p) = plt.subplots(
            3, 1, figsize=(6.2, 5.2), sharex=True,
            gridspec_kw={"height_ratios": [1.2, 2.5, 0.8], "hspace": 0.12})

        # Top panel: fractions of realizations in each phase
        ax_f.plot(x, p_disp, color=COLORS[-1])
        ax_f.plot(x, p_cond, color=COLORS[1])
        ax_f.plot(x, p_ind, color="gray", linestyle="--")
        ax_f.set_ylabel("Fraction")
        ax_f.set_ylim(-0.03, 1.03)
        ax_f.grid(alpha=0.25)

        # Middle panel: persistent entropy with phases assigned
        for i in np.where(phase != 0)[0]:
            ax_e.axvspan(i - 0.5, i + 0.5, color=COLORS[phase[i]], alpha=0.40, linewidth=0)
        ax_e.plot(x, mean_pe, color="black")
        ax_e.fill_between(x, mean_pe - std_pe, mean_pe + std_pe, color="gray", alpha=0.25)
        ax_e.set_ylabel(ylabel)
        ax_e.grid(alpha=0.25)

        # Bottom panel: sensitivity to delta
        ax_p.imshow(phase_matrix, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm,
                    extent=[-0.5, len(P) - 0.5, len(delta_values) - 0.5, -0.5])
        ax_p.set_yticks(np.arange(len(delta_values)))
        ax_p.set_yticklabels([rf"$\delta={d:.2f}$" for d in delta_values])
        ax_p.set_xlabel(rf"${param_symbol}$")
        _set_xticks(ax_p, P)

        # Single legend above the first panel
        handles = [
            Line2D([0], [0], color=COLORS[-1], label=lab["disp"]),
            Line2D([0], [0], color=COLORS[1], label=lab["cond"]),
            Line2D([0], [0], color="gray", linestyle="--", label=lab["ind"]),
            Line2D([0], [0], color="black", label="Mean PE"),
            Patch(facecolor="gray", alpha=0.25, label="PE standard deviation"),
            Patch(facecolor=COLORS[-1], label="Dispersed"),
            Patch(facecolor=COLORS[0], label="Undefined"),
            Patch(facecolor=COLORS[1], label="Condensed"),
        ]
        ax_f.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.05),
                    ncol=4, frameon=False, columnspacing=1.2, handletextpad=0.4)
        fig.subplots_adjust(left=0.16, right=0.98, bottom=0.10, top=0.78, hspace=0.12)
        _finish(fig, filename)