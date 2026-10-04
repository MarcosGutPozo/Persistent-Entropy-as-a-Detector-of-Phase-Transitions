import numpy as np

from src.phase_transitions.general.persistent_entropy import pe_from_distance_matrix


class Kuramoto():
    
    def __init__(self, N, t0, t1, steps, K_values, R_k, K_selected):
        # Simulation parameters
        self.N = N
        self.t0 = t0
        self.t1 = t1
        self.steps = steps
        self.K_values = np.array(K_values, dtype=float)
        self.R_k = R_k
        self.dt = (self.t1 - self.t0) / self.steps
        self.K_selected = K_selected

    def random_adjacency(self, p=0.762):
        # Generate a random symmetric adjacency matrix
        A = (np.random.rand(self.N, self.N) < p).astype(np.float32)
        A = np.triu(A, 1)
        A = A + A.T
        # Remove self-connections
        np.fill_diagonal(A, 0.0)
        return A

    def sample_kuramoto_initial(self):
        # Sample natural frequencies
        omega0 = np.random.normal(0.0, 1.0, size=self.N).astype(np.float32)
        # Sample initial phases
        theta0 = np.random.uniform(0.0, 2*np.pi, size=self.N).astype(np.float32)
        return omega0, theta0

    def kuramoto_step(self, theta, omega, K, G):
        # Compute the Kuramoto phase derivative
        return omega + (K / theta.size) * np.sum(G * np.sin(theta[None, :] - theta[:, None]), axis=1)

    def simulate_kuramoto(self, G, omega0, theta0, K):
        N = theta0.size
        theta = theta0.copy()
        # Store phase and synchronization time series
        theta_ts = np.zeros((self.steps, self.N), dtype=np.float32)
        r_ts = np.zeros(self.steps, dtype=np.float32)
        for t in range(self.steps):
           theta_ts[t] = theta
           # Compute Kuramoto order parameter
           c = np.sum(np.cos(theta))
           s = np.sum(np.sin(theta))
           r_ts[t] = np.sqrt(c*c + s*s) / self.N  # r in [0,1]
           theta = theta + self.dt * self.kuramoto_step(theta, omega0, K, G)
        return theta_ts, r_ts

    def synchrony_distance_from_theta(self, theta):
        phi = np.abs(np.cos(theta[:, None] - theta[None, :]))
        D = 1.0 - phi
        np.fill_diagonal(D, 0.0)
        return D

    def kuramoto_pe_h0_series(self, theta_ts, normalize=False, min_persistence=None):
        # Compute persistent entropy and weights for every time step
        idxs = list(range(0, theta_ts.shape[0]))
        pe0 = np.zeros(len(idxs), dtype=np.float32)
        weight0 = []
        for k, t in enumerate(idxs):
           # Build the synchrony distance matrix
           D = self.synchrony_distance_from_theta(theta_ts[t])
           # Compute persistent entropy and weights from the distance matrix
           _, weights, pe = pe_from_distance_matrix(D, dim=0, normalize=normalize, min_persistence=min_persistence)
           pe0[k] = pe
           weight0.append(weights)
    
        return pe0, weight0

    