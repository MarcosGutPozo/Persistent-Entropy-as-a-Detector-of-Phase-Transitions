import math
import numpy as np

from src.phase_transitions.general.persistent_entropy import pe_from_distance_matrix


class Vicsek():

    def __init__(self, L, v0, steps, dt_v, num_particles, eta_values, R_eta, r_int, eta_selected):
        # Simulation parameters
        self.L = L
        self.v0 = v0
        self.steps = steps
        self.dt_v = dt_v
        self.num_particles = num_particles
        self.eta_values = np.array(eta_values, dtype=float)
        self.R_eta = R_eta
        self.r_int = r_int
        self.eta_selected = eta_selected

    def angle_mean(self, angles):
        # Compute the circular mean of angles
        return math.atan2(np.sin(angles).mean(), np.cos(angles).mean())

    def pairwise_periodic_displacements(self, pos):
        # Compute pairwise distances with periodic boundaries
        dx = pos[:, None, 0] - pos[None, :, 0]
        dy = pos[:, None, 1] - pos[None, :, 1]
        dx -= self.L * np.round(dx / self.L)
        dy -= self.L * np.round(dy / self.L)
        return dx, dy

    def vicsek_step(self, pos, theta, eta):
        # Compute pairwise periodic displacements
        dx, dy = self.pairwise_periodic_displacements(pos)
        # Compute squared distances
        dist2 = dx*dx + dy*dy
        # Identify particles within interaction radius
        neigh = dist2 <= (self.r_int*self.r_int)
        # Compute average direction of neighbors
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        sum_sin = neigh @ sin_t
        sum_cos = neigh @ cos_t
        theta_avg = np.arctan2(sum_sin, sum_cos)
        # Add angular noise
        noise = (np.random.rand(theta.size) - 0.5) * eta
        theta_new = theta_avg + noise
         # Compute particle velocities
        vel = np.stack([np.cos(theta_new), np.sin(theta_new)], axis=1) * self.v0
        # Update positions
        pos_new = pos + self.dt_v * vel
        # Apply periodic boundary conditions
        pos_new = pos_new % self.L
        return pos_new.astype(np.float32), theta_new.astype(np.float32)

    def simulate_vicsek(self, eta):
        # Random initial positions and orientations
        pos = np.random.rand(self.num_particles, 2).astype(np.float32) * self.L
        theta = (np.random.rand(self.num_particles).astype(np.float32) * 2*np.pi)
        # Store simulation time series
        pos_ts = np.zeros((self.steps, self.num_particles, 2), dtype=np.float32)
        theta_ts = np.zeros((self.steps, self.num_particles), dtype=np.float32)
        psi_ts = np.zeros(self.steps, dtype=np.float32)

        for t in range(self.steps):
            pos_ts[t] = pos
            theta_ts[t] = theta
            c = np.sum(np.cos(theta))
            s = np.sum(np.sin(theta))
            psi_ts[t] = np.sqrt(c*c + s*s) / self.num_particles

            pos, theta = self.vicsek_step(pos, theta, eta)

        return pos_ts, theta_ts, psi_ts

    def vicsek_velocity(self, theta):
        # Compute velocity vectors from orientations
        return np.stack([np.cos(theta), np.sin(theta)], axis=1) * self.v0

    def distance_velocity(self, theta):
        # Compute pairwise distances between velocity vectors
        V = self.vicsek_velocity(theta)
        diff = V[:, None, :] - V[None, :, :]
        D = np.sqrt(np.sum(diff*diff, axis=2))
        np.fill_diagonal(D, 0.0)
        return D

    def vicsek_pe_h0_series(self, theta_ts, normalize=False, min_persistence=None):
        idxs = list(range(0, theta_ts.shape[0]))
        pe0 = np.zeros(len(idxs), dtype=np.float32)
        weight0 = []
        for k, t in enumerate(idxs):
            D = self.distance_velocity(theta_ts[t])
            dgms, weights, pe  = pe_from_distance_matrix(D, dim=0, normalize=normalize, min_persistence=min_persistence)
            pe0[k]  = pe
            weight0.append(weights)
        return pe0, weight0