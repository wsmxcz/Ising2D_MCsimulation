import numpy as np
from numba import njit, prange

@njit
def compute_total_energy(spins, neighbors_idx, neighbors_J, H):
    """
    Compute the total energy of the Ising system:
      E = -sum(J * s_i * s_j) - H * sum(s_i)
    Using 2D periodic boundary conditions.
    """
    N = spins.shape[0]
    E = 0.0
    for i in range(N):
        for k in range(neighbors_idx.shape[1]):
            j = neighbors_idx[i, k]
            E -= neighbors_J[i, k] * spins[i] * spins[j]
    # Each interaction counted twice above
    E *= 0.5
    # External field contribution
    E -= H * spins.sum()
    return E

@njit
def compute_delta_energy(spins, i, neighbors_idx, neighbors_J, H):
    """
    Compute the energy difference dE if spin 'i' is flipped (s_i -> -s_i).
    """
    dE = 0.0
    for k in range(neighbors_idx.shape[1]):
        j = neighbors_idx[i, k]
        dE += neighbors_J[i, k] * spins[i] * spins[j]
    # Flipping s_i adds 2 * s_i * (neighbors) and 2 * H * s_i
    dE = 2.0 * dE + 2.0 * H * spins[i]
    return dE

@njit
def update_metropolis(spins, neighbors_idx, neighbors_J, H, beta, N):
    """
    Sequential (per-spin) Metropolis update for the entire lattice.
    """
    for i in range(N):
        dE = compute_delta_energy(spins, i, neighbors_idx, neighbors_J, H)
        # Metropolis acceptance rule
        if dE < 0.0 or np.random.rand() < np.exp(-beta * dE):
            spins[i] *= -1
    return spins

@njit(parallel=True)
def update_metropolis_numba(spins, neighbors_idx, neighbors_J, H, beta, N):
    """
    Parallel Metropolis update using Numba's 'prange'. 
    All spins get updated "in parallel" (using a copy array).
    """
    new_spins = spins.copy()
    for i in prange(N):
        dE = compute_delta_energy(spins, i, neighbors_idx, neighbors_J, H)
        if dE < 0.0 or np.random.rand() < np.exp(-beta * dE):
            new_spins[i] *= -1
    return new_spins

class Ising2D:
    """
    2D Ising Model with pre-cached neighbor indices (periodic boundaries)
    and optional Numba-based parallel updates.
    """

    def __init__(self, L, J=1.0, H=0.0, init_random=True, use_parallel=False):
        self.L = L
        self.N = L * L
        self.J = J
        self.H = H
        self.use_parallel = use_parallel
        
        # Initialize spins randomly or all up
        if init_random:
            self.spins = np.random.choice(np.array([-1, 1]), size=self.N)
        else:
            self.spins = np.ones(self.N, dtype=np.int64)
        
        # Precompute neighbors
        self.neighbors_idx, self.neighbors_J = self._build_lattice()
    
    def _build_lattice(self):
        """
        Construct neighbor list for each site (4 neighbors in 2D).
        """
        neighbors_idx = np.empty((self.N, 4), dtype=np.int64)
        neighbors_J = np.empty((self.N, 4), dtype=np.float64)

        for r in range(self.L):
            for c in range(self.L):
                i = r * self.L + c
                # Periodic boundary conditions
                neighbors_idx[i, 0] = r * self.L + ((c + 1) % self.L)   # Right
                neighbors_idx[i, 1] = r * self.L + ((c - 1) % self.L)   # Left
                neighbors_idx[i, 2] = ((r - 1) % self.L) * self.L + c   # Up
                neighbors_idx[i, 3] = ((r + 1) % self.L) * self.L + c   # Down

                neighbors_J[i, :] = self.J
        
        return neighbors_idx, neighbors_J

    def total_energy(self):
        """Return the total energy of the current spin configuration."""
        return compute_total_energy(self.spins, self.neighbors_idx, self.neighbors_J, self.H)

    def delta_energy_if_flip(self, i):
        """Compute ΔE if spin i is flipped."""
        return compute_delta_energy(self.spins, i, self.neighbors_idx, self.neighbors_J, self.H)
    
    def flip_spin(self, i):
        """Flip spin i (s_i -> -s_i)."""
        self.spins[i] *= -1

    def update_spins(self, beta):
        """
        Update spins using either the sequential or parallel Numba-based Metropolis routine.
        """
        if self.use_parallel:
            self.spins = update_metropolis_numba(
                self.spins, self.neighbors_idx, self.neighbors_J, self.H, beta, self.N
            )
        else:
            self.spins = update_metropolis(
                self.spins, self.neighbors_idx, self.neighbors_J, self.H, beta, self.N
            )

    def set_parallel_mode(self, use_parallel):
        """Enable or disable parallel updates."""
        self.use_parallel = use_parallel