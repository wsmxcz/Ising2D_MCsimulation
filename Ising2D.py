# Ising2D.py - 2D Ising model implementation with Numba acceleration

import numpy as np
from numba import njit, prange

@njit
def compute_total_energy(spins, neighbors_idx, neighbors_J, H):
    """Compute total energy of the Ising system"""
    N = spins.shape[0]
    E = 0.0
    for i in range(N):
        for k in range(neighbors_idx.shape[1]):
            j = neighbors_idx[i, k]
            E -= neighbors_J[i, k] * spins[i] * spins[j]
    E *= 0.5  # Correct for double-counting
    E -= H * spins.sum()
    return E

@njit
def compute_delta_energy(spins, i, neighbors_idx, neighbors_J, H):
    """Compute energy change when flipping spin i"""
    dE = 0.0
    for k in range(neighbors_idx.shape[1]):
        j = neighbors_idx[i, k]
        dE += neighbors_J[i, k] * spins[i] * spins[j]
    dE = 2.0 * dE + 2.0 * H * spins[i]
    return dE

@njit
def update_spins_sequential(spins, neighbors_idx, neighbors_J, H, beta, N):
    """Update spins sequentially using Metropolis algorithm"""
    for i in range(N):
        dE = compute_delta_energy(spins, i, neighbors_idx, neighbors_J, H)
        if dE < 0 or np.random.rand() < np.exp(-beta * dE):
            spins[i] *= -1
    return spins

@njit(parallel=True)
def update_spins_in_parallel(spins, neighbors_idx, neighbors_J, H, beta, N):
    """Update spins in parallel using Numba's prange"""
    # Create a copy of spins to avoid race conditions
    new_spins = spins.copy()
    
    # Use prange for parallel execution
    for i in prange(N):
        dE = compute_delta_energy(spins, i, neighbors_idx, neighbors_J, H)
        if dE < 0 or np.random.rand() < np.exp(-beta * dE):
            new_spins[i] *= -1
    
    return new_spins

class Ising2D:
    """2D Ising Model with pre-cached neighbor indices for performance"""
    def __init__(self, L, J=1.0, H=0.0, init_random=True, use_parallel=False):
        self.L = L
        self.N = L * L
        self.J = J
        self.H = H
        self.use_parallel = use_parallel
        
        # Initialize spins (random or ordered)
        if init_random:
            self.spins = np.random.choice(np.array([-1, 1]), size=self.N)
        else:
            self.spins = np.ones(self.N, dtype=np.int64)
        
        # Precompute the neighbors and couplings for all spins
        self.neighbors_idx, self.neighbors_J = self._build_lattice()
    
    def _build_lattice(self):
        """Build neighbor indices and couplings for a 2D lattice with periodic boundaries"""
        neighbors_idx = np.empty((self.N, 4), dtype=np.int64)
        neighbors_J = np.empty((self.N, 4), dtype=np.float64)
        L = self.L
        for r in range(L):
            for c in range(L):
                i = r * L + c
                neighbors_idx[i, 0] = r * L + ((c + 1) % L)  # Right
                neighbors_idx[i, 1] = r * L + ((c - 1) % L)  # Left
                neighbors_idx[i, 2] = ((r - 1) % L) * L + c  # Up
                neighbors_idx[i, 3] = ((r + 1) % L) * L + c  # Down
                
                neighbors_J[i, :] = self.J
        return neighbors_idx, neighbors_J

    def total_energy(self):
        """Compute the total energy of the system"""
        return compute_total_energy(self.spins, self.neighbors_idx, self.neighbors_J, self.H)

    def delta_energy_if_flip(self, i):
        """Compute the energy change if spin i is flipped"""
        return compute_delta_energy(self.spins, i, self.neighbors_idx, self.neighbors_J, self.H)
    
    def flip_spin(self, i):
        """Flip spin i"""
        self.spins[i] *= -1

    def update_spins(self, beta):
        """Update all spins using either parallel or sequential algorithm"""
        if self.use_parallel:
            self.spins = update_spins_in_parallel(self.spins, self.neighbors_idx, self.neighbors_J, self.H, beta, self.N)
        else:
            self.spins = update_spins_sequential(self.spins, self.neighbors_idx, self.neighbors_J, self.H, beta, self.N)
            
    def set_parallel_mode(self, use_parallel):
        """Set whether to use parallel updates"""
        self.use_parallel = use_parallel