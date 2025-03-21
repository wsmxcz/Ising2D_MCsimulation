# MCMC.py - Monte Carlo simulation methods for the Ising model

import numpy as np
import math

class MCMC:
    """Monte Carlo simulation for the 2D Ising model using Metropolis algorithm"""
    def __init__(self, model, sweeps, temperature, method='metropolis'):
        """
        Initialize Monte Carlo simulation
        
        Args:
            model: Ising model instance
            sweeps: number of Monte Carlo sweeps
            temperature: system temperature
            method: 'metropolis' (currently only option supported)
        """
        self.model = model
        self.sweeps = sweeps
        self.beta = 1.0 / temperature
        self.method = method

    def step(self):
        """Perform one Monte Carlo step (single sweep through the lattice)"""
        self.single_spin_flip()

    def single_spin_flip(self):
        """Update spins one at a time using Metropolis algorithm"""
        N = self.model.N
        
        # Randomly select N sites to update (one sweep)
        rand_indices = np.random.randint(0, N, size=N)
        
        for idx in rand_indices:
            # Calculate energy change if we flip this spin
            dE = self.model.delta_energy_if_flip(idx)
            
            # Apply Metropolis acceptance criterion
            if dE < 0 or np.random.rand() < math.exp(-self.beta * dE):
                self.model.flip_spin(idx)

    def run(self):
        """Run the simulation for the specified number of sweeps"""
        for _ in range(self.sweeps):
            self.step()
            
    def set_parallel_mode(self, use_parallel):
        """Set whether to use parallel spin updates in the model"""
        if hasattr(self.model, 'set_parallel_mode'):
            self.model.set_parallel_mode(use_parallel)
        