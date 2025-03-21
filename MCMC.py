# MCMC.py - Monte Carlo simulation methods for the Ising model

import numpy as np
import math

class MCMC:
    """Monte Carlo simulation for the 2D Ising model using Metropolis algorithm"""
    def __init__(self, model, sweeps, temperature, method='metropolis', simultaneous_flip=False):
        """
        Args:
            model: Ising model instance
            sweeps: number of Monte Carlo sweeps
            temperature: system temperature
            method: 'metropolis' or 'heat_bath' (not implented now)
            simultaneous_flip: whether to use parallel updates
        """
        self.model = model
        self.sweeps = sweeps
        self.beta = 1.0 / temperature
        self.method = method
        self.simultaneous_flip = simultaneous_flip

    def step(self):
        """Perform one Monte Carlo step"""
        if self.simultaneous_flip:
            self.simultaneous_spin_flip()
        else:
            self.single_spin_flip()

    def single_spin_flip(self):
        """Update spins one at a time using Metropolis algorithm"""
        N = self.model.N
        rand_indices = np.random.randint(0, N, size=N)
        
        for idx in rand_indices:
            dE = self.model.delta_energy_if_flip(idx)
            # Metropolis acceptance criterion
            if dE < 0 or np.random.rand() < math.exp(-self.beta * dE):
                self.model.flip_spin(idx)

    def simultaneous_spin_flip(self):
        """Use model's parallel update method"""
        self.model.update_spins(self.beta)

    def run(self):
        """Run the simulation for the specified number of sweeps"""
        for _ in range(self.sweeps):
            self.step()
            
    def set_parallel_mode(self, use_parallel):
        """Set whether to use parallel spin updates in the model"""
        if hasattr(self.model, 'set_parallel_mode'):
            self.model.set_parallel_mode(use_parallel)
        