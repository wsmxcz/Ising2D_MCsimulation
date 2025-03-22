import numpy as np
import math

class MCMC:
    """
    Monte Carlo simulation for the 2D Ising model using Metropolis algorithm.
    """

    def __init__(self, model, sweeps, temperature, method='metropolis_numba'):
        """
        Args:
            model: Ising model instance (e.g., Ising2D)
            sweeps: number of Monte Carlo sweeps
            temperature: system temperature
            method: either 'metropolis' (single-spin updates done here)
                    or 'metropolis_numba' (use model's Numba-based updates)
        """
        self.model = model
        self.sweeps = sweeps
        self.beta = 1.0 / temperature
        self.method = method

    def step(self):
        """
        Perform one Monte Carlo step, depending on the chosen method.
        """
        if self.method == 'metropolis_numba':
            self.model.update_spins(self.beta)
        else:
            self.single_spin_flip()

    def single_spin_flip(self):
        """
        Update spins one by one in random order, using the Metropolis acceptance rule.
        """
        N = self.model.N
        rand_indices = np.random.randint(0, N, size=N)
        
        for idx in rand_indices:
            dE = self.model.delta_energy_if_flip(idx)
            # Metropolis acceptance criterion
            if dE < 0.0 or np.random.rand() < math.exp(-self.beta * dE):
                self.model.flip_spin(idx)

    def run(self):
        """
        Run the simulation for the specified number of sweeps.
        """
        for _ in range(self.sweeps):
            self.step()