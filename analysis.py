# analysis.py - Data analysis tools for the Ising model simulation

import numpy as np
import csv
import os

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

def compute_observables(energy_samples, magnetization_samples, temperature, system_size):
    """Compute thermodynamic observables from simulation data
    
    Args:
        energy_samples: List of energy measurements
        magnetization_samples: List of magnetization measurements
        temperature: Simulation temperature
        system_size: Linear system size L
    
    Returns:
        Dictionary containing calculated observables
    """
    # Convert to numpy arrays
    energy_samples = np.array(energy_samples)
    magnetization_samples = np.array(magnetization_samples)
    abs_magnetization_samples = np.abs(magnetization_samples)
    N = system_size * system_size
    
    # Calculate basic observables
    avg_energy = np.mean(energy_samples) / N
    avg_magnetization = np.mean(magnetization_samples)
    avg_abs_magnetization = np.mean(abs_magnetization_samples)
    
    # Fluctuations for specific heat and susceptibility
    energy_squared_avg = np.mean(energy_samples**2)
    abs_mag_squared_avg = np.mean(abs_magnetization_samples**2)
    
    # Specific heat: C = (⟨E²⟩ - ⟨E⟩²) / (N * T²)
    specific_heat = (energy_squared_avg - np.mean(energy_samples)**2) / (N * temperature**2)
    
    # Susceptibility: χ = (⟨|M|²⟩ - ⟨|M|⟩²) / (N * T)
    susceptibility = (abs_mag_squared_avg - avg_abs_magnetization**2) / (temperature * N)
    
    return {
        'avg_energy': avg_energy,
        'std_energy': np.std(energy_samples) / N,
        'specific_heat': specific_heat,
        'avg_magnetization': avg_magnetization,
        'avg_abs_magnetization': avg_abs_magnetization,
        'std_magnetization': np.std(magnetization_samples),
        'susceptibility': susceptibility
    }

def store_results_to_csv(temperatures, energy_data, magnetization_data, system_sizes, output_filename="data/ising_results.csv"):
    """Store simulation results to CSV file
    
    Args:
        temperatures: List of temperatures simulated
        energy_data: Nested list of energy samples for each temperature and system size
        magnetization_data: Nested list of magnetization samples for each temperature and system size
        system_sizes: List of system sizes L
        output_filename: Path to output CSV file
    """
    # Ensure data directory exists
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    
    rows = []
    header = ["L", "T", "avg_energy", "std_energy", "specific_heat", 
             "avg_magnetization", "avg_abs_magnetization", "std_magnetization", "susceptibility"]
    
    # Round temperatures to avoid floating point precision issues
    rounded_temperatures = np.round(temperatures, 4)
    
    # Calculate observables for each system size and temperature
    for size_idx, L in enumerate(system_sizes):
        energies_for_size = energy_data[size_idx]
        mags_for_size = magnetization_data[size_idx]
        
        for temp_idx, T in enumerate(rounded_temperatures):
            # Get samples for this temperature and system size
            energy_samples = energies_for_size[temp_idx]
            mag_samples = mags_for_size[temp_idx]
            
            # Calculate all observables
            results = compute_observables(energy_samples, mag_samples, T, L)
            
            # Create data row
            row = [
                L, 
                T, 
                results['avg_energy'], 
                results['std_energy'],
                results['specific_heat'], 
                results['avg_magnetization'], 
                results['avg_abs_magnetization'], 
                results['std_magnetization'], 
                results['susceptibility']
            ]
            rows.append(row)
    
    # Write to CSV file
    with open(output_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    
    print(f"Results stored in {output_filename}")
