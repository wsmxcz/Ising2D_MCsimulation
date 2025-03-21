# analysis.py - Data analysis tools for the Ising model simulation

import numpy as np
import csv
import glob

def compute_averages(energies, magnetizations):
    """Compute thermodynamic averages from energy and magnetization samples"""
    # Convert to numpy arrays
    energies = np.array(energies)
    magnetizations = np.array(magnetizations)
    abs_magnetizations = np.abs(magnetizations)
    
    # Average energy and magnetization
    avg_energy = np.mean(energies)
    avg_magnetization = np.mean(magnetizations)
    avg_abs_magnetization = np.mean(abs_magnetizations)
    
    # Specific heat: C = (E^2 - <E>^2) / (N * T^2)
    energy_sq = np.mean(energies**2)
    specific_heat = (energy_sq - avg_energy**2) / (len(energies) * (1 / len(energies))**2)
    
    # Susceptibility using absolute magnetization
    susceptibility = (np.mean(abs_magnetizations**2) - avg_abs_magnetization**2) / len(magnetizations)
    
    return {
        'avg_energy': avg_energy,
        'specific_heat': specific_heat,
        'magnetization': avg_magnetization,
        'abs_magnetization': avg_abs_magnetization,
        'susceptibility': susceptibility
    }

def store_results_to_csv(temperatures, all_energies, all_magnetizations, L_values, output_filename="ising_results.csv"):
    """Store simulation results to CSV file"""
    rows = []
    header = ["L", "T", "avg_energy", "std_energy", "specific_heat", 
             "avg_magnetization", "avg_abs_magnetization", "std_magnetization", "susceptibility"]
    
    # Round temperatures to avoid floating point precision issues
    rounded_temperatures = np.round(temperatures, 4)
    
    # Calculate observables for each system size and temperature
    for L_idx, L in enumerate(L_values):
        energies_for_L = all_energies[L_idx]
        mags_for_L = all_magnetizations[L_idx]
        N = L * L
        
        for i, T in enumerate(rounded_temperatures):
            e_samples = np.array(energies_for_L[i])
            m_samples = np.array(mags_for_L[i])
            abs_m_samples = np.abs(m_samples)
            
            avg_e = np.mean(e_samples)
            std_e = np.std(e_samples)
            avg_m = np.mean(m_samples)
            avg_abs_m = np.mean(abs_m_samples)
            std_m = np.std(m_samples)
            
            specific_heat = (np.mean(e_samples**2) - avg_e**2) / (N * T**2)
            susceptibility = (np.mean(abs_m_samples**2) - avg_abs_m**2) / (T * N)
            
            row = [L, T, avg_e, std_e, specific_heat, avg_m, avg_abs_m, std_m, susceptibility]
            rows.append(row)
    
    # Write to CSV file
    with open(output_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    
    print(f"Results stored in {output_filename}")

def combine_csv_files(pattern="ising_results_L*.csv", output_file="ising_results_combined.csv"):
    """Combine multiple per-system-size CSV files into a single file"""
    # Get all matching files
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files matching pattern '{pattern}' found.")
        return
    
    all_rows = []
    header = None
    
    # Read each file
    for file in files:
        with open(file, 'r', newline='') as f:
            reader = csv.reader(f)
            file_header = next(reader)
            
            if header is None:
                header = file_header
            
            # Add all rows, ensuring temperature is rounded consistently
            for row in reader:
                if len(row) > 1:
                    try:
                        row[1] = str(round(float(row[1]), 4))
                    except (ValueError, IndexError):
                        pass
                all_rows.append(row)
    
    # Write combined file
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_rows)
    
    print(f"Combined {len(files)} files into {output_file}")
