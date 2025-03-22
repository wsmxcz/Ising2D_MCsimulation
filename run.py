import numpy as np
from tqdm import tqdm
import concurrent.futures
import os
import platform
from Ising2D import Ising2D
from MCMC import MCMC
from analysis import store_results_to_csv
from plot import combine_and_plot_results

def simulate_single_temperature(params):
    """Run Ising model simulation for a single temperature"""
    L = params['L']
    T = params['T']
    warmup_sweeps = params['warmup_sweeps']
    measurement_sweeps = params['measurement_sweeps'] 
    sample_interval = params['sample_interval']
    
    # Use ordered initialization for low temperatures
    init_random = T >= 1.0
    
    # Initialize model and simulation
    model = Ising2D(L=L, J=1.0, H=0.0, init_random=init_random, use_parallel=False)
    mcmc = MCMC(model, sweeps=warmup_sweeps, temperature=T, method='metropolis_numba')
    
    for step in range(warmup_sweeps):
        mcmc.step()
        prev_energy = model.total_energy()
    
    # Measurement phase
    energies_at_T = []
    mags_at_T = []
    
    for sweep in range(measurement_sweeps):
        mcmc.step()
        
        if sweep % sample_interval == 0:
            E = model.total_energy()
            M = np.sum(model.spins) / model.N
            energies_at_T.append(E)
            mags_at_T.append(M)
    
    return (T, energies_at_T, mags_at_T)

def run_simulation(max_workers=None, use_process_pool=True):
    """Run parallel Ising model simulations for multiple system sizes and temperatures"""
    
    # System parameters
    L_values = [10, 20, 30]
    T_min, T_max = 0.015, 4.5
    num_temps = 300
    temperatures = np.round(np.linspace(T_min, T_max, num_temps), 4)
    
    # Simulation parameters
    warmup_sweeps = 100_000
    measurement_sweeps = 300_000
    sample_interval = 10
    
    # Process each system size
    for L in L_values:
        print(f"Processing system size L={L}")
        
        params_list = []
        for T in temperatures:
            params = {
                'L': L,
                'T': float(T),
                'warmup_sweeps': warmup_sweeps,
                'measurement_sweeps': measurement_sweeps,
                'sample_interval': sample_interval,
            }
            params_list.append(params)
        
        # Execute simulations in parallel
        results = []
        executor_class = concurrent.futures.ProcessPoolExecutor if use_process_pool else concurrent.futures.ThreadPoolExecutor
        
        with executor_class(max_workers=max_workers) as executor:
            futures = [executor.submit(simulate_single_temperature, params) for params in params_list]
            
            for future in tqdm(concurrent.futures.as_completed(futures), 
                               total=len(futures),
                               desc=f"Simulating L={L}"):
                results.append(future.result())
        
        # Process results
        results.sort(key=lambda x: x[0])
        energies_for_L = []
        mags_for_L = []
        
        for T, energies_at_T, mags_at_T in results:
            energies_for_L.append(energies_at_T)
            mags_for_L.append(mags_at_T)
        
        csv_filename = f"ising_results_L{L}.csv"
        store_results_to_csv(temperatures, [energies_for_L], [mags_for_L], [L], output_filename=csv_filename)
        print(f"Completed system size L={L}, results saved to {csv_filename}")
    
    print("All system sizes completed. Generating plots...")
    combine_and_plot_results()

if __name__ == "__main__":
    if platform.system() == 'Windows':
        import multiprocessing
        multiprocessing.set_start_method('spawn', force=True)
    
    cpu_count = os.cpu_count()
    recommended_workers = max(1, cpu_count - 2) if cpu_count else None
    
    print(f"System has {cpu_count} CPU cores, using {recommended_workers} worker processes")
    run_simulation(max_workers=recommended_workers, use_process_pool=True)