# 2D Ising Model Simulation

A high-performance implementation of the 2D Ising model simulation using Monte Carlo methods with parallel temperature processing.

!!!Note!!!: This is a course assignment code completed within a limited time frame. Parts of the code (primarily the plotting sections and comments) utilized generative AI and have not undergone rigorous testing. If you discover any bugs, please feel free to contact me at: wsmxcz@gmail.com.

## Overview

This project implements a parallelized 2D Ising model simulation. The code is optimized to run multiple temperature simulations in parallel while ensuring accurate equilibration.

Key features:
- Parallel temperature processing using Python's concurrent.futures
- Numba-accelerated core simulation functions
- Adaptive equilibration for different temperature regimes
- Finite-size scaling analysis for critical temperature estimation

## Requirements

- Python 3.7+
- NumPy
- Numba
- Matplotlib
- pandas
- tqdm
- SciPy

Install dependencies:
```bash
pip install numpy numba matplotlib pandas tqdm scipy
```

## Project Structure

- `Ising2D.py`: Core Ising model implementation with Numba optimization
- `MCMC.py`: Monte Carlo simulation methods (Metropolis algorithm)
- `run.py`: Main simulation runner with parallel temperature processing
- `analysis.py`: Data analysis tools for computing observables and storing results
- `plot.py`: Visualization tools for generating publication-quality figures

## Usage

Run the simulation with default parameters:
```bash
python run.py
```

The simulation will:
1. Run the 2D Ising model for system sizes L = 10, 20, and 30
2. Process temperatures from 0.015 to 4.5
3. Store results for each system size in CSV files
4. Combine results and generate analysis plots

## Physics Background

The 2D Ising model is a mathematical model of ferromagnetism in statistical mechanics. It consists of discrete spin variables on a lattice, with nearest-neighbor interactions. The model undergoes a phase transition at a critical temperature Tc ≈ 2.269 (in units of J/kB).

The simulation calculates several key observables:
- Energy and specific heat
- Magnetization and magnetic susceptibility
- Critical temperature estimates via finite-size scaling

## Performance Optimization

This implementation includes several optimizations:
- Parallel temperature processing for multi-core utilization
- Numba-compiled core functions for near-C performance
- Temperature-dependent equilibration strategy
- Ordered initialization for low temperatures

## License

This project is licensed under the MIT License - see the LICENSE file for details.


