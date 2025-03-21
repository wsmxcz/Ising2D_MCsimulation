# plot.py - Visualization tools for Ising model simulation results

import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
from scipy import optimize

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Plot style settings with larger fonts
PLOT_STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12, 
    'legend.fontsize': 12,
    'figure.dpi': 600,
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.5 
}

# Color palette
COLOR_PALETTE = [
    "#E46D85",  # Rose pink
    "#6FB36A",  # Jade green
    "#5F9E9C",  # Celadon
    "#D4A55E",  # Amber yellow
    "#C84C4C",  # Vermilion red
    "#4CA89E",  # Peacock blue
]

# Marker styles for data points
MARKER_STYLES = ["o", "s", "^", "D", "v", "p"]

# Update matplotlib style
plt.rcParams.update(PLOT_STYLE)

def plot_ising_results(csv_pattern="data/ising_results_L*.csv"):
    """Generate visualization of Ising model simulation results"""
    # Find all result files
    result_files = glob.glob(csv_pattern)
    if not result_files:
        print(f"No result files found matching pattern {csv_pattern}")
        return
    
    # Read and process each file
    data_frames = []
    for file in result_files:
        df = pd.read_csv(file)
        df["T"] = df["T"].astype(float).round(4)  # Ensure consistent temperature precision
        data_frames.append(df)
    
    # Combine all data
    combined_df = pd.concat(data_frames, ignore_index=True)
    
    # Get sorted system sizes
    L_values = sorted(combined_df["L"].unique())
    
    # Create output directory for figures
    os.makedirs("figures", exist_ok=True)
    
    # --- Figure 1: Energy vs Temperature ---
    plot_energy(combined_df, L_values)
    
    # --- Figure 2: Magnetization vs Temperature ---
    plot_magnetization(combined_df, L_values)
    
    # --- Figure 3: Susceptibility vs Temperature ---
    plot_susceptibility(combined_df, L_values)
    
    # --- Figure 4: Specific Heat vs Temperature ---
    plot_specific_heat(combined_df, L_values)
    
    # --- Figure 5: Critical Temperature Analysis ---
    if len(L_values) > 1:
        plot_critical_temperature(combined_df, L_values)

def plot_energy(df, L_values):
    """Plot average energy vs temperature for all system sizes"""
    plt.figure(figsize=(10, 7))
    for idx, L in enumerate(L_values):
        df_L = df[df["L"] == L]
        plt.plot(df_L["T"], df_L["avg_energy"], 
                 label=f"L={L}", 
                 color=COLOR_PALETTE[idx % len(COLOR_PALETTE)],
                 marker=MARKER_STYLES[idx % len(MARKER_STYLES)],
                 markersize=6,
                 linewidth=1.5)
    
    plt.xlabel("Temperature (T)")
    plt.ylabel("Average Energy per Site")
    plt.title("Energy vs Temperature")
    plt.legend(frameon=True, fancybox=True, facecolor='white', edgecolor='gray')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/energy.pdf", format="pdf")
    plt.close()

def plot_magnetization(df, L_values):
    """Plot absolute magnetization vs temperature for all system sizes"""
    plt.figure(figsize=(10, 7))
    for idx, L in enumerate(L_values):
        df_L = df[df["L"] == L]
        plt.plot(df_L["T"], df_L["avg_abs_magnetization"], 
                 label=f"L={L}",
                 color=COLOR_PALETTE[idx % len(COLOR_PALETTE)],
                 marker=MARKER_STYLES[idx % len(MARKER_STYLES)],
                 markersize=6,
                 linewidth=1.5)
    
    plt.xlabel("Temperature (T)")
    plt.ylabel("Average Absolute Magnetization")
    plt.title("Magnetization vs Temperature")
    plt.legend(frameon=True, fancybox=True, facecolor='white', edgecolor='gray')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/magnetization.pdf", format="pdf")
    plt.close()

def plot_susceptibility(df, L_values):
    """Plot magnetic susceptibility vs temperature for all system sizes"""
    plt.figure(figsize=(10, 7))
    for idx, L in enumerate(L_values):
        df_L = df[df["L"] == L]
        plt.plot(df_L["T"], df_L["susceptibility"], 
                 label=f"L={L}",
                 color=COLOR_PALETTE[idx % len(COLOR_PALETTE)],
                 marker=MARKER_STYLES[idx % len(MARKER_STYLES)],
                 markersize=6,
                 linewidth=1.5)
    
    plt.xlabel("Temperature (T)")
    plt.ylabel("Magnetic Susceptibility ($\\chi$)")
    plt.title("Susceptibility vs Temperature")
    plt.legend(frameon=True, fancybox=True, facecolor='white', edgecolor='gray')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/susceptibility.pdf", format="pdf")
    plt.close()

def plot_specific_heat(df, L_values):
    """Plot specific heat vs temperature for all system sizes"""
    plt.figure(figsize=(10, 7))
    for idx, L in enumerate(L_values):
        df_L = df[df["L"] == L]
        plt.plot(df_L["T"], df_L["specific_heat"], 
                 label=f"L={L}",
                 color=COLOR_PALETTE[idx % len(COLOR_PALETTE)],
                 marker=MARKER_STYLES[idx % len(MARKER_STYLES)],
                 markersize=6,
                 linewidth=1.5)
    
    plt.xlabel("Temperature (T)")
    plt.ylabel("Specific Heat (C)")
    plt.title("Specific Heat vs Temperature")
    plt.legend(frameon=True, fancybox=True, facecolor='white', edgecolor='gray')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/specific_heat.pdf", format="pdf")
    plt.close()

def plot_critical_temperature(df, L_values):
    """Plot critical temperature analysis including finite-size scaling"""
    # Fit function for finite size scaling
    def finite_size_scaling(L, Tc_inf, a, nu):
        return Tc_inf + a * L**(-1/nu)
    
    # Linear scaling function
    def linear_scaling(L_inv, Tc_inf, b):
        return Tc_inf + b * L_inv
    
    # Extract critical temperatures from peaks
    Tc_chi = []  # From susceptibility
    Tc_C = []    # From specific heat
    
    for L in L_values:
        df_L = df[df["L"] == L]
        
        # Find temperature at susceptibility peak
        idx_chi_peak = df_L["susceptibility"].idxmax()
        Tc_chi.append(round(df.loc[idx_chi_peak, "T"], 4))
        
        # Find temperature at specific heat peak
        idx_C_peak = df_L["specific_heat"].idxmax()
        Tc_C.append(round(df.loc[idx_C_peak, "T"], 4))
    
    # --- Figure 5a: Tc vs L ---
    plt.figure(figsize=(10, 7))
    
    # Plot data points
    plt.plot(L_values, Tc_chi,
             label="From susceptibility", 
             color=COLOR_PALETTE[0],
             marker=MARKER_STYLES[0],
             markersize=10,
             linestyle='-',
             linewidth=1.5)
    
    plt.plot(L_values, Tc_C,
             label="From specific heat", 
             color=COLOR_PALETTE[1],
             marker=MARKER_STYLES[1],
             markersize=10,
             linestyle='-',
             linewidth=1.5)
    
    # Perform finite-size scaling fits
    try:
        # Initial parameters and bounds
        p0 = [2.269, 1.0, 1.0]
        bounds = ([2.0, 0.1, 0.1], [2.5, 10.0, 2.0])
        
        # Fit susceptibility data
        params_chi, _ = optimize.curve_fit(finite_size_scaling, L_values, Tc_chi, 
                                         p0=p0, bounds=bounds, maxfev=10000)
        Tc_inf_chi, a_chi, nu_chi = params_chi
        
        # Fit specific heat data
        params_C, _ = optimize.curve_fit(finite_size_scaling, L_values, Tc_C, 
                                       p0=p0, bounds=bounds, maxfev=10000)
        Tc_inf_C, a_C, nu_C = params_C
        
        # Generate curves for plotting
        L_fit = np.linspace(min(L_values), max(L_values)*1.5, 100)
        Tc_fit_chi = finite_size_scaling(L_fit, Tc_inf_chi, a_chi, nu_chi)
        Tc_fit_C = finite_size_scaling(L_fit, Tc_inf_C, a_C, nu_C)
        
        # Plot fitted curves
        plt.plot(L_fit, Tc_fit_chi,
                 color=COLOR_PALETTE[0], 
                 label=f"$\\chi$ fit: $T_c = {Tc_inf_chi:.4f}$",
                 linestyle='--',
                 linewidth=2.0)
        
        plt.plot(L_fit, Tc_fit_C,
                 color=COLOR_PALETTE[1], 
                 label=f"C fit: $T_c = {Tc_inf_C:.4f}$",
                 linestyle='--',
                 linewidth=2.0)
        
        # Show extrapolated values
        plt.axhline(y=Tc_inf_chi, color=COLOR_PALETTE[0], linestyle=':', linewidth=1.5)
        plt.axhline(y=Tc_inf_C, color=COLOR_PALETTE[1], linestyle=':', linewidth=1.5)
    except Exception as e:
        print(f"Warning: Finite-size scaling fit failed: {e}")
    
    # Add exact Tc line
    plt.axhline(y=2.269, color="black", linestyle="--", 
                label="Exact $T_c = 2.269$", linewidth=2.0)
    
    plt.xlabel("System Size L")
    plt.ylabel("Critical Temperature $T_c$")
    plt.title("Critical Temperature vs System Size")
    plt.legend(frameon=True, fancybox=True, facecolor='white', edgecolor='gray')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/Tc_vs_L.pdf", format="pdf")
    plt.close()
    
    # --- Figure 5b: Tc vs 1/L ---
    plt.figure(figsize=(10, 7))
    L_inv = [1/L for L in L_values]
    
    # Plot data points
    plt.plot(L_inv, Tc_chi,
            label="From susceptibility", 
            color=COLOR_PALETTE[0],
            marker=MARKER_STYLES[0],
            markersize=10)
    
    plt.plot(L_inv, Tc_C,
            label="From specific heat", 
            color=COLOR_PALETTE[1],
            marker=MARKER_STYLES[1],
            markersize=10)
    
    # Linear extrapolation
    try:
        # Linear fit bounds
        bounds = ([2.0, 0.0], [2.5, 10.0])
        
        # Fit susceptibility data
        params_lin_chi, _ = optimize.curve_fit(linear_scaling, L_inv, Tc_chi,
                                             bounds=bounds, maxfev=5000)
        Tc_inf_lin_chi, b_chi = params_lin_chi
        
        # Fit specific heat data
        params_lin_C, _ = optimize.curve_fit(linear_scaling, L_inv, Tc_C,
                                           bounds=bounds, maxfev=5000)
        Tc_inf_lin_C, b_C = params_lin_C
        
        # Generate curves for plotting
        L_inv_fit = np.linspace(0, max(L_inv)*1.2, 100)
        Tc_lin_fit_chi = linear_scaling(L_inv_fit, Tc_inf_lin_chi, b_chi)
        Tc_lin_fit_C = linear_scaling(L_inv_fit, Tc_inf_lin_C, b_C)
        
        # Plot fitted curves
        plt.plot(L_inv_fit, Tc_lin_fit_chi,
                color=COLOR_PALETTE[0], 
                label=f"$\\chi$: $T_c = {Tc_inf_lin_chi:.4f}$",
                linestyle='--',
                linewidth=2.0)
        
        plt.plot(L_inv_fit, Tc_lin_fit_C,
                color=COLOR_PALETTE[1], 
                label=f"C: $T_c = {Tc_inf_lin_C:.4f}$",
                linestyle='--',
                linewidth=2.0)
        
        # Mark extrapolated values
        plt.plot(0, Tc_inf_lin_chi, marker='o', markersize=12, color=COLOR_PALETTE[0])
        plt.plot(0, Tc_inf_lin_C, marker='s', markersize=12, color=COLOR_PALETTE[1])
    except Exception as e:
        print(f"Warning: Linear extrapolation fit failed: {e}")
    
    # Add exact Tc line
    plt.axhline(y=2.269, color="black", linestyle="--", 
                label="Exact $T_c = 2.269$", linewidth=2.0)
    
    plt.xlabel("1/L")
    plt.ylabel("Critical Temperature $T_c$")
    plt.title("Finite-Size Scaling: $T_c$ vs 1/L")
    plt.legend(frameon=True, fancybox=True, facecolor='white', edgecolor='gray')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/Tc_extrapolation.pdf", format="pdf")
    plt.close()

if __name__ == "__main__":
    plot_ising_results()