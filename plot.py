# plot.py - Visualization tools for Ising model simulation results

import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
from scipy import optimize
from analysis import combine_csv_files

# Plot style settings
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
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.4
}

# Color palette
COLOR_PALETTE = [
    "#E46D85",  # Rose pink
    "#6FB36A",  # Jade green
    "#5F9E9C",  # Celadon
    "#D4A55E",  # Amber yellow
    "#C84C4C",  # Vermilion red
    "#4CA89E",  # Peacock blue
    "#666D34",  # Olive green
    "#8874B8",  # Purple
]

# Marker styles for data points
MARKER_STYLES = ["o", "D", "v", "^", "s", "p", "*", "X"]

# Update matplotlib style
plt.rcParams.update(PLOT_STYLE)

def combine_and_plot_results(csv_pattern="ising_results_L*.csv", combined_csv="ising_results_combined.csv"):
    """Combine multiple per-system-size CSV files and plot the results"""
    # Check if we need to combine files
    files = glob.glob(csv_pattern)
    if len(files) > 1:
        print(f"Found {len(files)} result files, combining them...")
        combine_csv_files(csv_pattern, combined_csv)
        plot_results(combined_csv)
    elif len(files) == 1:
        print(f"Found single result file {files[0]}, plotting directly...")
        plot_results(files[0])
    else:
        print(f"No result files found matching pattern {csv_pattern}")

def plot_results(csv_filename="ising_results.csv"):
    """Generate visualization of simulation results from CSV data"""
    # Read CSV data
    df = pd.read_csv(csv_filename)
    df["T"] = df["T"].astype(float).round(4)
    L_values = sorted(df["L"].unique())

    # --- Figure 1: Average Energy vs Temperature ---
    plt.figure(figsize=(6, 4))
    for idx, L in enumerate(L_values):
        df_L = df[df["L"] == L]
        plt.plot(df_L["T"], df_L["avg_energy"], 
                 label=f"L={L}", 
                 color=COLOR_PALETTE[idx % len(COLOR_PALETTE)],
                 marker=MARKER_STYLES[idx % len(MARKER_STYLES)],
                 markersize=4,
                 linewidth=1)
    plt.xlabel("Temperature (T)")
    plt.ylabel("Average Energy <E>")
    plt.title("Average Energy vs Temperature")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figure1_energy.pdf", format="pdf")
    plt.close()

    # --- Figure 2: Average Absolute Magnetization vs Temperature ---
    plt.figure(figsize=(6, 4))
    for idx, L in enumerate(L_values):
        df_L = df[df["L"] == L]
        plt.plot(df_L["T"], df_L["avg_abs_magnetization"], 
                 label=f"L={L}",
                 color=COLOR_PALETTE[idx % len(COLOR_PALETTE)],
                 marker=MARKER_STYLES[idx % len(MARKER_STYLES)],
                 markersize=4,
                 linewidth=1)
    plt.xlabel("Temperature (T)")
    plt.ylabel("Average Absolute Magnetization <|M|>")
    plt.title("Average Absolute Magnetization vs Temperature")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figure2_magnetization.pdf", format="pdf")
    plt.close()

    # --- Figure 3: Susceptibility vs Temperature ---
    plt.figure(figsize=(6, 4))
    for idx, L in enumerate(L_values):
        df_L = df[df["L"] == L]
        plt.plot(df_L["T"], df_L["susceptibility"], 
                 label=f"L={L}",
                 color=COLOR_PALETTE[idx % len(COLOR_PALETTE)],
                 marker=MARKER_STYLES[idx % len(MARKER_STYLES)],
                 markersize=4,
                 linewidth=1)
    plt.xlabel("Temperature (T)")
    plt.ylabel("Susceptibility (χ)")
    plt.title("Susceptibility vs Temperature")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figure3_susceptibility.pdf", format="pdf")
    plt.close()

    # --- Figure 4: Specific Heat vs Temperature ---
    plt.figure(figsize=(6, 4))
    for idx, L in enumerate(L_values):
        df_L = df[df["L"] == L]
        plt.plot(df_L["T"], df_L["specific_heat"], 
                 label=f"L={L}",
                 color=COLOR_PALETTE[idx % len(COLOR_PALETTE)],
                 marker=MARKER_STYLES[idx % len(MARKER_STYLES)],
                 markersize=4,
                 linewidth=1)
    plt.xlabel("Temperature (T)")
    plt.ylabel("Specific Heat (C)")
    plt.title("Specific Heat vs Temperature")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figure4_specific_heat.pdf", format="pdf")
    plt.close()

    # --- Figure 5: Critical Temperature Analysis ---
    if len(L_values) > 1:
        # Fit function for finite size scaling
        def finite_size_scaling(L, Tc_inf, a, nu):
            return Tc_inf + a * L**(-1/nu)

        # Collect Tc estimates from peaks
        Tc_chi_values = []
        Tc_C_values = []
        
        for L in L_values:
            df_L = df[df["L"] == L]
            
            # Find susceptibility peak
            idx_max_chi = df_L["susceptibility"].idxmax()
            Tc_chi = df.loc[idx_max_chi, "T"]
            Tc_chi_values.append(round(Tc_chi, 4))
            
            # Find specific heat peak
            idx_max_C = df_L["specific_heat"].idxmax()
            Tc_C = df.loc[idx_max_C, "T"]
            Tc_C_values.append(round(Tc_C, 4))
            
        # Plot Tc vs L
        plt.figure(figsize=(6, 4))
        
        plt.plot(L_values, Tc_chi_values,
                 label="Tc from susceptibility", 
                 color=COLOR_PALETTE[0],
                 marker=MARKER_STYLES[0],
                 markersize=8,
                 linewidth=1.5,
                 linestyle='-')
        
        plt.plot(L_values, Tc_C_values,
                 label="Tc from specific heat", 
                 color=COLOR_PALETTE[1],
                 marker=MARKER_STYLES[1],
                 markersize=8,
                 linewidth=1.5,
                 linestyle='-')
        
        # Perform finite-size scaling fits for both observables
        try:
            # Susceptibility fit
            p0_chi = [2.269, 1.0, 1.0]
            bounds_chi = ([2.0, 0.1, 0.1], [2.5, 10.0, 2.0])
            params_chi, _ = optimize.curve_fit(finite_size_scaling, L_values, Tc_chi_values, 
                                            p0=p0_chi, bounds=bounds_chi, maxfev=10000)
            Tc_inf_chi, a_chi, nu_chi = params_chi
            
            # Specific heat fit
            p0_C = [2.269, 1.0, 1.0]
            bounds_C = ([2.0, 0.1, 0.1], [2.5, 10.0, 2.0])
            params_C, _ = optimize.curve_fit(finite_size_scaling, L_values, Tc_C_values, 
                                          p0=p0_C, bounds=bounds_C, maxfev=10000)
            Tc_inf_C, a_C, nu_C = params_C
            
            # Generate curves for plotting
            L_fit = np.linspace(min(L_values), max(L_values)*1.5, 100)
            Tc_fit_chi = finite_size_scaling(L_fit, Tc_inf_chi, a_chi, nu_chi)
            Tc_fit_C = finite_size_scaling(L_fit, Tc_inf_C, a_C, nu_C)
            
            # Plot susceptibility fit
            plt.plot(L_fit, Tc_fit_chi,
                     color=COLOR_PALETTE[0], 
                     label=f"Susceptibility FSS: $T_c = {Tc_inf_chi:.4f}$",
                     linestyle='--')
            
            # Plot specific heat fit
            plt.plot(L_fit, Tc_fit_C,
                     color=COLOR_PALETTE[1], 
                     label=f"Specific heat FSS: $T_c = {Tc_inf_C:.4f}$",
                     linestyle='--')
            
            # Show the extrapolated values
            plt.axhline(y=Tc_inf_chi, color=COLOR_PALETTE[0], linestyle=':')
            plt.axhline(y=Tc_inf_C, color=COLOR_PALETTE[1], linestyle=':')
        except Exception as e:
            print(f"Warning: Could not perform finite-size scaling fit: {e}")
        
        # Add exact Tc line
        plt.axhline(y=2.269, color="black", linestyle="--", label="Exact $T_c = 2.269$")
        
        plt.xlabel("System Size L")
        plt.ylabel("Critical Temperature $T_c$")
        plt.title("Critical Temperature vs System Size")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("figure5_Tc_vs_L.pdf", format="pdf")
        
        # Create 1/L extrapolation plot
        plt.figure(figsize=(6, 4))
        L_inv = [1/L for L in L_values]
        
        plt.plot(L_inv, Tc_chi_values,
                label="Tc from susceptibility", 
                color=COLOR_PALETTE[0],
                marker=MARKER_STYLES[0],
                markersize=8)
        
        plt.plot(L_inv, Tc_C_values,
                label="Tc from specific heat", 
                color=COLOR_PALETTE[1],
                marker=MARKER_STYLES[1],
                markersize=8)
        
        # Linear fit to 1/L data for both observables
        try:
            def linear_scaling(L_inv, Tc_inf, b):
                return Tc_inf + b * L_inv
            
            # Susceptibility linear fit
            bounds_lin_chi = ([2.0, 0.0], [2.5, 10.0])
            params_lin_chi, _ = optimize.curve_fit(linear_scaling, L_inv, Tc_chi_values,
                                                bounds=bounds_lin_chi, maxfev=5000)
            Tc_inf_lin_chi, b_chi = params_lin_chi
            
            # Specific heat linear fit
            bounds_lin_C = ([2.0, 0.0], [2.5, 10.0])
            params_lin_C, _ = optimize.curve_fit(linear_scaling, L_inv, Tc_C_values,
                                              bounds=bounds_lin_C, maxfev=5000)
            Tc_inf_lin_C, b_C = params_lin_C
            
            # Generate curves for plotting
            L_inv_fit = np.linspace(0, max(L_inv)*1.2, 100)
            Tc_lin_fit_chi = linear_scaling(L_inv_fit, Tc_inf_lin_chi, b_chi)
            Tc_lin_fit_C = linear_scaling(L_inv_fit, Tc_inf_lin_C, b_C)
            
            # Plot susceptibility fit
            plt.plot(L_inv_fit, Tc_lin_fit_chi,
                    color=COLOR_PALETTE[0], 
                    label=f"Susceptibility: $T_c = {Tc_inf_lin_chi:.4f}$",
                    linestyle='-')
            
            # Plot specific heat fit
            plt.plot(L_inv_fit, Tc_lin_fit_C,
                    color=COLOR_PALETTE[1], 
                    label=f"Specific heat: $T_c = {Tc_inf_lin_C:.4f}$",
                    linestyle='-')
            
            # Mark the extrapolated values
            plt.plot(0, Tc_inf_lin_chi, marker='o', markersize=10, color=COLOR_PALETTE[0])
            plt.plot(0, Tc_inf_lin_C, marker='s', markersize=10, color=COLOR_PALETTE[1])
        except Exception as e:
            print(f"Warning: Could not perform linear extrapolation fit: {e}")
        
        plt.axhline(y=2.269, color="black", linestyle="--", label="Exact $T_c = 2.269$")
        
        plt.xlabel("1/L")
        plt.ylabel("Critical Temperature $T_c$")
        plt.title("Finite-Size Scaling: $T_c$ vs 1/L")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("figure5b_Tc_extrapolation.pdf", format="pdf")
        
        plt.close()

if __name__ == "__main__":
    combine_and_plot_results()