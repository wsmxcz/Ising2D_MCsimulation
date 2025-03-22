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
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
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

def plot_quantity_vs_temperature(df, L_values, quantity, ylabel, title, filename):
    """
    Generalized function to plot a given quantity vs temperature for different L values.
    """
    plt.figure(figsize=(4, 3))
    for idx, L in enumerate(L_values):
        df_L = df[df["L"] == L]
        plt.plot(df_L["T"], df_L[quantity], 
                 label=f"L={L}",
                 color=COLOR_PALETTE[idx % len(COLOR_PALETTE)],
                 marker=MARKER_STYLES[idx % len(MARKER_STYLES)],
                 markersize=2,
                 linewidth=1)
    plt.xlabel("Temperature (T)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, format="pdf")
    plt.close()

def plot_results(csv_filename="ising_results.csv"):
    """Generate visualization of simulation results from CSV data"""
    # Read CSV data
    df = pd.read_csv(csv_filename)
    df["T"] = df["T"].astype(float).round(4)
    L_values = sorted(df["L"].unique())
    
    plot_quantity_vs_temperature(df, L_values, "avg_energy", "Average Energy <E>", "Average Energy vs Temperature", "figure1_energy.pdf")
    plot_quantity_vs_temperature(df, L_values, "avg_abs_magnetization", "Average Absolute Magnetization <|M|>", "Average Absolute Magnetization vs Temperature", "figure2_magnetization.pdf")
    plot_quantity_vs_temperature(df, L_values, "susceptibility", "Susceptibility (χ)", "Susceptibility vs Temperature", "figure3_susceptibility.pdf")
    plot_quantity_vs_temperature(df, L_values, "specific_heat", "Specific Heat (C)", "Specific Heat vs Temperature", "figure4_specific_heat.pdf")

    # --- Figure 5: Critical Temperature Analysis ---
    if len(L_values) > 1:
        def get_Tc_from_peak(df_L, observable):
            idx_max = df_L[observable].idxmax()
            return round(df.loc[idx_max, "T"], 4)

        # Collect Tc estimates from peaks
        Tc_chi_values = [get_Tc_from_peak(df[df["L"] == L], "susceptibility") for L in L_values]
        Tc_C_values = [get_Tc_from_peak(df[df["L"] == L], "specific_heat") for L in L_values]
        
        # Plot Tc vs L without fitting
        plt.figure(figsize=(4, 3))
        plt.plot(L_values, Tc_chi_values,
                label="Tc from susceptibility", 
                color=COLOR_PALETTE[0],
                marker=MARKER_STYLES[0],
                markersize=4,
                linewidth=1.5,
                linestyle='-')
        
        plt.plot(L_values, Tc_C_values,
                label="Tc from specific heat", 
                color=COLOR_PALETTE[1],
                marker=MARKER_STYLES[1],
                markersize=4,
                linewidth=1.5,
                linestyle='-')
        
        # Add exact Tc line
        plt.axhline(y=2.269, color="black", linestyle="--", label="Exact $T_c = 2.269$")
        
        plt.xlabel("System Size L")
        plt.ylabel("Critical Temperature $T_c$")
        plt.title("Critical Temperature vs System Size")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("figure5_Tc_vs_L.pdf", format="pdf")
        
        # Create 1/L extrapolation plot (保留拟合)
        plt.figure(figsize=(4, 3))
        L_inv = [1/L for L in L_values]
        
        plt.plot(L_inv, Tc_chi_values,
                label="Tc from susceptibility", 
                color=COLOR_PALETTE[0],
                marker=MARKER_STYLES[0],
                markersize=4)
        
        plt.plot(L_inv, Tc_C_values,
                label="Tc from specific heat", 
                color=COLOR_PALETTE[1],
                marker=MARKER_STYLES[1],
                markersize=4)
        
        # Linear scaling function for 1/L extrapolation
        def linear_scaling(L_inv, Tc_inf, b):
            return Tc_inf + b * L_inv
        
        # Linear fit to 1/L data for both observables
        try:
            bounds_lin_chi = ([2.0, 0.0], [2.5, 10.0])
            params_lin_chi, _ = optimize.curve_fit(linear_scaling, L_inv, Tc_chi_values,
                                                bounds=bounds_lin_chi, maxfev=5000)
            Tc_inf_lin_chi, b_chi = params_lin_chi
            
            bounds_lin_C = ([2.0, 0.0], [2.5, 10.0])
            params_lin_C, _ = optimize.curve_fit(linear_scaling, L_inv, Tc_C_values,
                                                bounds=bounds_lin_C, maxfev=5000)
            Tc_inf_lin_C, b_C = params_lin_C
            
            L_inv_fit = np.linspace(0, max(L_inv)*1.2, 100)
            Tc_lin_fit_chi = linear_scaling(L_inv_fit, Tc_inf_lin_chi, b_chi)
            Tc_lin_fit_C = linear_scaling(L_inv_fit, Tc_inf_lin_C, b_C)
            
            plt.plot(L_inv_fit, Tc_lin_fit_chi,
                    color=COLOR_PALETTE[0],
                    linestyle="--", 
                    label=f"Susceptibility: $T_c = {Tc_inf_lin_chi:.4f}$")
            plt.plot(L_inv_fit, Tc_lin_fit_C,
                    color=COLOR_PALETTE[1],
                    linestyle="--",
                    label=f"Specific heat: $T_c = {Tc_inf_lin_C:.4f}$")
            
            plt.plot(0, Tc_inf_lin_chi, marker='o', markersize=4, color=COLOR_PALETTE[0])
            plt.plot(0, Tc_inf_lin_C, marker='s', markersize=4, color=COLOR_PALETTE[1])
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