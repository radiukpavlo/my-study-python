import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def generate_sensitivity_plot():
    """
    Generates and saves a styled sensitivity analysis plot using only Matplotlib.
    """
    # --- Style Configuration using Matplotlib rcParams ---
    matplotlib.use("Agg")  # Use a non-interactive backend for script execution
    plt.rcdefaults() # Reset to default settings to start fresh

    # Define a font that is likely to be available, with a fallback
    try:
        plt.rcParams['font.family'] = 'Palatino Linotype'
    except:
        plt.rcParams['font.family'] = 'serif' # Fallback font

    # Define all styling attributes in a dictionary for a polished look
    style_params = {
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.color": "#cccccc",
        "grid.linestyle": "-",
        "grid.linewidth": 0.8,
    }
    # Apply the styling
    plt.rcParams.update(style_params)

    # --- Data Definition (Logic is unchanged) ---
    # Define the percentage perturbations of membership parameters
    perturbations = np.array([-20, -10, 0, 10, 20])

    # Define corresponding MAE values to illustrate the robustness of the FIS
    mae_values = np.array([0.18, 0.16, 0.14, 0.15, 0.17])

    # --- Plotting Section ---
    # Create a plot with a slightly larger size for better readability of bold fonts
    plt.figure(figsize=(8, 6))
    
    # Plot the data with enhanced visual properties
    plt.plot(
        perturbations,
        mae_values,
        marker='o',
        markersize=8,
        lw=2.5,
        color='#1f77b4'  # Classic Matplotlib blue
    )
    
    # Set titles and labels (styling is now handled by rcParams)
    plt.title('Sensitivity Analysis of the Fuzzy Inference System')
    plt.xlabel('Parameter Perturbation (%)')
    plt.ylabel('Mean Absolute Error (MAE)')
    
    # Ensure all elements fit within the figure cleanly
    plt.tight_layout()

    # --- Save the Figure ---
    output_filename = './img/61_fig_a1.pdf'
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')

    # Restore default Matplotlib settings for any subsequent plots
    plt.rcdefaults()

    return f"Styled sensitivity analysis figure saved as '{output_filename}'."

# --- Run the generation script ---
if __name__ == '__main__':
    message = generate_sensitivity_plot()
    print(message)