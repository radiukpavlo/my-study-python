import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# --- Configure matplotlib aesthetics ---
# Set font family, size, and weight globally for publication-quality style
matplotlib.rcParams["font.family"] = "Palatino Linotype"
matplotlib.rcParams["font.size"] = 14
matplotlib.rcParams["font.weight"] = "bold"
# Use a non-interactive backend to facilitate saving without display.
matplotlib.use("Agg")

def find_distribution_separation(target_auc, n_samples=2000, tolerance=0.001, max_iter=100):
    """
    Finds the mean separation between two normal distributions (std=1) that results
    in a given target AUC for the ROC curve.
    (This function's logic is unchanged).
    """
    mean_separation = 1.0
    step = 0.5

    for _ in range(max_iter):
        y_true = np.concatenate([np.zeros(n_samples), np.ones(n_samples)])
        y_scores = np.concatenate([
            np.random.normal(0, 1, n_samples),
            np.random.normal(mean_separation, 1, n_samples)
        ])
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        current_auc = auc(fpr, tpr)
        error = current_auc - target_auc
        if abs(error) < tolerance:
            return mean_separation
        if error < 0:
            mean_separation += step
        else:
            mean_separation -= step
        step *= 0.95
    return mean_separation

def generate_stepped_roc_curves():
    """
    Generates and plots three realistic, step-like ROC curves with specified AUCs
    and saves the plot as a PDF, using only Matplotlib for styling.
    """
    # --- Style Configuration using Matplotlib rcParams ---
    matplotlib.use("Agg")  # Use a non-interactive backend for script execution
    plt.rcdefaults() # Reset to default settings to start fresh

    # Define a font that is likely to be available, with a fallback
    try:
        plt.rcParams['font.family'] = 'Palatino Linotype'
    except:
        plt.rcParams['font.family'] = 'serif'

    # Define all styling attributes in a dictionary
    style_params = {
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.titlesize": 18,
        "axes.labelsize": 15,
        "legend.fontsize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.grid": True,         # Turn on the grid
        "grid.color": "#cccccc",    # Set grid color
        "grid.linestyle": "-",      # Set grid line style
        "grid.linewidth": 0.8,      # Set grid line width
    }
    # Apply the styling
    plt.rcParams.update(style_params)
    
    # --- Plotting Data and Properties ---
    target_aucs = [0.95, 0.97, 0.98]
    class_labels = ['Crack', 'Erosion', 'Hotspot']
    
    # Define classic "Pythonic" colors (from Matplotlib's 'tab10' palette)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue, Orange, Green

    # --- Create the plot ---
    plt.figure(figsize=(10, 7))

    # --- Generate and plot each ROC curve (Logic is unchanged) ---
    for i, target_auc in enumerate(target_aucs):
        print(f"Generating curve for target AUC = {target_auc}...")
        
        mean_sep = find_distribution_separation(target_auc)
        
        n_plot_samples = 400
        y_true = np.concatenate([np.zeros(n_plot_samples), np.ones(n_plot_samples)])
        y_scores = np.concatenate([
            np.random.normal(0, 1, n_plot_samples),
            np.random.normal(mean_sep, 1, n_plot_samples)
        ])
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Plot the ROC curve with a step-like appearance
        plt.plot(fpr, tpr, color=colors[i], lw=2.5, drawstyle='steps-post',
                 label=f'{class_labels[i]} (AUC = {roc_auc:.2f})')
        print(f"-> Achieved AUC: {roc_auc:.4f}\n")

    # --- Customize and Finalize the Plot ---
    # Plot the chance line
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--') 
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    # Set labels and title; styling is controlled by rcParams
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-Class Receiver Operating Characteristic')
    
    plt.legend(loc="lower right")

    # Save the plot to a PDF file
    output_filename = './img/61_fig_2_roc.pdf'
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')
    
    # Restore default Matplotlib settings
    plt.rcdefaults()

    return f"Styled ROC curve plot has been saved to {output_filename}"

# --- Run the generation script ---
if __name__ == '__main__':
    # Required libraries: pip install numpy matplotlib scikit-learn
    
    # Set a seed for reproducibility
    np.random.seed(42)
    result_message = generate_stepped_roc_curves()
    print(result_message)