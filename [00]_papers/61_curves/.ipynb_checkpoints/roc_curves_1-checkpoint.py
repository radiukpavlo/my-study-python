import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def find_distribution_separation(target_auc, n_samples=2000, tolerance=0.001, max_iter=100):
    """
    Finds the mean separation between two normal distributions (std=1) that results
    in a given target AUC for the ROC curve.

    Args:
        target_auc (float): The desired AUC (e.g., 0.95).
        n_samples (int): The number of samples per class to generate for the test.
        tolerance (float): The acceptable error between actual and target AUC.
        max_iter (int): The maximum number of search iterations.

    Returns:
        float: The mean for the positive class distribution that yields the target AUC.
    """
    mean_separation = 1.0  # Initial guess for the mean separation
    step = 0.5  # Initial search step

    for _ in range(max_iter):
        # Generate scores from two normal distributions
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
        
        # Adjust mean_separation based on the error
        if error < 0:
            mean_separation += step
        else:
            mean_separation -= step
            
        # Reduce the step size for finer search
        step *= 0.95
        
    # If max_iter is reached, return the best found value
    return mean_separation

def generate_stepped_roc_curves():
    """
    Generates and plots three realistic, step-like ROC curves with specified AUCs
    and saves the plot as a PDF.
    """
    # --- Define target AUCs and corresponding plot properties ---
    target_aucs = [0.95, 0.97, 0.98]
    class_labels = ['Crack', 'Erosion', 'Hotspot']
    colors = ['darkorange', 'cornflowerblue', 'darkgreen']
    
    # --- Create the plot ---
    plt.figure(figsize=(8, 7))

    # --- Generate and plot each ROC curve ---
    for i, target_auc in enumerate(target_aucs):
        print(f"Generating curve for target AUC = {target_auc}...")
        
        # Find the necessary distribution separation to achieve the target AUC
        mean_sep = find_distribution_separation(target_auc)
        
        # Generate the final data for plotting
        n_plot_samples = 400 # Use a smaller number for a more 'steppy' look
        y_true = np.concatenate([np.zeros(n_plot_samples), np.ones(n_plot_samples)])
        y_scores = np.concatenate([
            np.random.normal(0, 1, n_plot_samples),
            np.random.normal(mean_sep, 1, n_plot_samples)
        ])
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Plot the ROC curve with a step-like appearance
        plt.plot(fpr, tpr, color=colors[i], lw=2, drawstyle='steps-post',
                 label=f'ROC curve for {class_labels[i]} (AUC = {roc_auc:.2f})')
        print(f"-> Achieved AUC: {roc_auc:.4f}\n")

    # --- Customize and Finalize the Plot ---
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Chance line
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Multi-Class Receiver Operating Characteristic Example', fontsize=16)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.4)

    # Save the plot to a PDF file
    output_filename = 'roc_curves_stepped.pdf'
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')

    return f"Step-like ROC curve plot has been saved to {output_filename}"

# --- Run the generation script ---
if __name__ == '__main__':
    # Set a seed for reproducibility of the random data
    np.random.seed(42)
    result_message = generate_stepped_roc_curves()
    print(result_message)