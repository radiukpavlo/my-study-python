import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from typing import List, Dict, Any

def plot_confusion_matrix(matrix: np.ndarray, labels: List[str], filename: str):
    """
    Generates and saves a publication-quality confusion matrix plot with
    custom number formatting, styled to match the provided PDF examples.

    Args:
        matrix (np.ndarray): The confusion matrix data.
        labels (List[str]): The class labels for the x and y axes.
        filename (str): The path to save the output PDF file.
    """
    # --- 1. Setup Plot Style and Font ---
    plt.style.use('default')
    font_properties = {
        'family': 'sans-serif',
        'size': 18,
        'weight': 'bold'
    }
    plt.rc('font', **font_properties)
    plt.rc('axes', titlesize=20)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    # --- 2. Create the Heatmap Plot ---
    fig, ax = plt.subplots(figsize=(8, 7))

    sns.heatmap(
        matrix,
        annot=True,
        fmt='d',  # Use integer formatting
        cmap='Blues',
        linewidths=0.5,
        linecolor='gray',
        cbar=True,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        annot_kws={"size": 18, "weight": "bold"},
    )

    # --- 3. Customize Annotation Colors ---
    # Set a threshold for when to switch text color from black to white
    color_threshold = matrix.max() * 0.6
    
    # Iterate through text objects to set color based on background
    for text_obj in ax.texts:
        # The text object's text is the data value as a string
        data_value = int(text_obj.get_text())
        if data_value > color_threshold:
            text_obj.set_color('white')
        else:
            text_obj.set_color('black')

    # --- 4. Finalize and Save Plot ---
    ax.set_xlabel('Predicted label', fontsize=18, weight='bold')
    ax.set_ylabel('True label', fontsize=18, weight='bold')
    
    # Set labels with bold formatting and horizontal orientation for x-axis
    ax.set_yticklabels(ax.get_yticklabels(), rotation=90, va='center', weight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', weight='bold')

    # Ensure layout is tight before saving
    plt.tight_layout()
    
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close(fig) # Close the figure to free memory
    print(f"Confusion matrix plot successfully saved as '{filename}'")


if __name__ == '__main__':
    # Data extracted from Figures 9a-f in the provided PDF
    confusion_matrices_data: List[Dict[str, Any]] = [
        {
            "filename": "67_fig_09a.pdf",
            "labels": ['People', 'Vehicles'],
            "matrix": np.array([[979, 18], [14, 3757]])
        },
        {
            "filename": "67_fig_09b.pdf",
            "labels": ['Trucks', 'Other vehicles'],
            "matrix": np.array([[590, 56], [39, 2567]])
        },
        {
            "filename": "67_fig_09c.pdf",
            "labels": ['Van', 'Bus', 'Other vehicles'],
            "matrix": np.array([[411, 22, 8], [14, 355, 21], [16, 42, 1539]])
        },
        {
            "filename": "67_fig_09d.pdf",
            "labels": ['People', 'Vehicles'],
            "matrix": np.array([[989, 28], [34, 3717]])
        },
        {
            "filename": "67_fig_09e.pdf",
            "labels": ['Trucks', 'Other vehicles'],
            "matrix": np.array([[601, 61], [23, 2576]])
        },
        {
            "filename": "67_fig_09f.pdf",
            "labels": ['Van', 'Bus', 'Other vehicles'],
            "matrix": np.array([[406, 20, 18], [19, 352, 28], [31, 42, 1519]])
        }
    ]

    # Generate and save all six confusion matrix plots
    for data in confusion_matrices_data:
        plot_confusion_matrix(
            matrix=data["matrix"],
            labels=data["labels"],
            filename=data["filename"]
        )