import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
import pandas as pd
from textwrap import wrap
import os

# Global variable for PDF curve width
PDF_LINEWIDTH = 5

def main():
    # Set random seed
    np.random.seed(42)

    # Generate dataset
    n_samples = 1000
    means = [-0.3, 0.1, 0.4]
    variances = [0.02, 0.05, 0.03]
    stds = [np.sqrt(var) for var in variances]

    data = []
    colors = ['red', 'green', 'blue']
    true_labels = []
    for i in range(3):
        samples = np.random.normal(means[i], stds[i], n_samples)
        data.append(samples)
        true_labels.extend([i] * n_samples)

    data = np.concatenate(data)
    noise = np.random.uniform(-0.05, 0.05, len(data))
    data += noise
    true_labels = np.array(true_labels)

    # Create DataFrame
    df = pd.DataFrame({'value': data, 'true_label': true_labels})
    df['color'] = df['true_label'].map({0: 'red', 1: 'green', 2: 'blue'})

    # GMM classification
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(data.reshape(-1, 1))
    predicted_labels = gmm.predict(data.reshape(-1, 1))
    df['predicted_label'] = predicted_labels
    df['predicted_color'] = df['predicted_label'].map({0: 'red', 1: 'green', 2: 'blue'})

    # Save data
    df.to_csv('distributions_data.csv', index=False)

    # Instructions for each tool
    instructions = {
        'Power BI': (
            "Import distributions_data.csv. Graph 1: Clustered Column Chart, bin 'value' (50 bins), count 'value', legend 'color'. "
            "Overlay PDFs with Line Chart using DAX for norm.pdf. Graph 2: Histogram of 'value' (grey), overlay PDFs with 'predicted_color'."
        ),
        'Streamlit': (
            "Save as app.py, install streamlit, run 'streamlit run app.py'. Displays graphs via st.pyplot or st.image. "
            "Data in distributions_data.csv."
        ),
        'Tableau': (
            "Import distributions_data.csv. Graph 1: Histogram of 'value' (50 bins), color by 'color', dual-axis PDFs (calculated fields). "
            "Graph 2: Histogram of 'value' (grey), overlay PDFs by 'predicted_color'."
        )
    }

    # Generate PNGs
    for tool, instruction in instructions.items():
        fig = plt.figure(figsize=(8, 12))
        ax1 = fig.add_subplot(2, 1, 1)
        bins = np.linspace(min(data), max(data), 50)
        for i, color in enumerate(colors):
            subset = df[df['true_label'] == i]['value']
            ax1.hist(subset, bins=bins, color=color, alpha=0.5, label=f'{color.capitalize()} Samples')
            x = np.linspace(min(data), max(data), 100)
            pdf = norm.pdf(x, means[i], stds[i]) * n_samples * (bins[1] - bins[0])
            ax1.plot(x, pdf, color=color, linestyle='--', linewidth=PDF_LINEWIDTH, label=f'{color.capitalize()} PDF')
        ax1.set_title('True Distributions with Known PDFs')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Count')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(2, 1, 2)
        ax2.hist(data, bins=bins, color='grey', alpha=0.7, label='All Samples')
        for i, color in enumerate(colors):
            subset = df[df['predicted_label'] == i]['value']
            if len(subset) > 0:
                mean_est = gmm.means_[i][0]
                std_est = np.sqrt(gmm.covariances_[i][0][0])
                x = np.linspace(min(data), max(data), 100)
                pdf = norm.pdf(x, mean_est, std_est) * len(subset) * (bins[1] - bins[0])
                ax2.plot(x, pdf, color=color, linestyle='--', linewidth=PDF_LINEWIDTH, label=f'{color.capitalize()} Estimated PDF')
        ax2.set_title('All Samples with Estimated PDFs')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Count')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.suptitle(f'Visualization in {tool}', fontsize=16)
        wrapped_instruction = "\n".join(wrap(instruction, 80))
        fig.text(0.5, 0.02, wrapped_instruction, ha='center', va='bottom', fontsize=8, wrap=True)

        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        plt.savefig(f'images/{tool.lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    main()
