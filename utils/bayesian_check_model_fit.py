import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def check_model_fit(observed_durations, predicted_samples, plot_path=None):
    """
    Compare observed durations with posterior predictive samples.
    Args:
        observed_durations (np.array): Array of observed unemployment durations.
        predicted_samples (np.array): Flattened posterior predictive samples.
        plot_path (str, optional): If provided, saves the plot to this path instead of showing it.
    """

    # 1. Compare summary statistics
    print("\nSummary Statistics Comparison:")
    print(f"Observed Mean: {np.mean(observed_durations):.2f} weeks")
    print(f"Predicted Mean: {np.mean(predicted_samples):.2f} weeks")
    print(f"Observed Median: {np.median(observed_durations):.2f} weeks")
    print(f"Predicted Median: {np.median(predicted_samples):.2f} weeks")
    print(f"Observed Std Dev: {np.std(observed_durations):.2f} weeks")
    print(f"Predicted Std Dev: {np.std(predicted_samples):.2f} weeks")

    # 2. Plot only KDEs and rug plots
    plt.figure(figsize=(10, 6))
    sns.kdeplot(observed_durations, color='blue', lw=2, label='Observed KDE', fill=True, alpha=0.3)
    sns.kdeplot(predicted_samples, color='green', lw=2, label='Predicted KDE', fill=True, alpha=0.3)
    sns.rugplot(observed_durations, color='blue', alpha=0.15, label='Observed Data')

    # Add percentiles for predicted
    for q, color in zip([2.5, 50, 97.5], ['red', 'black', 'red']):
        perc = np.percentile(predicted_samples, q)
        plt.axvline(perc, color=color, linestyle='--', alpha=0.7, label=f'Predicted {q}th %ile' if q != 50 else 'Predicted Median')

    # Limit x-axis to 99th percentile of combined data
    plt.xlim([0, np.percentile(np.concatenate([observed_durations, predicted_samples]), 99)])

    plt.xlabel('Unemployment Duration (weeks)')
    plt.ylabel('Density')
    plt.title('Posterior Predictive Check: Observed vs Predicted')
    plt.legend()

    if plot_path:
        plt.savefig(plot_path)
        print(f"\nPlot saved to {plot_path}")
    else:
        plt.show()