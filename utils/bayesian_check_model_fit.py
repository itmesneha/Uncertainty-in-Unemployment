import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def check_model_fit(observed_durations, predicted_samples, plot_path=None):
    """
    Compare observed durations with posterior predictive samples.

    Args:
        observed_durations (np.array): Array of observed unemployment durations.
        predicted_samples (np.array): Posterior predictive samples (e.g., lognormal_samples).
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

    # 2. Plot observed vs predicted histograms
    plt.figure(figsize=(10, 6))
    sns.histplot(observed_durations, bins=30, color='blue', label='Observed', kde=True, stat="density", alpha=0.6)
    sns.histplot(predicted_samples, bins=30, color='green', label='Predicted', kde=True, stat="density", alpha=0.6)
    plt.xlabel('Unemployment Duration (weeks)')
    plt.ylabel('Density')
    plt.title('Posterior Predictive Check: Observed vs Predicted')
    plt.legend()

    if plot_path:
        plt.savefig(plot_path)
        print(f"\nPlot saved to {plot_path}")
    else:
        plt.show()


