import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# ==========================================
#      1. INPUT YOUR REAL DATA HERE
# ==========================================
# Copy-paste the durations (in seconds) from your inference console logs.
# Blink a mix of Dots and Dashes!
REAL_HUMAN_BLINKS = [
    0.18, 0.22, 0.26, 0.19, 0.25,  # Real Dots (Example)
    0.65, 0.72, 0.81, 0.60, 0.75,  # Real Dashes (Example)
    0.31, 0.29, 0.55               # Messy/Ambiguous Blinks
]

# ==========================================
#      2. DEFINE SYNTHETIC PARAMETERS
# ==========================================
# These must match generate_synthetic_data.py
DOT_MEAN = 0.25
DASH_MEAN = 0.75
JITTER = 0.05
NUM_SAMPLES = 10000

def main():
    # 1. Generate Synthetic Distribution (What the LSTM learned)
    synthetic_dots = np.random.normal(DOT_MEAN, JITTER, NUM_SAMPLES // 2)
    synthetic_dashes = np.random.normal(DASH_MEAN, JITTER, NUM_SAMPLES // 2)
    
    # 2. Setup Plot
    plt.figure(figsize=(10, 6))
    
    # Plot Synthetic Data (The "Knowledge")
    plt.hist(synthetic_dots, bins=50, alpha=0.5, color='blue', density=True, label='Synthetic Training (Dots)')
    plt.hist(synthetic_dashes, bins=50, alpha=0.5, color='green', density=True, label='Synthetic Training (Dashes)')
    
    # Plot Gaussian Curves for aesthetics
    x = np.linspace(0, 1.2, 500)
    plt.plot(x, stats.norm.pdf(x, DOT_MEAN, JITTER), 'b--', alpha=0.6)
    plt.plot(x, stats.norm.pdf(x, DASH_MEAN, JITTER), 'g--', alpha=0.6)

    # 3. Plot REAL Human Data (The "Test")
    # We plot these as vertical lines to see where they fall
    for val in REAL_HUMAN_BLINKS:
        color = 'red'
        # Heuristic coloring just for visualization
        if val > 0.5: color = 'darkred' 
        plt.axvline(x=val, color=color, linestyle='-', alpha=0.8, linewidth=1.5)

    # Dummy line for legend
    plt.axvline(x=-1, color='red', label='Real-World Human Input')

    # 4. Formatting
    plt.title("SOP 4 Evidence: Synthetic Training vs. Real-World Inputs", fontsize=14)
    plt.xlabel("Blink Duration (seconds)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.xlim(0, 1.2)
    plt.grid(True, alpha=0.3)
    
    save_path = "SOP4_Generalization_Proof.png"
    plt.savefig(save_path)
    print(f"Graph saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    main()