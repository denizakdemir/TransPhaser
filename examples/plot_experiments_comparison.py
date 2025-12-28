"""
Generate comparison plots for the two TransPhaser experiments
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

# Data
methods = ['Random', 'Frequency', 'EM', 'Beagle', 'TransPhaser']

# Multi-population (with covariates for TransPhaser)
multipop_accuracy = [14.85, 48.50, 54.25, 79.35, 85.30]
multipop_hamming = [3.43, 1.97, 1.78, 0.70, 0.48]

# Single population (no covariates)
onepop_accuracy = [14.60, 54.60, 60.10, 81.80, 85.10]
onepop_hamming = [3.41, 1.65, 1.45, 0.56, 0.47]

# Create figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Colors
colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']

# Plot 1: Accuracy Comparison
x = np.arange(len(methods))
width = 0.35

bars1 = axes[0].bar(x - width/2, multipop_accuracy, width, label='Multi-Population\n(TransPhaser uses covariates)', 
                     color=colors, alpha=0.8, edgecolor='black')
bars2 = axes[0].bar(x + width/2, onepop_accuracy, width, label='Single Population\n(No covariates)', 
                     color=colors, alpha=0.5, edgecolor='black', hatch='//')

axes[0].set_ylabel('Phasing Accuracy (%)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Method', fontsize=12, fontweight='bold')
axes[0].set_title('Phasing Accuracy: Fair vs. Covariate-Enhanced Comparison', fontsize=13, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(methods, rotation=0, fontsize=10)
axes[0].legend(loc='upper left', fontsize=10)
axes[0].set_ylim(0, 100)
axes[0].grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

# Plot 2: Hamming Distance Comparison
bars3 = axes[1].bar(x - width/2, multipop_hamming, width, label='Multi-Population\n(TransPhaser uses covariates)', 
                     color=colors, alpha=0.8, edgecolor='black')
bars4 = axes[1].bar(x + width/2, onepop_hamming, width, label='Single Population\n(No covariates)', 
                     color=colors, alpha=0.5, edgecolor='black', hatch='//')

axes[1].set_ylabel('Avg. Hamming Distance (lower is better)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Method', fontsize=12, fontweight='bold')
axes[1].set_title('Hamming Distance: Error Rate Comparison', fontsize=13, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(methods, rotation=0, fontsize=10)
axes[1].legend(loc='upper right', fontsize=10)
axes[1].set_ylim(0, 4)
axes[1].grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('examples/output/experiments_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Comparison plot saved to: examples/output/experiments_comparison.png")
plt.close()

# Create a second figure: Delta comparison
fig2, ax = plt.subplots(1, 1, figsize=(10, 6))

# Calculate delta (Single - Multi) - positive means better in single population
delta_accuracy = np.array(onepop_accuracy) - np.array(multipop_accuracy)

bars = ax.barh(methods, delta_accuracy, color=colors, alpha=0.7, edgecolor='black')

# Color bars by direction
for i, (bar, delta) in enumerate(zip(bars, delta_accuracy)):
    if delta > 0:
        bar.set_color('#2ca02c')  # Green for improvement
    else:
        bar.set_color('#d62728')  # Red for decline

ax.set_xlabel('Δ Accuracy (Single-Pop minus Multi-Pop) [%]', fontsize=12, fontweight='bold')
ax.set_ylabel('Method', fontsize=12, fontweight='bold')
ax.set_title('Impact of Removing Population Structure\\n(Positive = Method benefits from reduced complexity)', 
             fontsize=13, fontweight='bold')
ax.axvline(0, color='black', linewidth=1.5, linestyle='--', alpha=0.5)
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, delta) in enumerate(zip(bars, delta_accuracy)):
    x_pos = delta + (0.2 if delta > 0 else -0.2)
    ha = 'left' if delta > 0 else 'right'
    ax.text(x_pos, bar.get_y() + bar.get_height()/2., 
            f'{delta:+.2f}%', ha=ha, va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('examples/output/experiments_delta_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Delta comparison plot saved to: examples/output/experiments_delta_comparison.png")
plt.close()

print("\n" + "="*60)
print("KEY INSIGHTS FROM THE COMPARISONS:")
print("="*60)
print(f"1. TransPhaser maintains ~85% accuracy in BOTH settings")
print(f"   - Multi-pop with covariates: 85.30%")
print(f"   - Single-pop no covariates:  85.10%")
print(f"   - Δ = {delta_accuracy[-1]:.2f}% (remarkably stable!)")
print()
print(f"2. TransPhaser STILL beats Beagle without covariates:")
print(f"   - TransPhaser: 85.10%")
print(f"   - Beagle:      81.80%")
print(f"   - Advantage:   +3.30 percentage points")
print()
print(f"3. Classical methods benefit more from single-population:")
print(f"   - EM:         +{onepop_accuracy[2] - multipop_accuracy[2]:.2f}%")
print(f"   - Frequency:  +{onepop_accuracy[1] - multipop_accuracy[1]:.2f}%")
print(f"   (Less heterogeneity → easier to learn frequencies)")
print()
print("4. TransPhaser's advantage is ARCHITECTURAL, not just metadata!")
print("="*60)
