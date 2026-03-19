"""Genera figuras de calidad publicación para el README."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 12

# Data
ch_counts = [64, 32, 16, 8, 4, 2]
models = {
    'ATCNet':       [0.742, 0.651, 0.639, 0.628, 0.584, 0.548],
    'EEGNet':       [0.733, 0.652, 0.640, 0.572, 0.583, 0.550],
    'ShallowConvNet':[0.725, 0.658, 0.647, 0.635, 0.599, 0.563],
    'Conformer':    [0.611, 0.655, 0.584, 0.595, 0.589, 0.548],
    'CSP+LDA':      [0.590, 0.589, 0.581, 0.587, 0.551, 0.508],
}

colors = {
    'ATCNet': '#E91E63', 'EEGNet': '#2196F3', 'ShallowConvNet': '#4CAF50',
    'Conformer': '#FF9800', 'CSP+LDA': '#78909C'
}
markers = {'ATCNet': 'D', 'EEGNet': 'o', 'ShallowConvNet': 's', 'Conformer': '^', 'CSP+LDA': 'x'}

perm_test = {
    'ATCNet':        [1, 1, 1, 1, 1, 0],
    'EEGNet':        [1, 1, 1, 1, 1, 0],
    'ShallowConvNet':[1, 1, 1, 1, 1, 1],
    'Conformer':     [1, 1, 1, 1, 1, 1],
    'CSP+LDA':       [1, 1, 1, 1, 1, 0],
}

# --- FIGURA 1: Hero chart ---
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

for name, accs in models.items():
    ax.plot(ch_counts, accs, marker=markers[name], linewidth=2.5, markersize=10,
            color=colors[name], label=name, zorder=3)

ax.axhline(y=0.5, color='#ff4444', linestyle='--', alpha=0.5, linewidth=1.5, label='Chance (50%)')

# Highlight zones
ax.axvspan(1.5, 4.5, alpha=0.08, color='#ff9800', zorder=0)
ax.annotate('Wearable\nzone', xy=(3, 0.46), fontsize=9, color='#ff9800',
            ha='center', style='italic')

ax.axvspan(4.5, 20, alpha=0.08, color='#4CAF50', zorder=0)
ax.annotate('Portable\nheadband', xy=(10, 0.46), fontsize=9, color='#4CAF50',
            ha='center', style='italic')

ax.set_xlabel('Number of EEG Channels', fontsize=14, color='white', labelpad=10)
ax.set_ylabel('Accuracy', fontsize=14, color='white', labelpad=10)
ax.set_title('Motor Imagery Classification: Electrode Reduction Experiment',
             fontsize=16, color='white', fontweight='bold', pad=15)
ax.set_xticks(ch_counts)
ax.set_xticklabels([f'{c}ch' for c in ch_counts], fontsize=12, color='white')
ax.set_ylim(0.44, 0.78)
ax.tick_params(colors='white')
ax.grid(True, alpha=0.15, color='white')
for spine in ax.spines.values():
    spine.set_color('#30363d')

legend = ax.legend(loc='upper left', fontsize=11, framealpha=0.9,
                   facecolor='#161b22', edgecolor='#30363d', labelcolor='white')

# Annotations
ax.annotate('ATCNet best\n(74.2%)', xy=(64, 0.742), xytext=(55, 0.76),
            fontsize=9, color='#E91E63', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#E91E63', lw=1.5))
ax.annotate('ShallowConvNet best\n(56.3%)', xy=(2, 0.563), xytext=(5, 0.60),
            fontsize=9, color='#4CAF50', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.5))

plt.tight_layout()
plt.savefig('figures/accuracy_vs_channels.png', dpi=200, bbox_inches='tight',
            facecolor='#0d1117', edgecolor='none')
plt.close()
print("1/3 accuracy_vs_channels.png")

# --- FIGURA 2: Permutation test heatmap ---
fig, ax = plt.subplots(figsize=(10, 4))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

model_names = list(perm_test.keys())
grid = np.array([perm_test[m] for m in model_names])

for i in range(len(model_names)):
    for j in range(len(ch_counts)):
        color = '#1a472a' if grid[i, j] == 1 else '#5c1a1a'
        border = '#2ea043' if grid[i, j] == 1 else '#da3633'
        rect = plt.Rectangle((j - 0.45, i - 0.4), 0.9, 0.8,
                              facecolor=color, edgecolor=border, linewidth=1.5)
        ax.add_patch(rect)
        text = 'PASS' if grid[i, j] == 1 else 'FAIL'
        text_color = '#2ea043' if grid[i, j] == 1 else '#da3633'
        ax.text(j, i, text, ha='center', va='center', fontweight='bold',
                fontsize=11, color=text_color)

ax.set_xticks(range(len(ch_counts)))
ax.set_xticklabels([f'{c}ch' for c in ch_counts], fontsize=12, color='white')
ax.set_yticks(range(len(model_names)))
ax.set_yticklabels(model_names, fontsize=12, color='white')
ax.set_xlim(-0.5, len(ch_counts) - 0.5)
ax.set_ylim(-0.5, len(model_names) - 0.5)
ax.set_title('Permutation Test Validation', fontsize=16, color='white',
             fontweight='bold', pad=15)
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_visible(False)

# Legend
ax.text(len(ch_counts) - 1, -0.9, 'PASS = statistically significant (p < 0.05)',
        fontsize=9, color='#8b949e', ha='right')

plt.tight_layout()
plt.savefig('figures/permutation_test.png', dpi=200, bbox_inches='tight',
            facecolor='#0d1117', edgecolor='none')
plt.close()
print("2/3 permutation_test.png")

# --- FIGURA 3: Key findings summary ---
fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')
ax.axis('off')

findings = [
    ('64ch  Full EEG Cap',     'ATCNet',       '74.2%', '#E91E63', 'Attention mechanism captures\ntemporal dynamics best'),
    ('8-16ch  Portable',       'ShallowConvNet','63.5-64.7%', '#4CAF50', 'Best accuracy-robustness\nbalance'),
    ('4ch  Minimal (C3,Cz,C4,CPz)', 'ShallowConvNet', '59.9%', '#4CAF50', 'All 5 models pass\npermutation test'),
    ('2ch  Wearable (C3,C4)',  'ShallowConvNet','56.3%', '#4CAF50', 'Only 2/5 models pass\npermutation test'),
]

y_positions = [0.82, 0.58, 0.34, 0.10]

ax.text(0.5, 0.98, 'Recommended Model by Hardware Target',
        fontsize=18, color='white', fontweight='bold',
        ha='center', va='top', transform=ax.transAxes)

for (config, model, acc, color, note), y in zip(findings, y_positions):
    # Background box
    rect = plt.Rectangle((0.02, y - 0.07), 0.96, 0.18,
                          facecolor='#161b22', edgecolor='#30363d',
                          linewidth=1, transform=ax.transAxes, clip_on=False)
    ax.add_patch(rect)

    # Left: config
    ax.text(0.04, y + 0.03, config, fontsize=13, color='white',
            fontweight='bold', transform=ax.transAxes, va='center')

    # Center: model + accuracy
    ax.text(0.55, y + 0.05, model, fontsize=14, color=color,
            fontweight='bold', transform=ax.transAxes, va='center')
    ax.text(0.55, y - 0.02, acc, fontsize=12, color='#8b949e',
            transform=ax.transAxes, va='center')

    # Right: note
    ax.text(0.78, y + 0.02, note, fontsize=10, color='#8b949e',
            transform=ax.transAxes, va='center')

plt.tight_layout()
plt.savefig('figures/recommendations.png', dpi=200, bbox_inches='tight',
            facecolor='#0d1117', edgecolor='none')
plt.close()
print("3/3 recommendations.png")

print("\nTodas las figuras generadas en figures/")
