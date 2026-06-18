---
name: data-visualization
description: Use for creating publication-quality charts and multi-panel analysis summaries. Triggers when tasks involve visualizing data, plotting results, creating charts, or producing visual reports.
---

# Data Visualization Skill

Create publication-quality analytical charts using matplotlib and seaborn.

## When to Use This Skill

Use this skill when:
- Visualizing results from data analysis or ML models
- Creating charts (bar, line, scatter, heatmap, histogram, box plot)
- Building multi-panel analysis summaries
- The user asks for visual output, plots, graphs, or charts

## Initialization (REQUIRED)

```python
import matplotlib
matplotlib.use('Agg')  # Headless backend — MUST be before pyplot import
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.constrained_layout.use': True,
})

# Colorblind-safe palette (Okabe-Ito)
COLORS = ['#0173B2', '#DE8F05', '#029E73', '#D55E00', '#CC78BC',
          '#CA9161', '#FBAFE4', '#949494', '#ECE133', '#56B4E9']
```

## Saving Charts

Always save with these settings:

```python
plt.savefig('generated_plots/chart_name.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
```

- `dpi=300` for print quality
- `bbox_inches='tight'` removes excess whitespace
- `facecolor='white'` ensures white background
- Always call `plt.close()` after saving to free memory

## Quick Reference

### Bar Chart
```python
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(result.index, result.values, color=COLORS[:len(result)],
              edgecolor='black', linewidth=0.8)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}', ha='center', va='bottom', fontsize=9)
ax.set_ylabel('Mean Value', fontweight='bold')
ax.set_title('Average Value by Category', fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.savefig('generated_plots/bar_chart.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
```

### Line Chart
```python
fig, ax = plt.subplots(figsize=(10, 5))
for i, col in enumerate(columns_to_plot):
    ax.plot(df["date"], df[col], label=col, color=COLORS[i], linewidth=2)
ax.legend(frameon=True)
ax.grid(True, alpha=0.3, linestyle='--')
plt.savefig('generated_plots/line_chart.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
```

### Scatter Plot
```python
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(df["x"], df["y"], c=df["color_var"], cmap='viridis',
                     alpha=0.7, edgecolors='black', linewidth=0.5, s=40)
plt.colorbar(scatter, ax=ax, label='Color Variable')
plt.savefig('generated_plots/scatter.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
```

### Heatmap (Correlation)
```python
import seaborn as sns
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True, ax=ax)
ax.set_title('Correlation Matrix', fontweight='bold')
plt.savefig('generated_plots/heatmap.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
```

### Multi-Panel Dashboard
```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# Panel 1: Distribution
axes[0, 0].hist(df["value"], bins=30, color=COLORS[0], edgecolor='black')
axes[0, 0].set_title('Distribution')
# Panel 2: Time series
axes[0, 1].plot(df["date"], df["metric"], color=COLORS[1])
axes[0, 1].set_title('Trend')
# Panel 3: Comparison
axes[1, 0].bar(categories, values, color=COLORS[:len(categories)])
axes[1, 0].set_title('Comparison')
# Panel 4: Scatter
axes[1, 1].scatter(df["x"], df["y"], alpha=0.5, color=COLORS[3])
axes[1, 1].set_title('Relationship')
fig.suptitle('Analysis Dashboard', fontsize=16, fontweight='bold')
plt.savefig('generated_plots/dashboard.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
```

## Output Guidelines

- Use descriptive filenames: `revenue_by_region.png`, not `chart1.png`
- Include axis labels with units
- Add titles that describe the insight, not just the data
- Use consistent color scheme across related charts
