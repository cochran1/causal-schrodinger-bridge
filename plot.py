import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from scipy.stats import gaussian_kde

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.linewidth'] = 1.5

def draw_pro_concept_fixed():
    fig, ax = plt.subplots(figsize=(7, 6))
    
    np.random.seed(42)
    x_bg = np.random.normal(3, 1.5, 5000)
    y_bg = x_bg + np.random.normal(0, 0.8, 5000)
    
    xx, yy = np.mgrid[-1:8:100j, -2:9:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x_bg, y_bg])
    kernel = gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    
    ax.contourf(xx, yy, f, levels=8, cmap='Greys', alpha=0.4)
    ax.contour(xx, yy, f, levels=8, colors='gray', linewidths=0.5, alpha=0.3)

    start_pos = (0.5, 0.5)
    ot_end_pos = (5.5, 5.5)
    csb_end_pos = (5.5, 0.5)

    arrow_ot = FancyArrowPatch(start_pos, ot_end_pos,
                               connectionstyle="arc3,rad=-0.15", 
                               color='#D55E00',
                               arrowstyle='Simple,tail_width=2,head_width=10,head_length=10',
                               linewidth=2, linestyle='--',
                               zorder=10, alpha=0.9)
    ax.add_patch(arrow_ot)
    
    arrow_csb = FancyArrowPatch(start_pos, csb_end_pos,
                                color='#009E73',
                                arrowstyle='Simple,tail_width=3,head_width=12,head_length=12',
                                linewidth=0,
                                zorder=10)
    ax.add_patch(arrow_csb)

    ax.scatter(*start_pos, s=300, color='#0072B2', edgecolors='white', linewidth=2.5, zorder=20)
    ax.scatter(*ot_end_pos, s=300, color='#D55E00', marker='X', edgecolors='white', linewidth=2.5, zorder=20)
    ax.scatter(*csb_end_pos, s=350, color='#009E73', marker='*', edgecolors='white', linewidth=2.5, zorder=20)

    ax.text(1.0, 7.0, r"$\mathcal{M}_{data}$: Statistical Manifold" + "\n(Spurious Correlations)", 
            color='#666666', fontsize=11, ha='left', fontweight='light', style='italic')
    ax.annotate("", xy=(2.0, 5.0), xytext=(2.0, 6.8),
                arrowprops=dict(arrowstyle="->", color='#666666', lw=0.8))
    
    ax.text(2.6, 4.2, "Standard OT\n(Manifold Adherence)", 
            color='#D55E00', fontsize=10, fontweight='bold', ha='right', rotation=42, zorder=30)
            
    ax.text(3.0, -0.8, "Causal SB (Ours)\n(Structural Adherence)", 
            color='#009E73', fontsize=11, fontweight='bold', ha='center', zorder=30)

    ax.set_xlabel(r"Intervention Target $do(Y)$", fontsize=14, labelpad=10)
    ax.set_ylabel(r"Effect Variable $Z$", fontsize=14, labelpad=10)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(-1, 7)
    ax.set_ylim(-2, 8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, linestyle=':', alpha=0.3)
    
    ax.set_title(r"$\bf{a.}$ Conceptual: The Geometry of Counterfactuals", loc='left', fontsize=16, pad=20)

    plt.tight_layout()
    plt.savefig('neurips_concept_pro_fixed.png', bbox_inches='tight')
    print("Fixed concept figure saved to neurips_concept_pro_fixed.png")

if __name__ == "__main__":
    draw_pro_concept_fixed()