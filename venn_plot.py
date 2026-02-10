import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

sns.set_style("white")

def create_venn_diagram():
    fig, ax = plt.subplots(figsize=(8, 6))

    path_space = patches.Ellipse((0.5, 0.5), width=0.9, height=0.8, angle=0, 
                                  facecolor='#F0F0F0', edgecolor='gray', ls=':', lw=2, alpha=0.5)
    ax.add_patch(path_space)
    ax.text(0.5, 0.85, r'Path Space $\Omega$', ha='center', fontsize=12, color='gray')

    stat_manifold = patches.Ellipse((0.4, 0.5), width=0.6, height=0.4, angle=30, 
                                    facecolor='#FFEEEE', edgecolor='#FF3333', lw=2, alpha=0.6)
    ax.add_patch(stat_manifold)
    ax.text(0.2, 0.65, r'Statistical Manifold $\mathcal{M}_{stat}$' '\n(Correlations)', 
            ha='center', fontsize=10, color='#FF3333', rotation=30)

    struct_manifold = patches.Ellipse((0.6, 0.5), width=0.6, height=0.4, angle=-30, 
                                      facecolor='#EEFFEE', edgecolor='#00CC66', lw=2, alpha=0.6)
    ax.add_patch(struct_manifold)
    ax.text(0.8, 0.65, r'Structural Manifold $\mathcal{M}_{struct}$' '\n(Causal Laws)', 
            ha='center', fontsize=10, color='#00CC66', rotation=-30)
            
    ax.scatter(0.5, 0.5, color='black', s=100, marker='*')
    ax.text(0.5, 0.45, 'Observational\nData', ha='center', va='top', fontsize=9, fontweight='bold')

    ax.scatter(0.25, 0.5, color='#FF3333', s=150, marker='X')
    ax.annotate('Standard Generative Models\n(Adhere to $\mathcal{M}_{stat}$, violate causality)', 
                xy=(0.25, 0.5), xytext=(0.05, 0.2),
                arrowprops=dict(facecolor='#FF3333', shrink=0.05, width=1, headwidth=6),
                fontsize=10, color='#FF3333', ha='left')

    ax.scatter(0.65, 0.4, color='#00CC66', s=150, marker='P')
    ax.annotate('Causal Schrödinger Bridge\n(Optimal transport on $\mathcal{M}_{struct}$)', 
                xy=(0.65, 0.4), xytext=(0.55, 0.15),
                arrowprops=dict(facecolor='#00CC66', shrink=0.05, width=1, headwidth=6),
                fontsize=10, color='#00CC66', ha='left', fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("A Geometric Unification of Generative Models", fontsize=14, fontweight='bold', pad=15)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig('concept_geometry_venn.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    create_venn_diagram()