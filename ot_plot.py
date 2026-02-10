import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

sns.set_style("white")

def create_lightcone_plot():
    fig, ax = plt.subplots(figsize=(8, 8))

    t_start = 0
    t_end = 1
    x_start = 0
    
    c = 1.5 
    
    cone_x = [x_start, x_start - c*t_end, x_start + c*t_end]
    cone_t = [t_start, t_end, t_end]
    cone_polygon = patches.Polygon(xy=list(zip(cone_x, cone_t)), closed=True, 
                                    facecolor='#E0F2F7', alpha=0.6, edgecolor='#009E73', ls='--')
    ax.add_patch(cone_polygon)
    
    ax.text(0, 0.6, 'Allowed Causal Future\n(The Light Cone)', ha='center', fontsize=12, color='#006644')

    start_node = (x_start, t_start)
    target_node = (1.0, t_end)

    t_causal = np.linspace(t_start, t_end, 100)
    x_causal = np.linspace(x_start, target_node[0], 100) + 0.2 * np.sin(np.pi * t_causal)
    
    ax.plot(x_causal, t_causal, color='#009E73', lw=3, label='CSB (Causal)\nRespects Arrow of Time')

    t_acausal = np.linspace(t_start, t_end, 100)
    x_acausal = np.linspace(x_start, target_node[0], 100) - 0.8 * np.sin(np.pi * t_acausal)
    
    ax.plot(x_acausal, t_acausal, color='#D55E00', lw=3, ls='--', label='Standard OT (Acausal)\nAnticipatory Control')
    
    cone_left_boundary = x_start - c * t_acausal
    violation_mask = x_acausal < cone_left_boundary
    
    ax.fill_between(x_acausal, t_acausal, cone_left_boundary, where=violation_mask, 
                    color='#D55E00', alpha=0.3, hatch='//')
    
    ax.annotate('Violation of\nCausality!', xy=(-0.55, 0.3), xytext=(-1.2, 0.4),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                fontsize=10, color='#D55E00', fontweight='bold')

    ax.scatter(*start_node, color='black', s=150, zorder=10)
    ax.text(start_node[0], start_node[1]-0.05, r'Cause $X_0 (t=0)$', ha='center', fontweight='bold')
    
    ax.scatter(*target_node, color='black', s=150, zorder=10)
    ax.text(target_node[0], target_node[1]+0.03, r'Effect $X_1 (t=1)$', ha='center', fontweight='bold')

    ax.set_xlim(-1.5, 1.8)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel('State Space $X$', fontsize=12)
    ax.set_ylabel('Time $t$ (The Arrow of Time)', fontsize=12)
    ax.set_title("The Causal Light Cone vs. Anticipatory Control", fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right')
    
    ax.arrow(-1.4, 0, 0, 1.05, lw=1.5, head_width=0.05, color='gray', zorder=0)

    sns.despine()
    plt.tight_layout()
    plt.savefig('concept_lightcone.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    create_lightcone_plot()