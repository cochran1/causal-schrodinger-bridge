import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.linewidth'] = 1.5

def create_tunneling_plot_v2():
    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.linspace(-3, 7, 500)
    y = np.linspace(-3, 3, 500)
    X, Y = np.meshgrid(x, y)

    Z0 = 1.2 * np.exp(-((X - 0)**2 + (Y - 0)**2) / (2 * 0.7**2))
    Z1 = 1.0 * np.exp(-((X - 4)**2 + (Y - 0)**2) / (2 * 0.6**2))
    Z = Z0 + Z1 + 0.02

    cmap = plt.cm.Blues 
    
    contour = ax.contourf(X, Y, Z, levels=25, cmap=cmap, alpha=0.8)
    ax.contour(X, Y, Z, levels=15, colors='k', linewidths=0.3, alpha=0.2)

    start_point = np.array([0.0, 0.0])
    end_point = np.array([4.0, 0.0])

    t = np.linspace(0, np.pi, 100)
    ode_x = np.linspace(0, 4, 100)
    ode_y = 1.8 * np.sin(t) 
    
    ax.plot(ode_x, ode_y, color='#D55E00', lw=3.5, ls='--', 
            label='ODE Flow (Standard)\nStuck or Forced Detour', zorder=5)
    
    ax.arrow(ode_x[50], ode_y[50], ode_x[51]-ode_x[50], ode_y[51]-ode_y[50], 
             shape='full', lw=0, length_includes_head=True, head_width=0.15, color='#D55E00', zorder=5)

    np.random.seed(42)
    t_sde = np.linspace(0, 1, 300)
    base_x = (1-t_sde) * start_point[0] + t_sde * end_point[0]
    base_y = (1-t_sde) * start_point[1] + t_sde * end_point[1]
    
    noise_x = np.cumsum(np.random.randn(300)) * np.sqrt(1/300)
    noise_y = np.cumsum(np.random.randn(300)) * np.sqrt(1/300)
    noise_x -= t_sde * noise_x[-1]
    noise_y -= t_sde * noise_y[-1]
    
    diffusion_scale = 0.6
    sde_x = base_x + diffusion_scale * noise_x
    sde_y = base_y + diffusion_scale * noise_y + 0.2 * np.sin(np.pi * t_sde)

    ax.plot(sde_x, sde_y, color='white', lw=4.5, alpha=1.0, zorder=6) 
    ax.plot(sde_x, sde_y, color='#009E73', lw=3, label='Causal SB (Ours)\nTunneling via Diffusion', zorder=7) 

    ax.scatter(*start_point, color='#0072B2', s=300, edgecolors='white', linewidth=2, zorder=10)
    ax.scatter(*end_point, color='#0072B2', s=300, edgecolors='white', linewidth=2, zorder=10)
    
    ax.text(start_point[0], start_point[1]-0.6, r'Source $\mu_0$', ha='center', fontsize=12, fontweight='bold', color='#333333')
    ax.text(end_point[0], end_point[1]-0.6, r'Target $\mu_1$', ha='center', fontsize=12, fontweight='bold', color='#333333')

    ax.text(2.0, 0.2, 'Low-Density Region\n(Support Mismatch)', ha='center', va='center', 
            fontsize=10, color='gray', style='italic', backgroundcolor='white', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    ax.set_title("Why SDE? Robust Transport Across Disjoint Supports", fontsize=14, fontweight='bold', pad=15)
    
    legend = ax.legend(loc='upper left', frameon=True, fontsize=10, framealpha=0.9, edgecolor='gray')
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig('concept_tunneling_v2.png', dpi=300, bbox_inches='tight')
    print("New plot saved: concept_tunneling_v2.png")
    plt.show()

if __name__ == "__main__":
    create_tunneling_plot_v2()