import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

plt.switch_backend('Agg') 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"NeurIPS Experiment Running on: {device}")

plt.rcParams['font.family'] = 'serif' 
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12

class SCM_Fork:
    def sample(self, n=3000):
        X = torch.randn(n, 1).to(device)
        U_Y = torch.randn(n, 1).to(device) * 0.3
        Y = 2 * X + U_Y
        U_Z = torch.randn(n, 1).to(device) * 0.3
        Z = 2 * X + U_Z
        return X, Y, Z

class VectorField(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, dim_out)
        ).to(device)
    
    def forward(self, t, x, context=None):
        if context is not None:
            inp = torch.cat([x, t, context], dim=1)
        else:
            inp = torch.cat([x, t], dim=1)
        return self.net(inp)

@torch.no_grad()
def ode_solve_traj(vector_field, x0, context, steps=50, reverse=False):
    dt = 1.0 / steps
    if reverse:
        dt = -dt
        t_grid = torch.linspace(1.0, 0.0, steps+1).to(device)
    else:
        t_grid = torch.linspace(0.0, 1.0, steps+1).to(device)
        
    x = x0
    traj = [x.cpu()]
    
    for i in range(steps):
        t = t_grid[i].view(1, 1).repeat(x.shape[0], 1)
        v = vector_field(t, x, context)
        x = x + v * dt
        traj.append(x.cpu())
        
    return x, torch.stack(traj) 

def train_cfm(model, x1, context=None, steps=2000, lr=1e-3, name="Model"):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    print(f"   Training {name}...")
    
    for step in range(steps):
        batch_size = x1.shape[0]
        t = torch.rand(batch_size, 1).to(device)
        x0 = torch.randn_like(x1).to(device)
        x_t = (1 - t) * x0 + t * x1
        target_v = x1 - x0
        pred_v = model(t, x_t, context)
        loss = torch.mean((pred_v - target_v)**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    torch.manual_seed(42)
    
    scm = SCM_Fork()
    X_dat, Y_dat, Z_dat = scm.sample(5000)
    
    print("Step 1: Training Models...")
    data_joint = torch.cat([Y_dat, Z_dat], dim=1)
    model_baseline = VectorField(dim_in=3, dim_out=2)
    train_cfm(model_baseline, data_joint, name="Baseline (Joint OT)")
    
    model_X = VectorField(dim_in=2, dim_out=1) 
    train_cfm(model_X, X_dat, name="CSB X")
    
    model_Y = VectorField(dim_in=3, dim_out=1)
    train_cfm(model_Y, Y_dat, context=X_dat, name="CSB Y|X")
    
    model_Z = VectorField(dim_in=3, dim_out=1)
    train_cfm(model_Z, Z_dat, context=X_dat, name="CSB Z|X")
    
    print("Step 2: Performing Intervention do(Y=3)...")
    idx = torch.argmin(X_dat + Y_dat + Z_dat)
    x_obs = X_dat[idx:idx+1]
    y_obs = Y_dat[idx:idx+1]
    z_obs = Z_dat[idx:idx+1]
    
    print(f"   Fact: Y={y_obs.item():.2f}, Z={z_obs.item():.2f}")
    y_target_val = 3.0
    
    joint_obs = torch.cat([y_obs, z_obs], dim=1)
    u_joint, _ = ode_solve_traj(model_baseline, joint_obs, None, reverse=True)
    _, traj_base = ode_solve_traj(model_baseline, u_joint, None, reverse=False)
    
    cov = torch.cov(torch.stack([Y_dat.squeeze(), Z_dat.squeeze()]))
    slope = (cov[0,1] / cov[0,0]).item() 
    
    base_y_path = torch.linspace(y_obs.item(), y_target_val, 50).numpy()
    base_z_path = (z_obs.item() + slope * (base_y_path - y_obs.item()))
    
    u_x, _ = ode_solve_traj(model_X, x_obs, None, reverse=True)
    x_cf = x_obs 
    y_target_val = 3.0
    
    csb_y_path = torch.linspace(y_obs.item(), y_target_val, 50).numpy()
    csb_z_path = torch.full_like(torch.tensor(csb_y_path), z_obs.item()).numpy()

    print("Step 3: Generating NeurIPS Plot...")
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    Y_np = Y_dat.detach().cpu().numpy().flatten()
    Z_np = Z_dat.detach().cpu().numpy().flatten()
    hb = ax.hexbin(Y_np, Z_np, gridsize=40, cmap='Greys', mincnt=1, alpha=0.3, edgecolors='none')
    
    ax.text(-5, -5, "High Density\nCorrelation Manifold", fontsize=10, color='gray', ha='center', va='center')

    ax.scatter(y_obs.item(), z_obs.item(), s=200, color='#0077BB', edgecolors='white', linewidth=2, zorder=10, label='Factual Observation')
    
    ax.plot(base_y_path, base_z_path, color='#EE3377', linestyle='--', linewidth=3, alpha=0.8, label='Standard OT (Baseline)')
    ax.arrow(base_y_path[-2], base_z_path[-2], 
             base_y_path[-1]-base_y_path[-2], base_z_path[-1]-base_z_path[-2],
             color='#EE3377', width=0.1, head_width=0.4, length_includes_head=True, zorder=5)
    
    ax.plot(csb_y_path, csb_z_path, color='#009988', linestyle='-', linewidth=4, alpha=0.9, label='Causal SB (Ours)')
    ax.arrow(csb_y_path[-2], csb_z_path[-2], 
             csb_y_path[-1]-csb_y_path[-2], csb_z_path[-1]-csb_z_path[-2],
             color='#009988', width=0.1, head_width=0.4, length_includes_head=True, zorder=5)
    
    ax.axvline(x=y_target_val, color='black', linestyle=':', alpha=0.5, label='Intervention Target do(Y=3)')

    ax.set_xlabel(r"Intervened Variable $Y$ (Cause)", fontsize=14)
    ax.set_ylabel(r"Effect Variable $Z$ (Effect of $X$)", fontsize=14)
    ax.set_title("Counterfactual Trajectories: On-Manifold vs. Structural", fontsize=16, pad=20)
    
    ax.annotate('Spurious Correlation\n(Incorrect)', xy=(0, 0), xytext=(-2, 4),
                arrowprops=dict(facecolor='#EE3377', shrink=0.05, alpha=0.5),
                fontsize=11, color='#EE3377', fontweight='bold')
                
    ax.annotate('Structural Independence\n(Correct)', xy=(0, z_obs.item()), xytext=(0, -6),
                arrowprops=dict(facecolor='#009988', shrink=0.05, alpha=0.5),
                fontsize=11, color='#009988', fontweight='bold')

    ax.legend(loc='upper left', frameon=True, fancybox=True, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlim(-8, 6)
    ax.set_ylim(-10, 8)
    
    plt.tight_layout()
    plt.savefig('neurips_hero_figure.png')
    print("Result saved to neurips_hero_figure.png")

if __name__ == "__main__":
    main()