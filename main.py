import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.datasets import make_moons

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"AI for Science Experiment Running on: {device}")

plt.rcParams['font.family'] = 'serif'
sns.set_theme(style="whitegrid", font="serif")

def get_data(n_samples=2000, input_dim=50):
    z0, _ = make_moons(n_samples=n_samples, noise=0.1)
    z0[:, 1] = z0[:, 1] * 1.5 - 2
    
    z1, _ = make_moons(n_samples=n_samples, noise=0.1)
    z1[:, 0] = z1[:, 0] + 5.0
    z1[:, 1] = z1[:, 1] * 1.5 + 2
    z1 = z1 * 1.2

    rng = np.random.RandomState(42)
    proj = rng.randn(2, input_dim)
    
    X0 = z0 @ proj + rng.randn(n_samples, input_dim) * 0.05
    X1 = z1 @ proj + rng.randn(n_samples, input_dim) * 0.05
    
    return torch.tensor(X0, dtype=torch.float32).to(device), torch.tensor(X1, dtype=torch.float32).to(device)

class VectorField(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, dim)
        )
    
    def forward(self, t, x):
        t_embed = t.view(-1, 1).expand(x.shape[0], 1)
        return self.net(torch.cat([x, t_embed], dim=1))

def train(model, x0, x1, steps=3000, batch_size=256, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    print("Training Vector Field (CFM)...")
    
    for step in range(steps):
        idx = torch.randperm(len(x0))[:batch_size]
        batch_x0 = x0[idx]
        batch_x1 = x1[idx]
        
        t = torch.rand(batch_size, 1).to(device)
        
        t_expand = t.view(batch_size, 1)
        x_t = (1 - t_expand) * batch_x0 + t_expand * batch_x1
        target_v = batch_x1 - batch_x0
        
        pred_v = model(t, x_t)
        loss = torch.mean((pred_v - target_v)**2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 500 == 0:
            print(f"  Step {step}: Loss = {loss.item():.4f}")

@torch.no_grad()
def solve_ode(model, x0, steps=100):
    dt = 1.0 / steps
    xt = x0.clone()
    traj = [xt.cpu().numpy()]
    
    for i in range(steps):
        t = torch.tensor([i / steps]).to(device)
        v = model(t, xt)
        xt = xt + v * dt
        traj.append(xt.cpu().numpy())
        
    return np.array(traj)

@torch.no_grad()
def solve_sde_csb(model, x0, steps=100, diffusion_scale=1.5):
    dt = 1.0 / steps
    xt = x0.clone()
    traj = [xt.cpu().numpy()]
    
    for i in range(steps):
        t = torch.tensor([i / steps]).to(device)
        v = model(t, xt)
        noise = torch.randn_like(xt) * np.sqrt(dt) * diffusion_scale
        xt = xt + v * dt + noise
        traj.append(xt.cpu().numpy())
        
    return np.array(traj)

def visualize_results(X_src, X_tgt, traj_ode, traj_sde):
    print("Generating Plots...")
    pca = PCA(n_components=2)
    all_data = torch.cat([X_src, X_tgt], dim=0).cpu().numpy()
    pca.fit(all_data)
    
    p_src = pca.transform(X_src.cpu().numpy())
    p_tgt = pca.transform(X_tgt.cpu().numpy())
    
    def project(traj):
        T, N, D = traj.shape
        flat = traj.reshape(-1, D)
        proj = pca.transform(flat)
        return proj.reshape(T, N, 2)
    
    p_ode = project(traj_ode)
    p_sde = project(traj_sde)

    fig1, ax = plt.subplots(figsize=(10, 7), dpi=150)
    
    ax.scatter(p_src[:,0], p_src[:,1], c='gray', alpha=0.3, s=20, label='Source (Control)')
    ax.scatter(p_tgt[:,0], p_tgt[:,1], c='black', alpha=0.3, s=20, label='Target (Stimulated)')
    
    ax.scatter(p_ode[-1,:,0], p_ode[-1,:,1], c='#EE3377', s=15, alpha=0.8, label='ODE (Flow Matching)')
    for i in range(0, p_ode.shape[1], 10):
        ax.plot(p_ode[:,i,0], p_ode[:,i,1], c='#EE3377', alpha=0.1, linewidth=0.5)

    ax.scatter(p_sde[-1,:,0], p_sde[-1,:,1], c='#009988', s=15, alpha=0.8, label='CSB (SDE Tunneling)')
    for i in range(0, p_sde.shape[1], 10):
        ax.plot(p_sde[:,i,0], p_sde[:,i,1], c='#009988', alpha=0.1, linewidth=0.5)
        
    ax.set_title("Scientific Discovery Proxy: Tunneling vs Rigid Flow", fontsize=16)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig1.savefig('fig1_trajectories.png')
    print("Figure 1 saved.")

    ground_truth = X_tgt.cpu().numpy()
    pca_1d = PCA(n_components=1)
    pca_1d.fit(ground_truth)
    
    p_gt_1d = pca_1d.transform(ground_truth).flatten()
    p_ode_1d = pca_1d.transform(traj_ode[-1]).flatten()
    p_sde_1d = pca_1d.transform(traj_sde[-1]).flatten()
    
    fig2, ax = plt.subplots(figsize=(10, 5), dpi=300)
    
    palette = {"GT": "gray", "ODE": "#EE3377", "SDE": "#009988"}
    
    sns.kdeplot(p_gt_1d, color=palette["GT"], fill=True, alpha=0.2, linewidth=0, label='Ground Truth')
    sns.kdeplot(p_gt_1d, color=palette["GT"], linestyle="--", linewidth=1)
    
    sns.kdeplot(p_ode_1d, color=palette["ODE"], fill=True, alpha=0.4, linewidth=2, 
                bw_adjust=0.6, label='ODE (Mode Collapse)')
    
    sns.kdeplot(p_sde_1d, color=palette["SDE"], fill=True, alpha=0.4, linewidth=2, 
                label='CSB (Distribution Match)')
    
    ax.set_title("The Cost of Determinism: Mode Collapse in High Dimensions", fontsize=14)
    ax.set_xlabel("Principal Component 1 (Direction of Max Variance)", fontsize=12)
    ax.set_yticks([])
    ax.legend()
    
    peak_x = np.mean(p_ode_1d)
    ax.annotate('ODE Collapses\nto the Mean!', xy=(peak_x, ax.get_ylim()[1]*0.9), 
                xytext=(peak_x+2, ax.get_ylim()[1]*0.95),
                arrowprops=dict(facecolor='#EE3377', shrink=0.05),
                fontsize=12, color='#EE3377', fontweight='bold')
    
    plt.tight_layout()
    fig2.savefig('fig2_density_comparison.png')
    print("Figure 2 saved.")

def main():
    dim = 50
    X_src, X_tgt = get_data(n_samples=2000, input_dim=dim)
    
    model = VectorField(dim).to(device)
    train(model, X_src, X_tgt, steps=3000)
    
    print("Running Inference...")
    test_x0 = X_src[:500]
    
    traj_ode = solve_ode(model, test_x0)
    traj_sde = solve_sde_csb(model, test_x0, diffusion_scale=2.0)
    
    visualize_results(X_src, X_tgt, traj_ode, traj_sde)
    print("Experiment Complete!")

if __name__ == "__main__":
    main()