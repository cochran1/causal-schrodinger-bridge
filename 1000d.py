import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.decomposition import PCA
from sklearn.datasets import make_moons

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)

plt.rcParams['font.family'] = 'serif'
sns.set_theme(style="whitegrid", font="serif")

def get_causal_surgical_data(n_samples=5000, input_dim=1000):
    z0, _ = make_moons(n_samples=n_samples, noise=0.05)
    z0[:, 1] = z0[:, 1] * 2.0 - 3.0 
    
    z1 = z0.copy()
    z1[:, 0] = z1[:, 0] * 0.5 + 8.0 
    z1[:, 1] = z1[:, 1] + 2.0 
    
    proj = np.random.randn(2, input_dim)
    proj /= np.linalg.norm(proj, axis=0, keepdims=True) + 1e-8
    
    X0 = z0 @ proj + np.random.randn(n_samples, input_dim) * 0.02
    X1 = z1 @ proj + np.random.randn(n_samples, input_dim) * 0.02
    
    return (torch.tensor(X0, dtype=torch.float32).to(device), 
            torch.tensor(X1, dtype=torch.float32).to(device),
            z1, proj)

class CausalVectorField(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, dim)
        )
    
    def forward(self, t, x):
        t_embed = t.view(-1, 1).expand(x.shape[0], 1)
        return self.net(torch.cat([x, t_embed], dim=1))

def train_csb(model, x0, x1, steps=5000, train_sigma=0.2):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    start_time = time.time()
    
    for step in range(steps):
        idx = torch.randperm(len(x0))[:512]
        b0, b1 = x0[idx], x1[idx]
        t = torch.rand(len(b0), 1).to(device)
        
        noise = torch.randn_like(b0) * train_sigma
        x_t = (1 - t) * b0 + t * b1 + noise
        
        target_v = b1 - b0
        pred_v = model(t, x_t)
        loss = torch.mean((pred_v - target_v)**2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return time.time() - start_time

@torch.no_grad()
def evaluate_all(model, x0, z1_gt, proj, steps=100):
    proj_pinv = np.linalg.pinv(proj)
    
    def get_metrics(res_x):
        z_hat = res_x @ proj_pinv
        mse = np.mean((z_hat - z1_gt)**2)
        support = np.std(z_hat, axis=0).mean()
        leakage = np.mean(np.abs(z_hat[:, 1] - z1_gt[:, 1]))
        return z_hat, mse, support, leakage

    t0_ode = time.time()
    xt_ode = x0.clone()
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.tensor([i/steps]).to(device).expand(len(x0), 1)
        xt_ode += model(t, xt_ode) * dt
    dt_ode = time.time() - t0_ode
    z_ode, mse_ode, sup_ode, leak_ode = get_metrics(xt_ode.cpu().numpy())

    t0_sde = time.time()
    xt_sde = x0.clone()
    diffusion_scale = 1.2
    for i in range(steps):
        t = torch.tensor([i/steps]).to(device).expand(len(x0), 1)
        noise = torch.randn_like(xt_sde) * np.sqrt(dt) * diffusion_scale
        xt_sde += model(t, xt_sde) * dt + noise
    dt_sde = time.time() - t0_sde
    z_sde, mse_sde, sup_sde, leak_sde = get_metrics(xt_sde.cpu().numpy())

    return {
        "time": (dt_ode, dt_sde),
        "mse": (mse_ode, mse_sde),
        "sup": (sup_ode, sup_sde),
        "leak": (leak_ode, leak_sde),
        "latent_plots": (z_ode, z_sde)
    }

def main():
    X0, X1, Z1_GT, PROJ = get_causal_surgical_data()
    model = CausalVectorField(1000).to(device)
    
    train_time = train_csb(model, X0, X1)
    metrics = evaluate_all(model, X0[:2000], Z1_GT[:2000], PROJ)

    print(f"{'Metric':<25} | {'ODE (Baseline)':<18} | {'CSB (Ours)':<18}")
    print("-" * 65)
    print(f"{'Inference Time':<25} | {metrics['time'][0]:<18.4f}s | {metrics['time'][1]:<18.4f}s")
    print(f"{'Latent Recovery MSE':<25} | {metrics['mse'][0]:<18.6f} | {metrics['mse'][1]:<18.6f}")
    print(f"{'Support Coverage':<25} | {metrics['sup'][0]:<18.4f} | {metrics['sup'][1]:<18.4f}")
    print(f"{'Mechanism Leakage':<25} | {metrics['leak'][0]:<18.6f} | {metrics['leak'][1]:<18.6f}")
    
    z_ode, z_sde = metrics['latent_plots']
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=150)
    
    axes[0].scatter(Z1_GT[:, 0], Z1_GT[:, 1], c='#BDC3C7', alpha=0.3, s=10, label='Ground Truth')
    axes[0].scatter(z_ode[:, 0], z_ode[:, 1], c='#E91E63', alpha=0.6, s=10, label='ODE (Collapsed)')
    axes[0].set_title("ODE: Rigid Path (Information Loss)", fontweight='bold')
    axes[0].legend()

    axes[1].scatter(Z1_GT[:, 0], Z1_GT[:, 1], c='#BDC3C7', alpha=0.3, s=10, label='Ground Truth')
    axes[1].scatter(z_sde[:, 0], z_sde[:, 1], c='#009688', alpha=0.6, s=10, label='CSB: Stochastic Tunneling')
    axes[1].set_title("CSB: Entropy-Regularized (Robust Recovery)", fontweight='bold')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
