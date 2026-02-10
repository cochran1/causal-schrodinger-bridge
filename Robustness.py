import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Robustness Experiment Running on: {device}")
plt.switch_backend('Agg') 

class SCM_Fork:
    def sample(self, n=3000):
        X = torch.randn(n, 1).to(device)
        Y = 2 * X + torch.randn(n, 1).to(device) * 0.3
        Z = 2 * X + torch.randn(n, 1).to(device) * 0.3
        return X, Y, Z

class ConditionalVectorField(nn.Module):
    def __init__(self, dim_x, dim_context):
        super().__init__()
        input_dim = dim_x + 1 + dim_context
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, dim_x)
        ).to(device)
    
    def forward(self, t, x, context=None):
        if context is not None:
            inp = torch.cat([x, t, context], dim=1)
        else:
            inp = torch.cat([x, t], dim=1)
        return self.net(inp)

@torch.no_grad()
def ode_solve(model, x0, context=None, steps=50, reverse=False):
    dt = 1.0 / steps
    if reverse: dt = -dt; t_grid = torch.linspace(1.0, 0.0, steps+1).to(device)
    else: t_grid = torch.linspace(0.0, 1.0, steps+1).to(device)
    
    x = x0
    for i in range(steps):
        t = t_grid[i].view(1, 1).repeat(x.shape[0], 1)
        v = model(t, x, context)
        x = x + v * dt
    return x

def train_cfm(model, target, context=None, steps=1000, name="Model"):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(steps):
        t = torch.rand(target.shape[0], 1).to(device)
        x0 = torch.randn_like(target).to(device)
        x1 = target
        x_t = (1 - t) * x0 + t * x1
        target_v = x1 - x0
        pred_v = model(t, x_t, context)
        loss = torch.mean((pred_v - target_v)**2)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

def main():
    torch.manual_seed(42)
    
    scm = SCM_Fork()
    X_dat, Y_dat, Z_dat = scm.sample(2000)
    
    print("Training Correct Graph Models...")
    model_corr_X = ConditionalVectorField(1, 0)
    train_cfm(model_corr_X, X_dat, None, name="Correct P(X)")
    
    model_Z_given_X = ConditionalVectorField(1, 1)
    train_cfm(model_Z_given_X, Z_dat, context=X_dat, name="P(Z|X)")
    
    print("Training Wrong Graph Models (Reversed Arrow)...")
    model_wrong_Y = ConditionalVectorField(1, 0) 
    train_cfm(model_wrong_Y, Y_dat, None, name="Wrong P(Y)")
    
    model_wrong_X_given_Y = ConditionalVectorField(1, 1)
    train_cfm(model_wrong_X_given_Y, X_dat, context=Y_dat, name="Wrong P(X|Y)")
    
    print("\nPerforming Intervention do(Y=3)...")
    
    idx = 10
    x_obs = X_dat[idx:idx+1]
    y_obs = Y_dat[idx:idx+1]
    z_obs = Z_dat[idx:idx+1]
    
    y_intervened = torch.tensor([[3.0]]).to(device)
    
    u_x_corr = ode_solve(model_corr_X, x_obs, None, reverse=True)
    u_z_corr = ode_solve(model_Z_given_X, z_obs, context=x_obs, reverse=True)
    
    x_cf_corr = ode_solve(model_corr_X, u_x_corr, None, reverse=False) 
    
    z_cf_corr = ode_solve(model_Z_given_X, u_z_corr, context=x_cf_corr, reverse=False)
    
    u_x_wrong = ode_solve(model_wrong_X_given_Y, x_obs, context=y_obs, reverse=True)
    u_z_wrong = ode_solve(model_Z_given_X, z_obs, context=x_obs, reverse=True)
    
    x_cf_wrong = ode_solve(model_wrong_X_given_Y, u_x_wrong, context=y_intervened, reverse=False)
    
    z_cf_wrong = ode_solve(model_Z_given_X, u_z_wrong, context=x_cf_wrong, reverse=False)
    
    err_corr = torch.abs(z_cf_corr - z_obs).item()
    err_wrong = torch.abs(z_cf_wrong - z_obs).item()
    
    print(f"Observed Z: {z_obs.item():.2f}")
    print(f"Correct Graph Z*: {z_cf_corr.item():.2f} (Error: {err_corr:.4f})")
    print(f"Wrong Graph Z*:   {z_cf_wrong.item():.2f} (Error: {err_wrong:.4f})")
    
    plt.figure(figsize=(6, 5))
    methods = ['Correct Graph\n(CSB)', 'Wrong Graph\n(Misspecified)']
    errors = [err_corr, err_wrong]
    colors = ['#009E73', '#D55E00']
    
    bars = plt.bar(methods, errors, color=colors, alpha=0.8, width=0.5)
    
    plt.ylabel('Counterfactual Error on Z ($|\Delta Z|$)')
    plt.title('Robustness Analysis: Effect of Graph Misspecification')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f"{yval:.2f}", ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('robustness_analysis.png')
    print("Saved robustness_analysis.png")

if __name__ == "__main__":
    main()