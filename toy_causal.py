import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

plt.switch_backend('Agg')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

class SCM:
    def __init__(self):
        pass
    
    def sample(self, n=2000):
        X = torch.randn(n, 1).to(device)
        U_Y = torch.randn(n, 1).to(device) * 0.5
        Y = 2 * X + U_Y
        U_Z = torch.randn(n, 1).to(device) * 0.5
        Z = torch.sin(Y) + X + U_Z
        return X, Y, Z

class VectorField(nn.Module):
    def __init__(self, dim_x, dim_context):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_x + 1 + dim_context, 64),
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

def ode_solve(vector_field, x0, context, steps=100, reverse=False):
    dt = 1.0 / steps
    if reverse:
        dt = -dt
        t_grid = torch.linspace(1.0, 0.0, steps+1).to(device)
    else:
        t_grid = torch.linspace(0.0, 1.0, steps+1).to(device)
        
    x = x0
    traj = [x]
    
    for i in range(steps):
        t = t_grid[i].view(1, 1).repeat(x.shape[0], 1)
        v = vector_field(t, x, context)
        x = x + v * dt
        traj.append(x)
        
    return x, torch.stack(traj)

def train_cfm(model, x1, context, steps=2000, lr=1e-3, name="Model"):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    print(f"Training {name}...")
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
        
    print(f"{name} Loss: {loss.item():.6f}")

def main():
    torch.manual_seed(42)
    
    print("Step 0: Generating Data from SCM...")
    scm = SCM()
    X_dat, Y_dat, Z_dat = scm.sample(2000)
    
    print("\nStep 1: Training Causal Bridges (Sequential Fitting)...")
    
    model_X = VectorField(dim_x=1, dim_context=0)
    train_cfm(model_X, X_dat, context=None, name="Model X")
    
    model_Y = VectorField(dim_x=1, dim_context=1)
    train_cfm(model_Y, Y_dat, context=X_dat, name="Model Y")
    
    context_Z = torch.cat([X_dat, Y_dat], dim=1)
    model_Z = VectorField(dim_x=1, dim_context=2)
    train_cfm(model_Z, Z_dat, context=context_Z, name="Model Z")
    
    print("\nStep 2: Performing Counterfactual Inference...")
    
    idx = 10
    x_obs = X_dat[idx:idx+1]
    y_obs = Y_dat[idx:idx+1]
    z_obs = Z_dat[idx:idx+1]
    
    print(f"[Fact] Observed: X={x_obs.item():.2f}, Y={y_obs.item():.2f}, Z={z_obs.item():.2f}")
    
    x_latent, _ = ode_solve(model_X, x_obs, context=None, reverse=True)
    y_latent, _ = ode_solve(model_Y, y_obs, context=x_obs, reverse=True)
    context_z_obs = torch.cat([x_obs, y_obs], dim=1)
    z_latent, _ = ode_solve(model_Z, z_obs, context=context_z_obs, reverse=True)
    
    print(f"[Abduction] Latents (t=0): u_x={x_latent.item():.2f}, u_y={y_latent.item():.2f}, u_z={z_latent.item():.2f}")
    
    x_intervened_val = 3.0
    x_cf = torch.tensor([[x_intervened_val]]).to(device)
    print(f"[Action] Intervention: do(X = {x_intervened_val})")
    
    y_cf, _ = ode_solve(model_Y, y_latent, context=x_cf, reverse=False)
    context_z_cf = torch.cat([x_cf, y_cf], dim=1)
    z_cf, _ = ode_solve(model_Z, z_latent, context=context_z_cf, reverse=False)
    
    print(f"[Counterfactual] Result: X*={x_cf.item():.2f}, Y*={y_cf.item():.2f}, Z*={z_cf.item():.2f}")
    
    delta_x = x_cf - x_obs
    delta_y = y_cf - y_obs
    delta_z = z_cf - z_obs
    print(f"[Delta] dX={delta_x.item():.2f}, dY={delta_y.item():.2f}, dZ={delta_z.item():.2f}")
    
    print("\nStep 3: Plotting results...")
    plt.figure(figsize=(10, 6))
    
    X_cpu = X_dat.cpu().numpy()
    Y_cpu = Y_dat.cpu().numpy()
    plt.scatter(X_cpu, Y_cpu, alpha=0.1, color='gray', label='Observational Data P(X,Y)')
    plt.scatter(x_obs.cpu().numpy(), y_obs.cpu().numpy(), s=150, c='blue', marker='o', label='Factual Observation')
    plt.scatter(x_cf.cpu().numpy(), y_cf.cpu().numpy(), s=150, c='red', marker='*', label='Counterfactual (do X=3)')
    plt.arrow(x_obs.item(), y_obs.item(), 
              x_cf.item() - x_obs.item(), y_cf.item() - y_obs.item(), 
              color='black', width=0.05, length_includes_head=True)
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Causal Schrödinger Bridge Result\n(Computed on {device})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = "csb_result.png"
    plt.savefig(save_path)
    print(f"Result saved to {save_path}")

if __name__ == "__main__":
    main()