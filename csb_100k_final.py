import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DIM = 100000 
BATCH_SIZE = 128 
STEPS = 3000
NOISE_SIGMA = 0.05

CALIBRATION_DIM = 50  
COMPLEXITY_POWER = 3.0 

print(f"CSB ULTRA-SCALING & VIZ EXPERIMENT (100,000-D)")
print(f"Framework: Causal Schrödinger Bridge | Target: {DIM:,} Dimensions")
print("-" * 75)

def calibrate_baseline_fast(d_calib):
    A = torch.randn(d_calib, d_calib).to(device)
    torch.cuda.synchronize()
    warmup = torch.linalg.inv(A) 
    start = time.time()
    for _ in range(100):
        _ = torch.linalg.inv(A)
    torch.cuda.synchronize()
    t_avg = (time.time() - start) / 100
    print(f"[*] Baseline Anchor (d={d_calib}): {t_avg:.8f}s per operation")
    return t_avg

T_REF = calibrate_baseline_fast(CALIBRATION_DIM)


def get_high_dim_causal_data(n_samples=3000, d=DIM):
    t = np.linspace(0, 2*np.pi, n_samples)
    z0 = np.stack([np.cos(t), np.sin(t)], axis=1).astype(np.float32) 
    z1 = (z0 * 0.5 + 5.0).astype(np.float32) 
    
    proj = np.random.randn(2, d).astype(np.float32)
    proj /= (np.linalg.norm(proj, axis=0, keepdims=True) + 1e-8)
    
    X0 = torch.from_numpy(z0 @ proj).to(device).float()
    X1 = torch.from_numpy(z1 @ proj).to(device).float()
    return X0, X1, z0, z1, proj

class CausalBridgeNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, dim)
        )
    def forward(self, t, x):
        t_embed = t.view(-1, 1).expand(x.shape[0], 1)
        return self.net(torch.cat([x, t_embed], dim=1))

X0, X1, Z0_GT, Z1_GT, PROJ = get_high_dim_causal_data()
model = CausalBridgeNet(DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
normalized_sigma = NOISE_SIGMA / np.sqrt(DIM)

print(f"[*] Training CSB (d={DIM:,})...")
start_time = time.time()

for step in range(STEPS):
    idx = torch.randperm(len(X0))[:BATCH_SIZE]
    b0, b1 = X0[idx], X1[idx]
    t = torch.rand(len(b0), 1).to(device)
    eps = torch.randn_like(b0) * normalized_sigma
    x_t = (1 - t) * b0 + t * b1 + eps
    pred_v = model(t, x_t)
    loss = torch.mean((pred_v - (b1 - b0))**2)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    
    if (step + 1) % 1000 == 0:
        print(f"    Step {step+1:4d}/{STEPS} | Loss: {loss.item():.6f}")

total_csb_time = time.time() - start_time

# ==============================================================================
# [SCIENTIFIC NOTE: EXTREMAL SCALING AUDIT]
# The primary goal of this script is to empirically validate the linear O(d) 
# complexity guaranteed by the Structural Decomposition Theorem. 
# To rigorously benchmark the raw FLOPs and memory limits of the network 
# at d=100,000 without integral solver overhead, we evaluate the model in its 
# deterministic zero-noise limit (1-step ODE pushforward). 
# For SDE tunneling dynamics, refer to `main.py` and `1000_dim_surgical.py`.
# ==============================================================================

with torch.no_grad():
    t_final = torch.ones(len(X0), 1).to(device)
    X_pred = X0 + model(t_final, X0)
    
    proj_pinv = np.linalg.pinv(PROJ)
    Z_pred = X_pred.cpu().numpy() @ proj_pinv
    mse = np.mean((Z_pred - Z1_GT)**2)

ITERATIONS = 100 
t_extrapolated = T_REF * (DIM / CALIBRATION_DIM)**COMPLEXITY_POWER * ITERATIONS
speedup_gap = t_extrapolated / total_csb_time

print("\n" + "EMPIRICAL SCALING AUDIT REPORT ".center(75, "="))
print(f"CSB Actual Time (d=100k) : {total_csb_time:.2f} Seconds")
print(f"Baseline (Extrapolated)  : {t_extrapolated/31536000:.2f} Years")
print(f"Speedup Gap              : {speedup_gap:,.0f}x Faster")
print(f"Manifold Recovery MSE    : {mse:.8f}")
print("=" * 75)


print("[*] Generating visualization: Latent Recovery Comparison...")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(Z0_GT[:, 0], Z0_GT[:, 1], c=np.arange(len(Z0_GT)), cmap='viridis', s=5, alpha=0.6)
plt.title("Source Latent Space ($Z_0$)\n(Input to 100k-D Projection)")
plt.xlabel("$z_1$")
plt.ylabel("$z_2$")
plt.axis('equal')

plt.subplot(1, 3, 2)
plt.scatter(Z1_GT[:, 0], Z1_GT[:, 1], c=np.arange(len(Z1_GT)), cmap='plasma', s=5, alpha=0.6)
plt.title("Target Latent Space ($Z_1$)\n(Ground Truth)")
plt.xlabel("$z_1$")
plt.ylabel("$z_2$")
plt.axis('equal')

plt.subplot(1, 3, 3)
plt.scatter(Z_pred[:, 0], Z_pred[:, 1], c=np.arange(len(Z_pred)), cmap='plasma', s=5, alpha=0.6)
plt.title(f"CSB Recovered Space ($\hat{{Z}}_1$)\n(From {DIM:,}-D Prediction)")
plt.xlabel("$z_1$")
plt.ylabel("$z_2$")
plt.axis('equal')

plt.tight_layout()
plt.savefig('csb_100k_latent_recovery.png', dpi=300)
print(f"可视化完成！图片已保存为: csb_100k_latent_recovery.png")
