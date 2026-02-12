import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Quantitative Experiment Running on: {device}")
plt.switch_backend('Agg')

def get_mnist_data(batch_size=256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    idx = dataset.targets == 3
    data = dataset.data[idx].float() / 255.0 
    data = (data - 0.5) / 0.5 
    X = data.view(data.shape[0], -1).to(device)
    
    T = X.mean(dim=1, keepdim=True)
    T_mean, T_std = T.mean(), T.std()
    T = (T - T_mean) / T_std
    
    full_dataset = TensorDataset(T, X)
    
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_ds, test_ds = random_split(full_dataset, [train_size, test_size])
    
    return train_ds, test_ds, T_mean, T_std

def calc_ssim(img1_batch, img2_batch):
    scores = []
    b1 = img1_batch.cpu().detach().numpy().reshape(-1, 28, 28)
    b2 = img2_batch.cpu().detach().numpy().reshape(-1, 28, 28)
    for i in range(b1.shape[0]):
        score = ssim(b1[i], b2[i], data_range=2.0)
        scores.append(score)
    return np.mean(scores)

class ConditionalVectorField(nn.Module):
    def __init__(self, dim_x, dim_context):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_x + 1 + dim_context, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, dim_x)
        ).to(device)
    
    def forward(self, t, x, context):
        inp = torch.cat([x, t, context], dim=1)
        return self.net(inp)

@torch.no_grad()
def solve_csb_sde(model, x0, context, steps=50, diffusion_scale=0.0, reverse=False):
    dt = 1.0 / steps
    
    if reverse:
        dt = -dt
        t_grid = torch.linspace(1.0, 0.0, steps+1).to(device)
    else:
        t_grid = torch.linspace(0.0, 1.0, steps+1).to(device)
    
    x = x0.clone()
    for i in range(steps):
        t = t_grid[i].view(1, 1).repeat(x.shape[0], 1)
        v = model(t, x, context)
        
        if not reverse and diffusion_scale > 0:
            brownian_noise = torch.randn_like(x) * np.sqrt(abs(dt)) * diffusion_scale
            x = x + v * dt + brownian_noise
        else:
            x = x + v * dt
            
    return x

def train_model(train_loader, train_sigma=0.1):
    model = ConditionalVectorField(784, 1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    
    print(f"Step 1: Training Model (P(X|T)) with Robust CFM (sigma={train_sigma})...")
    for epoch in range(15): 
        total_loss = 0
        for t_val, x_val in train_loader:
            b_size = x_val.shape[0]
            t = torch.rand(b_size, 1).to(device)
            x0 = torch.randn_like(x_val).to(device)
            x1 = x_val.to(device)
            context = t_val.to(device)
            
            noise = torch.randn_like(x_val).to(device) * train_sigma
            x_t = (1 - t) * x0 + t * x1 + noise
            
            target_v = x1 - x0
            pred_v = model(t, x_t, context)
            loss = torch.mean((pred_v - target_v)**2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 5 == 0:
            print(f"  Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f}")
    return model

def evaluate():
    train_ds, test_ds, T_mean, T_std = get_mnist_data()
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=200, shuffle=False)
    
    model = train_model(train_loader, train_sigma=0.1)
    
    print("\nStep 2: Quantitative Evaluation on Test Set...")
    
    target_t_val = 2.5
    
    metrics = {
        'Baseline': {'mae': [], 'ssim': [], 'mse': []},
        'CSB (Ours)': {'mae': [], 'ssim': [], 'mse': []}
    }
    
    for t_obs, x_obs in test_loader:
        t_obs = t_obs.to(device)
        x_obs = x_obs.to(device)
        
        t_target = torch.full_like(t_obs, target_t_val)
        
        z_rand = torch.randn_like(x_obs)
        x_baseline = solve_csb_sde(model, z_rand, context=t_target, reverse=False, diffusion_scale=0.0)
        
        u_x = solve_csb_sde(model, x_obs, context=t_obs, reverse=True, diffusion_scale=0.0)
        
        x_csb = solve_csb_sde(model, u_x, context=t_target, reverse=False, diffusion_scale=0.5)
        
        def calc_thickness(img_batch):
            return (img_batch.mean(dim=1, keepdim=True) - T_mean) / T_std
            
        thick_base = calc_thickness(x_baseline)
        thick_csb = calc_thickness(x_csb)
        
        metrics['Baseline']['mae'].append(torch.abs(thick_base - target_t_val).mean().item())
        metrics['CSB (Ours)']['mae'].append(torch.abs(thick_csb - target_t_val).mean().item())
        
        metrics['Baseline']['ssim'].append(calc_ssim(x_obs, x_baseline))
        metrics['CSB (Ours)']['ssim'].append(calc_ssim(x_obs, x_csb))
        
        metrics['Baseline']['mse'].append(nn.functional.mse_loss(x_obs, x_baseline).item())
        metrics['CSB (Ours)']['mse'].append(nn.functional.mse_loss(x_obs, x_csb).item())

    results = {}
    for method in metrics:
        results[method] = {k: np.mean(v) for k, v in metrics[method].items()}
    
    df_res = pd.DataFrame(results).T
    
    print("\n" + "="*50)
    print("QUANTITATIVE RESULTS (Morpho-MNIST Intervention)")
    print("Task: do(Thickness = 2.5 sigma)")
    print("="*50)
    print(df_res)
    print("="*50)
    
    print("\n[LaTeX Table Code]")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"\textbf{Method} & \textbf{Thickness MAE} ($\downarrow$) & \textbf{Identity SSIM} ($\uparrow$) & \textbf{L2 Distance} ($\downarrow$) \\")
    print(r"\midrule")
    for method, row in df_res.iterrows():
        print(r"{} & {:.4f} & {:.4f} & {:.4f} \\".format(method, row['mae'], row['ssim'], row['mse']))
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Quantitative comparison on Morpho-MNIST. MAE measures intervention success. SSIM measures structural identity preservation.}")
    print(r"\label{tab:mnist_quant}")
    print(r"\end{table}")
    
    visualize_metrics(df_res)

def visualize_metrics(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics_map = [
        ('mae', r'Thickness Error (MAE) $\downarrow$', 'green'),
        ('ssim', r'Structural Similarity (SSIM) $\uparrow$', 'blue'),
        ('mse', r'L2 Distance (Minimal Action) $\downarrow$', 'purple')
    ]
    
    for i, (col, title, color) in enumerate(metrics_map):
        ax = axes[i]
        df[col].plot(kind='bar', ax=ax, color=color, alpha=0.7, rot=0)
        ax.set_title(title, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points')

    plt.suptitle("Quantitative Evaluation: CSB vs Baseline", fontsize=16)
    plt.tight_layout()
    plt.savefig('mnist_quantitative_metrics.png')
    print("\nMetric plot saved to mnist_quantitative_metrics.png")

if __name__ == "__main__":
    evaluate()
