import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend('Agg')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

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
    T = (T - T.mean()) / T.std()
    
    return TensorDataset(T, X) 

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

def train(model, dataloader, epochs=20, train_sigma=0.1):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for t_val, x_val in dataloader:
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
            
        print(f"Epoch {epoch+1}: Loss {total_loss/len(dataloader):.4f}")

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

def main():
    dataset = get_mnist_data()
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    model_X = ConditionalVectorField(784, 1)
    
    train(model_X, loader, epochs=30, train_sigma=0.1) 
    
    all_t, all_x = dataset.tensors
    idx = torch.argmin(all_t) 
    x_obs = all_x[idx:idx+1].to(device)
    t_obs = all_t[idx:idx+1].to(device)
    
    u_x = solve_csb_sde(model_X, x_obs, context=t_obs, reverse=True, diffusion_scale=0.0)
    
    t_intervened = torch.tensor([[2.5]]).to(device) 
    
    x_cf = solve_csb_sde(model_X, u_x, context=t_intervened, reverse=False, diffusion_scale=0.15)
    
    plt.figure(figsize=(10, 4), dpi=300)
    
    def show_img(ax, img_tensor, title):
        img = img_tensor.cpu().view(28, 28).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

    ax1 = plt.subplot(1, 3, 1)
    show_img(ax1, x_obs, f"Factual\nIntensity={x_obs.mean():.2f}")
    
    ax2 = plt.subplot(1, 3, 2)
    show_img(ax2, u_x, "Latent")
    
    ax3 = plt.subplot(1, 3, 3)
    show_img(ax3, x_cf, f"Counterfactual")
    
    plt.tight_layout()
    plt.savefig('mnist_counterfactual_sharp.png')

if __name__ == "__main__":
    main()
