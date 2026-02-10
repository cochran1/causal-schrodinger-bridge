import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"MNIST Experiment on: {device}")

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

def train(model, dataloader, epochs=20):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    print("Training MNIST Bridge...")
    
    for epoch in range(epochs):
        total_loss = 0
        for t_val, x_val in dataloader:
            b_size = x_val.shape[0]
            t = torch.rand(b_size, 1).to(device)
            x0 = torch.randn_like(x_val).to(device)
            x1 = x_val.to(device)
            context = t_val.to(device) 
            
            x_t = (1 - t) * x0 + t * x1
            target_v = x1 - x0
            
            pred_v = model(t, x_t, context)
            loss = torch.mean((pred_v - target_v)**2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"  Epoch {epoch+1}: Loss {total_loss/len(dataloader):.4f}")

@torch.no_grad()
def ode_solve(model, x0, context, reverse=False):
    steps = 50
    dt = 1.0 / steps
    if reverse: dt = -dt; t_grid = torch.linspace(1.0, 0.0, steps+1).to(device)
    else: t_grid = torch.linspace(0.0, 1.0, steps+1).to(device)
    
    x = x0
    for i in range(steps):
        t = t_grid[i].view(1, 1).repeat(x.shape[0], 1)
        v = model(t, x, context)
        x = x + v * dt
    return x

def main():
    dataset = get_mnist_data()
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    model_X = ConditionalVectorField(784, 1)
    train(model_X, loader, epochs=30) 
    
    print("Performing Intervention...")
    
    all_t, all_x = dataset.tensors
    idx = torch.argmin(all_t)
    x_obs = all_x[idx:idx+1].to(device)
    t_obs = all_t[idx:idx+1].to(device)
    
    print(f"Fact Thickness: {t_obs.item():.2f}")
    
    u_x = ode_solve(model_X, x_obs, context=t_obs, reverse=True)
    
    t_intervened = torch.tensor([[2.5]]).to(device) 
    
    x_cf = ode_solve(model_X, u_x, context=t_intervened, reverse=False)
    
    plt.figure(figsize=(8, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(x_obs.cpu().view(28, 28).numpy(), cmap='gray')
    plt.title(f"Fact (Thin)\nT={t_obs.item():.2f}")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(u_x.cpu().view(28, 28).numpy(), cmap='gray')
    plt.title("Abducted Latent\n(Style/Id)")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(x_cf.cpu().view(28, 28).numpy(), cmap='gray')
    plt.title(f"Counterfactual (Thick)\ndo(T={t_intervened.item():.1f})")
    plt.axis('off')
    
    plt.savefig('mnist_counterfactual.png')
    print("Saved mnist_counterfactual.png")

if __name__ == "__main__":
    main()