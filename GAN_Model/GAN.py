import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os  # Necessário para gerenciar pastas e caminhos

# ==========================================
# 1. CONFIGURAÇÕES E HIPERPARÂMETROS
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.0002
z_dim = 100         
image_dim = 64 * 64 * 3  
batch_size = 32
epochs = 20
data_path = './data_set_RNN' 

# --- DEFINA AQUI O CAMINHO DA PASTA ONDE QUER SALVAR ---
pasta_destino = './GAN_Model' 
# -------------------------------------------------------

# ==========================================
# 2. CARREGAMENTO DO BANCO DE DADOS LOCAL
# ==========================================
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

try:
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Dataset carregado com {len(dataset)} imagens.")
except Exception as e:
    print(f"Erro ao carregar pasta: {e}. Verifique se a estrutura de pastas está correta.")

# ==========================================
# 3. DEFINIÇÃO DAS REDES
# ==========================================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(image_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, image_dim),
            nn.Tanh(), 
        )

    def forward(self, x):
        return self.gen(x)

disc = Discriminator().to(device)
gen = Generator(z_dim).to(device)
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()

# ==========================================
# 4. LOOP DE TREINAMENTO
# ==========================================
print("Iniciando treinamento...")
for epoch in range(epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, image_dim).to(device)
        curr_batch_size = real.shape[0]

        ### Treinar Discriminador
        noise = torch.randn(curr_batch_size, z_dim).to(device)
        fake = gen(noise)
        
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        
        disc_fake = disc(fake.detach()).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward()
        opt_disc.step()

        ### Treinar Gerador
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

    if (epoch + 1) % 10 == 0:
        print(f"Época [{epoch+1}/{epochs}] | Loss D: {lossD:.4f}, Loss G: {lossG:.4f}")

# ==========================================
# 5. EXIBIR RESULTADO
# ==========================================
gen.eval()
with torch.no_grad():
    test_noise = torch.randn(1, z_dim).to(device)
    generated = gen(test_noise).cpu().view(3, 64, 64)
    generated = (generated * 0.5) + 0.5
    
    plt.imshow(generated.permute(1, 2, 0)) 
    plt.title(f"Imagem Gerada - Época {epochs}")
    plt.axis('off')
    plt.show()

# ==========================================
# 6. SALVAMENTO EM PASTA ESPECÍFICA (.pth)
# ==========================================

# Cria a pasta caso ela não exista
if not os.path.exists(pasta_destino):
    os.makedirs(pasta_destino)

# Define os caminhos completos
caminho_gen = os.path.join(pasta_destino, "gerador_gan_2025.pth")
caminho_disc = os.path.join(pasta_destino, "discriminador_gan_2025.pth")

# Salva os estados dos modelos
torch.save(gen.state_dict(), caminho_gen)
torch.save(disc.state_dict(), caminho_disc)

print(f"\nSucesso! Modelos salvos em: {os.path.abspath(pasta_destino)}")

