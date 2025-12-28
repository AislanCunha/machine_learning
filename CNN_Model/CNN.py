import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import os

# 1. Configuração de Dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Preparação dos Dados (Alta Resolução em Escala de Cinza)
IMAGE_SIZE = 224  # Você pode aumentar este valor conforme sua necessidade

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # Converte para Escala de Cinza
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Padroniza o tamanho para a rede
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))         # Normalização para 1 canal
])

# Carregando datasets das suas pastas
train_dataset = datasets.ImageFolder(root="data_set/train", transform=transform)
val_dataset   = datasets.ImageFolder(root="data_set/val", transform=transform)
test_dataset  = datasets.ImageFolder(root="data_set/teste", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)

print("-" * 30)
print(f"Classes: {train_dataset.classes}")
print(f"Modo: Escala de Cinza (1 canal)")
print(f"Resolução: {IMAGE_SIZE}x{IMAGE_SIZE}")
print("-" * 30)

# 3. Arquitetura Adaptativa para Escala de Cinza
class GrayCNN(nn.Module):
    def __init__(self, num_classes):
        super(GrayCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # Entrada: 1 canal
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Reduz resolução por 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Reduz resolução por 4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Reduz resolução por 8
        )
        
        # O tamanho final após 3 MaxPools (2^3 = 8) é IMAGE_SIZE / 8
        self.feature_dim = IMAGE_SIZE // 8
        self.flatten_size = 128 * self.feature_dim * self.feature_dim
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))

model = GrayCNN(len(train_dataset.classes)).to(device)

# 4. Configuração de Treino
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 5. Loop de Treinamento e Validação
print("\nIniciando treinamento (2025)...")
epochs = 5

for epoch in range(epochs):
    model.train()
    train_loss, train_acc = 0.0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)
        train_acc += (outputs.argmax(1) == labels).sum().item()

    model.eval()
    val_loss, val_acc = 0.0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            val_acc += (outputs.argmax(1) == labels).sum().item()

    print(f"Época [{epoch+1}/{epochs}] | "
          f"Treino: Loss {train_loss/len(train_dataset):.4f}, Acc {(train_acc/len(train_dataset))*100:.2f}% | "
          f"Val: Loss {val_loss/len(val_dataset):.4f}, Acc {(val_acc/len(val_dataset))*100:.2f}%")

# 6. Salvamento
os.makedirs("CNN_Model", exist_ok=True)
torch.save(model.state_dict(), "CNN_Model/modelo_cinza_alta_res.pth")
print("\n[OK] Modelo salvo com sucesso!")

