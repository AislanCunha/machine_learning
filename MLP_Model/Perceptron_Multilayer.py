import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ===============================
# 1. Dispositivo (CPU / GPU)
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# ===============================
# 2. Hiperparâmetros
# ===============================
img_size = 224          # tamanho da imagem (224x224)
input_dim = img_size * img_size * 1  # Grayscale
hidden_dim = 128
learning_rate = 0.001
batch_size = 32
epochs = 10

# ===============================
# 3. Transformações
# ===============================
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5),
        std=(0.5)
    )
])

# ===============================
# 4. Datasets
# ===============================
train_dataset = datasets.ImageFolder(
    root="data_set/train",
    transform=transform
)

val_dataset = datasets.ImageFolder(
    root="data_set/val",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    root="data_set/teste",
    transform=transform
)

num_classes = len(train_dataset.classes)
print("Classes:", train_dataset.classes)

# ===============================
# 5. DataLoaders
# ===============================
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ===============================
# 6. Modelo MLP
# ===============================
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = MLP(input_dim, hidden_dim, num_classes).to(device)

# ===============================
# 7. Loss e Otimizador
# ===============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ===============================
# 8. Treinamento + Validação
# ===============================
print("\nIniciando treinamento...\n")

for epoch in range(epochs):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total

    # ---------- Validação ----------
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total

    print(f"Época [{epoch+1}/{epochs}] "
          f"| Loss: {train_loss/len(train_loader):.4f} "
          f"| Treino: {train_acc:.2f}% "
          f"| Val: {val_acc:.2f}%")

# ===============================
# 9. Teste Final
# ===============================
model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)

print(f"\nAcurácia no TESTE: {100 * test_correct / test_total:.2f}%")

# ===============================
# 10. Salvar modelo
# ===============================
torch.save(model.state_dict(), "modelo_img_mlp.pth")
print("Modelo salvo com sucesso!")

