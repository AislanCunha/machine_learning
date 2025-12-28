import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Configurações
DATA_DIR = './data_set_RNN' 
IMG_SIZE = 200  # Tamanho que você escolheu
BATCH_SIZE = 16
EPOCHS = 10

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 3. Arquitetura da RNN (Ajustada)
class ImageRNN(nn.Module):
    # Alterado input_size para 200 (ou o valor de IMG_SIZE)
    def __init__(self, input_size=200, hidden_size=128, num_layers=2):
        super(ImageRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # A LSTM agora sabe que cada linha tem 200 pixels
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.squeeze(1) # Remove canal de cor: (Batch, 200, 200)
        
        # Inicialização dos estados internos (garantindo que fiquem no mesmo device das imagens)
        device = x.device
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) 
        return self.sigmoid(out)

# 4. Treinamento
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# IMPORTANTE: Passamos o IMG_SIZE aqui para o construtor da classe
model = ImageRNN(input_size=IMG_SIZE).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Iniciando treino em: {device}")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if len(train_loader) > 0:
        print(f"Época [{epoch+1}/{EPOCHS}] - Perda: {total_loss/len(train_loader):.4f}")

# 5. Salvar
torch.save(model.state_dict(), 'meu_modelo_imagens.pth')
print("Modelo salvo como meu_modelo_imagens.pth")
