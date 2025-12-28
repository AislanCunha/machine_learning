import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

# --- CONFIGURAÇÕES ---
PASTA_DATASET = 'data_set/'       
PASTA_SALVAMENTO = 'LSTM_Model/'  
NOME_MODELO = 'modelo_lstm_imagens.pth'
EPOCHS = 10  
BATCH_SIZE = 4

if not os.path.exists(PASTA_SALVAMENTO):
    os.makedirs(PASTA_SALVAMENTO)

# --- 1. DATASET: CARREGAMENTO SEQUENCIAL ---
class ImageSequenceDataset(Dataset):
    def __init__(self, root_dir, seq_length=5, transform=None):
        self.root_dir = root_dir
        self.seq_length = seq_length
        self.transform = transform
        self.sequences = []
        self.labels = []
        
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Diretório {root_dir} não encontrado.")

        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            for seq_folder in os.listdir(cls_path):
                seq_path = os.path.join(cls_path, seq_folder)
                if os.path.isdir(seq_path):
                    images = sorted([os.path.join(seq_path, img) for img in os.listdir(seq_path) 
                                   if img.lower().endswith(('.png', '.jpg', '.jpeg'))])
                    if len(images) >= seq_length:
                        self.sequences.append(images[:seq_length])
                        self.labels.append(class_to_idx[cls])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_images = []
        for img_path in self.sequences[idx]:
            img = Image.open(img_path).convert('RGB')
            if self.transform: img = self.transform(img)
            seq_images.append(img)
        return torch.stack(seq_images), self.labels[idx]

# --- 2. MODELO CNN-LSTM ---
class ImageSequenceLSTM(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(ImageSequenceLSTM, self).__init__()
        # ResNet18 como extrator de características fixo (Transfer Learning)
        resnet = models.resnet18(weights='DEFAULT')
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        ii = x.view(batch_size * seq_len, c, h, w)
        features = self.feature_extractor(ii)
        features = features.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(features)
        return self.fc(lstm_out[:, -1, :])

# --- 3. TREINAMENTO COM LOSS E ACURÁCIA ---

transformacoes = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

try:
    train_dataset = ImageSequenceDataset(root_dir=PASTA_DATASET, seq_length=5, transform=transformacoes)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = ImageSequenceLSTM(hidden_size=256, num_classes=len(train_dataset.classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(modelo.parameters(), lr=0.001)

    print(f"Iniciando treinamento em {device}...\n")
    
    for epoch in range(EPOCHS):
        modelo.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = modelo(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Lógica de Acurácia
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Época [{epoch+1}/{EPOCHS}] - Perda: {total_loss/len(train_loader):.4f} - Acurácia: {accuracy:.2f}%")

    # Salvar o modelo no formato .pth na pasta específica
    caminho_final = os.path.join(PASTA_SALVAMENTO, NOME_MODELO)
    torch.save(modelo.state_dict(), caminho_final)
    print(f"\nTreino concluído! Modelo salvo em: {caminho_final}")

except Exception as e:
    print(f"Ocorreu um erro: {e}")

