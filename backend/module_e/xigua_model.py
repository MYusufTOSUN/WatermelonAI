import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os

class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()
        self.lstm = nn.LSTM(input_size=100, hidden_size=128, batch_first=True)
        
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        self.fc1 = nn.Linear(128 + 2048, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, audio, image):
        lstm_out, _ = self.lstm(audio)
        audio_feat = lstm_out[:, -1, :] 
        img_feat = self.resnet(image)
        img_feat = img_feat.view(img_feat.size(0), -1) 
        merged = torch.cat((audio_feat, img_feat), dim=1) 
        out = self.fc1(merged)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class XiguaPytorchModel:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FusionModel().to(self.device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            self.is_loaded = True
            print(f"[Xigua] PyTorch modeli başarıyla yüklendi: {model_path}")
        else:
            self.is_loaded = False
            print(f"[Xigua] HATA: PyTorch model dosyası bulunamadı -> {model_path}")
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_embedding(self, audio_data: np.ndarray, image_bgr: np.ndarray) -> np.ndarray:
        """
        Xigua FusionModel'in fc1 katmanindan 64-D embedding cikarir.
        Transfer learning icin: ResNet50 + LSTM ortak temsilini downstream
        siniflandiriciya feature olarak verir.

        Returns: (64,) float32 vektor
        """
        if not self.is_loaded:
            return np.zeros(64, dtype=np.float32)

        audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
        if len(audio_tensor.shape) > 1:
            audio_tensor = audio_tensor[0]
        if audio_tensor.shape[0] >= 16000:
            audio_tensor = audio_tensor[:16000]
        else:
            padding = torch.zeros(16000 - audio_tensor.shape[0])
            audio_tensor = torch.cat((audio_tensor, padding))
        audio_input = audio_tensor.view(1, 160, 100).to(self.device)

        rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_image)
        image_input = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            lstm_out, _ = self.model.lstm(audio_input)
            audio_feat = lstm_out[:, -1, :]
            img_feat = self.model.resnet(image_input)
            img_feat = img_feat.view(img_feat.size(0), -1)
            merged = torch.cat((audio_feat, img_feat), dim=1)
            embedding = self.model.relu(self.model.fc1(merged))

        return embedding.cpu().numpy().flatten().astype(np.float32)

    def predict(self, audio_data: np.ndarray, image_bgr: np.ndarray) -> float:
        """
        PyTorch üzerinden Late Fusion tahmini yapar.
        Çıktı bir regresyon skoru (0 ile 2 arası bir label/brix skoru)
        """
        if not self.is_loaded:
            return -1.0
            
        # Audio preprocessing
        audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
        if len(audio_tensor.shape) > 1:
            audio_tensor = audio_tensor[0]
            
        if audio_tensor.shape[0] >= 16000:
            audio_tensor = audio_tensor[:16000]
        else:
            padding = torch.zeros(16000 - audio_tensor.shape[0])
            audio_tensor = torch.cat((audio_tensor, padding))
            
        audio_input = audio_tensor.view(1, 160, 100).to(self.device)
        
        # Image preprocessing
        rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_image)
        image_input = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(audio_input, image_input)
            
        return float(output.item())
