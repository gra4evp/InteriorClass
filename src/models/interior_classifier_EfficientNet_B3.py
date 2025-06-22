from torch import nn
import timm  # Библиотека для современных архитектур


class InteriorClassifier(nn.Module):
    """Модель для классификации интерьеров"""
    
    def __init__(self, num_classes: int = 8, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b3", 
            pretrained=pretrained,
            num_classes=0  # Без финального слоя
        )
        self.head = nn.Sequential(
            nn.Linear(1536, 512),  # 1536 - выходной размер EfficientNet-B3
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
