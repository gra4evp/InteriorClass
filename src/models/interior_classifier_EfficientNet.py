from typing import Literal
from torch import nn
import timm
from src.schemas import HeadConfig, ModelConfig


class InteriorClassifier(nn.Module):
    """
    Модель для классификации интерьеров
    """
    
    def __init__(
        self,
        num_classes: int = 8,
        backbone_name: str = 'efficientnet_b3',
        pretrained: bool = True,
        use_head: bool = False,
        head_hidden_dim: int = 512,
        head_dropout: float = 0.3,
        head_activation: Literal['relu', 'gelu'] = 'relu'
    ):
        super().__init__()

        self.num_classes = num_classes
        self.backbone_name = backbone_name
        self.pretrained = pretrained

        self.use_head = use_head
        self.head_hidden_dim = head_hidden_dim
        self.head_dropout = head_dropout
        self.head_activation = head_activation

        self.backbone = timm.create_model(
            backbone_name, 
            pretrained=pretrained,
            num_classes=0
        )

        self.feature_dim = self.backbone.num_features
        if use_head:
            # Полноценная голова
            activation = nn.Identity()
            if head_activation == 'relu':
                activation = nn.ReLU
            elif head_activation == 'gelu':
                activation == nn.GELU()
            
            self.head = nn.Sequential(
                nn.Linear(self.feature_dim, head_hidden_dim),
                activation,
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden_dim, num_classes)
            )
        else:
            # Просто заменяем финальный классификатор
            self.head = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

    def to_config(self) -> ModelConfig:
        head = None
        if self.use_head:
            head = HeadConfig(
                hidden_dim=self.head_hidden_dim,
                dropout=self.head_dropout,
                activation=self.head_activation
            )
        return ModelConfig(
            backbone_name=self.backbone_name,
            num_classes=self.num_classes,
            pretrained=self.pretrained,
            head=head
        )

    @classmethod
    def from_config(cls, config: ModelConfig) -> 'InteriorClassifier':
        head = config.head
        if head is not None:
            return cls(
                num_classes=config.num_classes,
                backbone_name=config.backbone_name,
                pretrained=config.pretrained,
                head_hidden_dim=config.head.hidden_dim,
                head_dropout=config.head.dropout,
                head_activation=config.head.activation
            )
        
        return cls(
            num_classes=config.num_classes,
            backbone_name=config.backbone_name,
            pretrained=config.pretrained,
        )
