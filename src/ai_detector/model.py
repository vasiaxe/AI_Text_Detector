import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class DebertaStylometryFusion(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_style_features: int,
        num_labels: int = 2,
        dropout: float = 0.2,
        hidden_dim: int = 256
    ):
        super().__init__()

        self.deberta = AutoModel.from_pretrained(model_name)
        deberta_hidden_size = self.deberta.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(deberta_hidden_size + num_style_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, input_ids, attention_mask, style_features):
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_embedding = outputs.last_hidden_state[:, 0, :]
        fused_features = torch.cat([cls_embedding, style_features], dim=1)

        return self.classifier(fused_features)


def load_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


def build_model(config: dict) -> DebertaStylometryFusion:
    return DebertaStylometryFusion(
        model_name=config['model_name'],
        num_style_features=config['style_feature_dim'],
        num_labels=config['num_labels'],
        dropout=config.get('dropout', 0.2),
        hidden_dim=config.get('hidden_dim', 256)
    )


def build_optimizer(model: DebertaStylometryFusion, config: dict):
    return torch.optim.AdamW(
        [
            {
                'params': model.deberta.parameters(),
                'lr': config['deberta_learning_rate']
            },
            {
                'params': model.classifier.parameters(),
                'lr': config['classifier_learning_rate']
            }
        ],
        eps=config.get('adam_epsilon', 1e-6),
        weight_decay=config.get('weight_decay', 0)
    )