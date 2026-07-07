import torch.nn as nn
from torchvision import models

class ReIDNet(nn.Module):
    def __init__(self, embedding_dim: int, head: str = "mlp"):
        super(ReIDNet, self).__init__()
        # Load a pre-trained ResNet-50 model
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Replace the final classification layer with our embedding head
        num_ftrs = self.base_model.fc.in_features
        if head == "linear":
            self.base_model.fc = nn.Linear(num_ftrs, embedding_dim)
        elif head == "mlp":
            # Matches historical checkpoints in this repo (state_dict keys: base_model.fc.0, .1, .4, .5 ...)
            self.base_model.fc = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(512, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
            )
        else:
            raise ValueError(f"Unknown head type: {head} (expected 'linear' or 'mlp')")

    def forward(self, x):
        x = self.base_model(x)
        # L2-normalize the embeddings, which is a common practice in Re-ID
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


def infer_reid_head_from_state_dict(state_dict: dict) -> str:
    """
    Infer whether a checkpoint expects a linear or MLP head.
    """
    if any(k.startswith("base_model.fc.0.") for k in state_dict.keys()):
        return "mlp"
    if any(k.startswith("base_model.fc.weight") for k in state_dict.keys()):
        return "linear"
    # Default to MLP (repo's common case), but allow strict=False loads elsewhere if needed.
    return "mlp"
