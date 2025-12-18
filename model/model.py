from torch import nn
import torch
from brainlm_mae.modeling_brainlm import BrainLMForPretraining
from torchvision.models import resnet18

class MultimodalfMRI(nn.Module):
    def __init__(self, brain_lm_path, channel_structure_features):
        super(MultimodalfMRI, self).__init__()
        
        # Load pretrained BrainLM model as time series encoder
        ts_encoder = BrainLMForPretraining.from_pretrained(brain_lm_path)
        for param in ts_encoder.parameters():
            param.requires_grad = False
        ts_encoder.vit.embeddings.mask_ratio = 0.0
        ts_encoder.vit.embeddings.config.mask_ratio = 0.0
        self.ts_encoder = ts_encoder.vit
        # Functional connectivity encoder
        fc_encoder = resnet18(pretrained=False)
        fc_encoder.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_encoder.fc.in_features, 256),
        )
        self.fc_encoder = fc_encoder
        # Structure features encoder
        if channel_structure_features == -1: # Set this to -1 to use all structure features without linear layer
            channel_structure_features = 45
        self.linear_structure = nn.Linear(45, channel_structure_features) if channel_structure_features > 0 else None
        self.feature_channels = 512 + channel_structure_features # 256 TS + 256 FC + structure features
        self.channel_structure_features = channel_structure_features
        # Predictor
        self.fc = nn.Linear(self.feature_channels, 2)

        

    def forward(self, example):
        signal_vectors = example["signal_vectors"].to(self.ts_encoder.device)
        xyz_vectors = example["xyz_vectors"].to(self.ts_encoder.device)
        structure_features = example["structure_feature"].to(self.ts_encoder.device)
        feature = []
        # Time series features
        self.ts_encoder.eval() # Set TS encoder to eval mode
        with torch.no_grad():
            ts_feature = self.ts_encoder(
                signal_vectors=signal_vectors,
                xyz_vectors=xyz_vectors,
                output_attentions=True,
                output_hidden_states=True
            ).last_hidden_state[:,0,:]
        feature.append(ts_feature)
        # Functional connectivity features
        fc_input = torch.bmm(signal_vectors, signal_vectors.permute(0, 2, 1))
        fc_input = fc_input.unsqueeze(1).repeat(1, 3, 1, 1)
        feature.append(self.fc_encoder(fc_input).squeeze(-1).squeeze(-1))
        # Structure features
        if self.channel_structure_features > 0:
            feature.append(self.linear_structure(structure_features))
        elif self.channel_structure_features == -1:
            feature.append(structure_features)
        
        # Concatenate all features to obtain multimodal embedding
        out = torch.concat(feature, dim=1)
        # Final classification layer
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out