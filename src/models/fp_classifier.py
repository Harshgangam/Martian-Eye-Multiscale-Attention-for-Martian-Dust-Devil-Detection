import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class HardNegativeFalsePositiveClassifier(nn.Module):
    """
    Novelty 9: False-Positive Classifier / Candidate Verifier.
    Stage 2 classifier trained on cropped ROI candidate patches to distinguish 
    true Martian dust-devil atmospheric columns from terrain hard negatives 
    (craters, mesa edges, shadows, dunes, and camera artifacts).
    """
    def __init__(self, pretrained=True):
        super(HardNegativeFalsePositiveClassifier, self).__init__()
        # Compact efficient backbone for patch verification
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(576, 128),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 6) # Outputs probabilities for: True Dust Devil, Crater Rim, Ridge/Shadow, Dune Ripple, Camera Artifact, Haze/Atmospheric Noise
        )

    def forward(self, patch):
        feat = self.features(patch)
        pooled = self.avgpool(feat).flatten(1)
        logits = self.classifier(pooled)
        probs = F.softmax(logits, dim=-1)
        
        # True dust devil confidence (class index 0)
        true_dust_devil_score = probs[:, 0]
        # Max hard negative score (classes 1 to 5)
        hard_negative_score, hard_neg_type = torch.max(probs[:, 1:], dim=-1)
        
        labels_map = ["True Dust Devil", "Crater Rim", "Ridge/Shadow", "Dune Ripple", "Camera Artifact", "Atmospheric Haze"]
        
        return {
            'logits': logits,
            'probs': probs,
            'verification_score': true_dust_devil_score,
            'hard_negative_score': hard_negative_score,
            'predicted_class_id': torch.argmax(probs, dim=-1),
            'predicted_class_name': [labels_map[i.item()] for i in torch.argmax(probs, dim=-1)]
        }

if __name__ == '__main__':
    verifier = HardNegativeFalsePositiveClassifier()
    patch = torch.randn(4, 3, 128, 128)
    res = verifier(patch)
    print("Verification Scores:", res['verification_score'])
    print("Predicted Classes:", res['predicted_class_name'])
