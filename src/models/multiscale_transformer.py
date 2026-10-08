import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MultiscaleSpatialAttention(nn.Module):
    """
    Multiscale Attention Transformer block that cross-attends features 
    across multi-level CNN feature maps to capture global contextual patterns 
    and faint dust-devil plumes against complex Martian terrain.
    """
    def __init__(self, in_channels=256, num_heads=4):
        super(MultiscaleSpatialAttention, self).__init__()
        self.num_heads = num_heads
        self.scale = (in_channels // num_heads) ** -0.5
        
        self.q_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.norm = nn.GroupNorm(8, in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q_conv(x).view(B, self.num_heads, C // self.num_heads, H * W).permute(0, 1, 3, 2)
        k = self.k_conv(x).view(B, self.num_heads, C // self.num_heads, H * W)
        v = self.v_conv(x).view(B, self.num_heads, C // self.num_heads, H * W).permute(0, 1, 3, 2)
        
        attn = torch.matmul(q, k) * self.scale
        attn_weights = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn_weights, v).permute(0, 1, 3, 2).contiguous().view(B, C, H, W)
        out = self.proj(out)
        out = self.norm(x + out)
        return out, attn_weights

class MultiscaleAttentionDetector(nn.Module):
    """
    Complete Architecture:
    1. CNN Backbone (ResNet feature pyramid)
    2. Multiscale Attention Transformer
    3. Detection & Bounding Box Regression Head
    """
    def __init__(self, pretrained=True, dropout=0.2):
        super(MultiscaleAttentionDetector, self).__init__()
        
        # Backbone
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1 # 64 ch
        self.layer2 = resnet.layer2 # 128 ch
        self.layer3 = resnet.layer3 # 256 ch
        self.layer4 = resnet.layer4 # 512 ch
        
        # Lateral reduction to 256
        self.reduce3 = nn.Conv2d(256, 256, kernel_size=1)
        self.reduce4 = nn.Conv2d(512, 256, kernel_size=1)
        
        # Multiscale Attention Transformer
        self.attention_transformer = MultiscaleSpatialAttention(in_channels=256, num_heads=4)
        
        # Feature Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        
        # Classifier Head (Dust Devil Objectness)
        self.cls_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 1)
        )
        
        # Bounding Box Head [x1, y1, x2, y2]
        self.bbox_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 4),
            nn.Sigmoid() # Scale to [0, 1]
        )
        
        # Save last attention map for XAI visualization
        self.last_attn_weights = None

    def forward(self, x):
        # Feature extraction
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        
        # Multiscale feature fusion
        feat3 = self.reduce3(c3)
        feat4 = F.interpolate(self.reduce4(c4), size=feat3.shape[2:], mode='bilinear', align_corners=False)
        fused = feat3 + feat4
        
        # Multiscale Attention Transformer
        attn_out, attn_weights = self.attention_transformer(fused)
        self.last_attn_weights = attn_weights
        
        # Global Pooling
        pooled = self.global_pool(attn_out).view(attn_out.size(0), -1)
        pooled = self.dropout(pooled)
        
        cls_logits = self.cls_head(pooled).squeeze(-1)
        bbox_pred = self.bbox_head(pooled)
        
        return {
            'logits': cls_logits,
            'probs': torch.sigmoid(cls_logits),
            'bboxes': bbox_pred,
            'features': fused,
            'attn_weights': attn_weights
        }

if __name__ == '__main__':
    model = MultiscaleAttentionDetector()
    dummy_input = torch.randn(2, 3, 416, 416)
    out = model(dummy_input)
    print("Logits shape:", out['logits'].shape)
    print("Probs shape:", out['probs'].shape)
    print("BBoxes shape:", out['bboxes'].shape)
    print("Attn shape:", out['attn_weights'].shape)
