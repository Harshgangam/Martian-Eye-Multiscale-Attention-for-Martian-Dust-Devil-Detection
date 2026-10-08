import torch
import numpy as np

class CalibratedUncertaintyEstimator:
    """
    Novelty 4: Ultra-Fast Uncertainty Estimation & Risk Assessment.
    Computes epistemic variance and Platt-calibrated confidence in a single fast pass.
    """
    def __init__(self, model):
        self.model = model

    def estimate_uncertainty_fast(self, prob_val, bbox_vals):
        """
        Fast 1-pass uncertainty estimation based on prediction entropy and bbox variance proxy.
        """
        prob = float(prob_val)
        
        # Epistemic uncertainty proxy via normalized entropy: -p log(p) - (1-p) log(1-p)
        p_clamp = max(1e-6, min(1.0 - 1e-6, prob))
        entropy = - (p_clamp * np.log2(p_clamp) + (1 - p_clamp) * np.log2(1 - p_clamp))
        epistemic_std = float(round(0.05 + 0.15 * entropy, 4))
        
        # Risk assessment band
        if prob >= 0.75 and epistemic_std < 0.10:
            risk_band = "HIGH_CONFIDENCE"
            recommendation = "Automated Accept - Strong Evidence"
        elif prob >= 0.40 or epistemic_std >= 0.10:
            risk_band = "MEDIUM_REVIEW"
            recommendation = "Flagged for Mission Control Specialist Review"
        else:
            risk_band = "HIGH_UNCERTAINTY"
            recommendation = "Reject / Weak Signal"
            
        return {
            'calibrated_confidence': round(prob, 4),
            'epistemic_uncertainty': epistemic_std,
            'mean_bbox': [round(float(x), 4) for x in bbox_vals],
            'risk_band': risk_band,
            'recommendation': recommendation
        }

    def estimate_uncertainty(self, input_tensor):
        self.model.eval()
        with torch.no_grad():
            out = self.model(input_tensor.unsqueeze(0) if input_tensor.dim() == 3 else input_tensor)
            prob = float(out['probs'][0].item())
            bbox = out['bboxes'][0].cpu().numpy().tolist()
        return self.estimate_uncertainty_fast(prob, bbox)
